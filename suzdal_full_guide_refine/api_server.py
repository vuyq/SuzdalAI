import os
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from datetime import datetime, timedelta
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_gigachat import GigaChat, GigaChatEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential
from ddgs import DDGS
from rapidfuzz import process, fuzz
import logging
from typing import List, Tuple, Optional, Dict, Any
import uuid
import uvicorn
import json
from functools import lru_cache
import time
import re
import numpy as np

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Переменные окружения
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Модели базы данных
class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(String(100), primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_question = Column(Text, nullable=True)
    user_preferences = Column(JSON, nullable=True)
    conversation_summary = Column(Text, nullable=True)
    clarification_context = Column(JSON, nullable=True)
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String(10), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    embeddings = Column(JSON, nullable=True)
    session = relationship("ChatSession", back_populates="messages")

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class Config:
    MAX_RETRIES = 3
    REQUEST_TIMEOUT = 15
    SEARCH_RESULTS = 3
    RETRIEVER_K = 5
    CERT_PATH = os.getenv("CERT_PATH", "./cert.pem")
    CERT_URL = os.getenv("CERT_URL")
    GIGACHAT_AUTH = os.getenv("GIGACHAT_AUTH")
    CSV_DATA_URL = os.getenv(
        "CSV_DATA_URL",
        "https://raw.githubusercontent.com/vuyq/SuzdalAI/main/suzdal_full_guide_refine/attractions.csv"
    )
    TOKEN_EXPIRY_MINUTES = 25
    MEMORY_CONTEXT_SIZE = 10
    PREFERENCES_UPDATE_INTERVAL = 5
    SEMANTIC_SEARCH_K = 3

# Глобальные переменные
embedding_model = None
ai_assistant = None
documents = []
vector_store = None
document_retriever = None
app_initialized = False
token_manager = None

class GigaChatTokenManager:
    def __init__(self):
        self.access_token = None
        self.token_expires = None
        self.lock = False
        
    def get_valid_token(self) -> str:
        if self.access_token and self.token_expires and datetime.now() < self.token_expires:
            return self.access_token
        
        if self.lock:
            for _ in range(10):
                time.sleep(0.1)
                if self.access_token and datetime.now() < self.token_expires:
                    return self.access_token
        
        self.lock = True
        try:
            self.access_token = get_gigachat_token()
            self.token_expires = datetime.now() + timedelta(minutes=Config.TOKEN_EXPIRY_MINUTES)
            logger.info("Токен GigaChat успешно обновлен")
            return self.access_token
        except Exception as e:
            logger.error(f"Ошибка получения токена: {e}")
            raise
        finally:
            self.lock = False
    
    def is_token_valid(self) -> bool:
        return bool(self.access_token and self.token_expires and datetime.now() < self.token_expires)

def download_certificate():
    if Config.CERT_URL and not Path(Config.CERT_PATH).exists():
        try:
            response = requests.get(Config.CERT_URL, timeout=Config.REQUEST_TIMEOUT)
            response.raise_for_status()
            with open(Config.CERT_PATH, "wb") as f:
                f.write(response.content)
            logger.info("Сертификат успешно загружен")
        except Exception as e:
            logger.error(f"Ошибка загрузки сертификата: {e}")
            raise

@retry(stop=stop_after_attempt(Config.MAX_RETRIES), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_gigachat_token() -> str:
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': str(uuid.uuid4()),
        'Authorization': f'Basic {Config.GIGACHAT_AUTH}'
    }
    payload = {'scope': 'GIGACHAT_API_PERS'}
    response = requests.post(
        url, headers=headers, data=payload,
        verify=Config.CERT_PATH, timeout=Config.REQUEST_TIMEOUT
    )
    response.raise_for_status()
    return response.json().get("access_token")

def initialize_models() -> Tuple[GigaChatEmbeddings, GigaChat]:
    try:
        access_token = token_manager.get_valid_token()
        embedding_model = GigaChatEmbeddings(
            access_token=access_token, model="Embeddings", scope="GIGACHAT_API_PERS",
            verify_ssl_certs=bool(Config.CERT_PATH),
            ca_bundle_file=Config.CERT_PATH if Path(Config.CERT_PATH).exists() else None,
            timeout=Config.REQUEST_TIMEOUT
        )
        ai_assistant = GigaChat(
            access_token=access_token, model="GigaChat", temperature=0.2,
            verify_ssl_certs=bool(Config.CERT_PATH),
            ca_bundle_file=Config.CERT_PATH if Path(Config.CERT_PATH).exists() else None,
            timeout=Config.REQUEST_TIMEOUT, verbose=True
        )
        return embedding_model, ai_assistant
    except Exception as e:
        logger.error(f"Ошибка инициализации моделей: {e}")
        raise

def refresh_models():
    global embedding_model, ai_assistant
    try:
        access_token = token_manager.get_valid_token()
        
        if embedding_model:
            embedding_model.access_token = access_token
        
        if ai_assistant:
            ai_assistant.access_token = access_token
        else:
            ai_assistant = GigaChat(
                access_token=access_token, model="GigaChat", temperature=0.2,
                verify_ssl_certs=bool(Config.CERT_PATH),
                ca_bundle_file=Config.CERT_PATH if Path(Config.CERT_PATH).exists() else None,
                timeout=Config.REQUEST_TIMEOUT, verbose=True
            )
            
        logger.info("Модели успешно обновлены с новым токеном")
    except Exception as e:
        logger.error(f"Ошибка обновления моделей: {e}")
        raise

def load_data() -> List[Document]:
    try:
        df = pd.read_csv(Config.CSV_DATA_URL, sep=';', on_bad_lines='skip')
    except Exception:
        try:
            df = pd.read_csv(Config.CSV_DATA_URL, on_bad_lines='skip')
        except Exception as e:
            logger.error(f"Ошибка загрузки CSV: {e}")
            return []

    documents = []
    for _, row in df.iterrows():
        metadata, content = {}, []
        for col, val in row.items():
            if pd.isna(val) or str(val).strip() == '':
                continue
            col_lower = col.lower()
            if col_lower in ['name', 'title', 'название']:
                metadata['name'] = str(val)
            elif col_lower in ['type', 'тип', 'category', 'категория']:
                metadata['type'] = str(val)
            elif col_lower in ['address', 'адрес']:
                metadata['address'] = str(val)
            elif col_lower in ['price', 'цена']:
                metadata['price'] = str(val)
            elif col_lower in ['hours', 'часы']:
                metadata['hours'] = str(val)
            elif col_lower in ['description', 'описание']:
                metadata['description'] = str(val)
            else:
                content.append(f"{col}: {val}")
        if 'name' not in metadata:
            continue
        doc = Document(
            page_content="\n".join(content) if content else metadata.get('description', ''),
            metadata=metadata
        )
        documents.append(doc)
    return documents

@lru_cache(maxsize=100)
def perform_web_search(query: str) -> str:
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(f"{query} Суздаль", max_results=Config.SEARCH_RESULTS, timelimit='y'):
                results.append(f"• {r['title']}\n  {r['href']}\n  {r['body'][:200]}...")
        return "\n\n".join(results) if results else "Не найдено результатов"
    except Exception as e:
        logger.error(f"Ошибка веб-поиска: {e}")
        return "Ошибка при выполнении поиска"

def search_in_vector_store(query: str, k: int = None) -> List[Document]:
    if not vector_store or not document_retriever:
        return []
    
    try:
        k = k or Config.RETRIEVER_K
        results = vector_store.similarity_search(query, k=k)
        return results
    except Exception as e:
        logger.error(f"Ошибка поиска в векторной базе: {e}")
        return []

def fuzzy_retrieval(question: str, docs: List[Document], limit: int = 5) -> List[Document]:
    if not docs:
        return []
    
    corpus = [doc.metadata.get('name', '') for doc in docs]
    matches = process.extract(
        question,
        corpus,
        scorer=fuzz.WRatio,
        limit=limit
    )
    results = []
    for match_text, score, idx in matches:
        if score > 50:
            results.append(docs[idx])
    return results

def cosine_similarity(vec1, vec2):
    if not vec1 or not vec2:
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0

def build_gigachat_messages(db: Session, user_id: str, current_question: str) -> List[Dict[str, str]]:
    """Формируем массив сообщений для GigaChat API"""
    messages = db.query(Message).filter(Message.session_id == user_id).order_by(Message.timestamp.asc()).all()
    
    chat_history = []
    for msg in messages:
        chat_history.append({
            "role": msg.role,
            "content": msg.content
        })
    
    # Добавляем текущий вопрос
    chat_history.append({"role": "user", "content": current_question})
    
    return chat_history

TOURISM_PROMPT_TEMPLATE = """
Ты виртуальный гид по Суздалю. Учитывай контекст предыдущего диалога и предпочтения пользователя.

[Семантическая память диалога]:
{conversation_summary}

[Предпочтения пользователя]:
{user_preferences}

[Данные из базы о достопримечательностях]:
{context}

[Веб-результаты]:
{web_search}

[Текущий вопрос]:
{question}

Отвечай на русском языке.
"""
tourism_prompt = PromptTemplate.from_template(TOURISM_PROMPT_TEMPLATE)

def generate_ai_response(db: Session, user_id: str, question: str, 
                         context_docs: List[Document], web_results: str,
                         conversation_summary: str, user_preferences: Dict) -> str:
    try:
        if not token_manager.is_token_valid():
            refresh_models()

        messages = build_gigachat_messages(db, user_id, question)

        # Системный промпт
        system_prompt = tourism_prompt.format(
            question=question,
            context="\n\n".join(d.page_content for d in context_docs) if context_docs else "Нет данных в базе",
            web_search=web_results or "Нет результатов",
            conversation_summary=conversation_summary or "Нет истории",
            user_preferences=json.dumps(user_preferences, ensure_ascii=False, indent=2)
        )

        messages.insert(0, {"role": "system", "content": system_prompt})

        response = ai_assistant.invoke({"model": "GigaChat", "messages": messages})
        response_text = response.content if hasattr(response, "content") else str(response)
        return response_text

    except Exception as e:
        logger.error(f"Ошибка генерации ответа: {e}")
        return "Извините, произошла ошибка при генерации ответа."

def handle_question(db: Session, question: str, user_id: str) -> str:
    if not app_initialized:
        return "Приложение еще не инициализировано. Попробуйте позже."
    
    question = question.strip()
    if not question:
        return "Пожалуйста, задайте ваш вопрос о Суздале."

    session = db.query(ChatSession).filter(ChatSession.id == user_id).first()

    # сохраняем вопрос пользователя
    db.add(Message(session_id=user_id, role="user", content=question, timestamp=datetime.utcnow()))
    db.commit()

    user_preferences = session.user_preferences if session and session.user_preferences else {}
    conversation_summary = session.conversation_summary if session else ""

    context_docs = search_in_vector_store(question)
    if not context_docs and documents:
        context_docs = fuzzy_retrieval(question, documents, limit=Config.RETRIEVER_K)

    ai_answer = generate_ai_response(db, user_id, question, context_docs, "", conversation_summary, user_preferences)

    response = ai_answer

    db.add(Message(session_id=user_id, role="assistant", content=response, timestamp=datetime.utcnow()))
    db.commit()

    return response

def safe_init_db():
    """Безопасная инициализация базы данных без удаления таблиц"""
    from sqlalchemy import inspect, text
    
    try:
        inspector = inspect(engine)
        
        # Проверяем существование таблиц
        existing_tables = inspector.get_table_names()
        
        # Создаем chat_sessions если не существует
        if 'chat_sessions' not in existing_tables:
            with engine.connect() as conn:
                conn.execute(text("""
                    CREATE TABLE chat_sessions (
                        id VARCHAR(100) PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_question TEXT,
                        user_preferences JSONB,
                        conversation_summary TEXT,
                        clarification_context JSONB
                    )
                """))
                conn.commit()
                logger.info("Таблица chat_sessions создана")
        
        # Создаем messages если не существует
        if 'messages' not in existing_tables:
            with engine.connect() as conn:
                conn.execute(text("""
                    CREATE TABLE messages (
                        id SERIAL PRIMARY KEY,
                        session_id VARCHAR(100) REFERENCES chat_sessions(id) ON DELETE CASCADE,
                        role VARCHAR(10) NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        embeddings JSONB
                    )
                """))
                conn.commit()
                logger.info("Таблица messages создана")
        
        # БЕЗОПАСНО добавляем недостающие колонки
        try:
            if 'chat_sessions' in existing_tables:
                chat_columns = [col['name'] for col in inspector.get_columns('chat_sessions')]
                
                # Добавляем только отсутствующие колонки
                if 'user_preferences' not in chat_columns:
                    with engine.connect() as conn:
                        conn.execute(text("ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS user_preferences JSONB"))
                        conn.commit()
                        logger.info("Добавлена колонка user_preferences")
                        
                if 'conversation_summary' not in chat_columns:
                    with engine.connect() as conn:
                        conn.execute(text("ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS conversation_summary TEXT"))
                        conn.commit()
                        logger.info("Добавлена колонка conversation_summary")
                        
                if 'clarification_context' not in chat_columns:
                    with engine.connect() as conn:
                        conn.execute(text("ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS clarification_context JSONB"))
                        conn.commit()
                        logger.info("Добавлена колонка clarification_context")
            
            if 'messages' in existing_tables:
                messages_columns = [col['name'] for col in inspector.get_columns('messages')]
                if 'embeddings' not in messages_columns:
                    with engine.connect() as conn:
                        conn.execute(text("ALTER TABLE messages ADD COLUMN IF NOT EXISTS embeddings JSONB"))
                        conn.commit()
                        logger.info("Добавлена колонка embeddings")
                        
        except Exception as e:
            logger.warning(f"Ошибка при добавлении колонок: {e}")
            # Продолжаем работу даже если есть ошибки
            
    except Exception as e:
        logger.error(f"Ошибка инициализации базы: {e}")
        # Не падаем, пытаемся работать дальше

# Инициализация базы данных
safe_init_db()

def initialize_app():
    global embedding_model, ai_assistant, documents, vector_store, document_retriever, app_initialized, token_manager
    
    try:
        download_certificate()
        token_manager = GigaChatTokenManager()
        embedding_model, ai_assistant = initialize_models()
        documents = load_data()
        
        if documents:
            vector_store = FAISS.from_documents(documents, embedding_model)
            document_retriever = vector_store.as_retriever(search_kwargs={"k": Config.RETRIEVER_K})
            logger.info(f"Приложение успешно инициализировано, загружено {len(documents)} документов")
        else:
            logger.warning("Документы не загружены, RAG будет работать в ограниченном режиме")
        
        app_initialized = True
        
    except Exception as e:
        logger.critical(f"Ошибка инициализации: {e}")
        documents = []
        app_initialized = False
        raise

initialize_app()

app = FastAPI(title="Суздаль Tourism Assistant")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class Question(BaseModel):
    question: str
    user_id: str = "default"

@app.get("/")
async def root():
    return {"message": "Suzdal Tourism Assistant API работает. Используйте /ask для вопросов."}

@app.post("/ask")
async def ask(item: Question, db: Session = Depends(get_db)):
    if not app_initialized:
        raise HTTPException(status_code=503, detail="Сервис временно недоступен. Идет инициализация.")
    
    if not item.question.strip():
        return {"answer": "Пожалуйста, задайте ваш вопрос."}
    
    try:
        response = handle_question(db, item.question, item.user_id)
        return {"answer": response}
    except Exception as e:
        logger.error(f"Ошибка обработки вопроса: {e}")
        raise HTTPException(status_code=500, detail="Ошибка обработки вопроса")

@app.get("/health")
async def health_check():
    status = "healthy" if app_initialized and documents else "degraded"
    return {
        "status": status, 
        "initialized": app_initialized,
        "timestamp": datetime.utcnow(),
        "documents_loaded": len(documents),
        "token_valid": token_manager.is_token_valid() if token_manager else False
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
