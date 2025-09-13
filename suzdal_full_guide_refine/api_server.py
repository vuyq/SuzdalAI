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
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
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

# Создание таблиц
Base.metadata.create_all(bind=engine)

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
    MAX_PLACES_TO_SHOW = 2  # Максимальное количество мест для показа
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

# Глобальные переменные
embedding_model = None
ai_assistant = None
documents = []
vector_store = None
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

def search_in_vector_store(query: str, k: int = Config.MAX_PLACES_TO_SHOW) -> List[Document]:
    if not vector_store:
        return []
    
    try:
        results = vector_store.similarity_search(query, k=k)
        return results[:Config.MAX_PLACES_TO_SHOW]  # Явно ограничиваем количество
    except Exception as e:
        logger.error(f"Ошибка поиска в векторной базе: {e}")
        return []

def fuzzy_retrieval(question: str, docs: List[Document], limit: int = Config.MAX_PLACES_TO_SHOW) -> List[Document]:
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
    return results[:Config.MAX_PLACES_TO_SHOW]  # Явно ограничиваем количество

def build_gigachat_messages(db: Session, user_id: str, current_question: str) -> List:
    """Формируем массив сообщений для GigaChat в правильном формате"""
    messages = db.query(Message).filter(Message.session_id == user_id).order_by(Message.timestamp.asc()).all()
    
    chat_history = []
    for msg in messages:
        if msg.role == "user":
            chat_history.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            chat_history.append(AIMessage(content=msg.content))
    
    # Добавляем текущий вопрос
    chat_history.append(HumanMessage(content=current_question))
    
    return chat_history

def check_general_question(question: str) -> bool:
    """Проверяет, является ли вопрос общим"""
    general_patterns = [
        r'что посмотреть', r'что посетить', r'куда сходить', r'куда пойти',
        r'что интересного', r'достопримечательности', r'что можно посмотреть',
        r'чем заняться', r'что делать', r'рекомендации', r'советы',
        r'куда съездить', r'что интересное', r'места для посещения'
    ]
    
    question_lower = question.lower()
    for pattern in general_patterns:
        if re.search(pattern, question_lower):
            return True
    return False

def ask_clarification_questions(question: str, session: ChatSession) -> str:
    """Задает уточняющие вопросы для общих запросов"""
    clarification_context = session.clarification_context or {}
    
    # Если это первый уточняющий вопрос
    if not clarification_context.get('clarification_step'):
        clarification_context = {
            'original_question': question,
            'clarification_step': 1,
            'user_preferences': {}
        }
        session.clarification_context = clarification_context
        
        return ("Чтобы дать вам лучшие рекомендации, расскажите немного о ваших предпочтениях:\n"
                "1. Какой тип мест вас интересует? (музеи, храмы, рестораны, природа)\n"
                "2. Вы путешествуете один, с семьей или друзьями?\n"
                "3. Есть ли у вас особые интересы или ограничения?")

def generate_rag_response(db: Session, user_id: str, question: str, 
                         context_docs: List[Document],
                         conversation_summary: str, user_preferences: Dict) -> str:
    """Генерирует ответ на основе RAG"""
    try:
        if not token_manager.is_token_valid():
            refresh_models()

        # Форматируем контекст из документов
        context_text = ""
        limited_docs = context_docs[:Config.MAX_PLACES_TO_SHOW]
        
        if limited_docs:
            for i, doc in enumerate(limited_docs, 1):
                name = doc.metadata.get('name', 'Неизвестно')
                address = doc.metadata.get('address', 'Адрес не указан')
                description = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                context_text += f"{i}. {name}\n   Адрес: {address}\n   Описание: {description}\n\n"
        else:
            context_text = "К сожалению, в моей локальной базе нет информации по вашему запросу."

        # Создаем системный промпт
        system_prompt = f"""
Ты виртуальный гид по Суздалю. Отвечай на основе предоставленной информации.

Информация из базы:
{context_text}

Отвечай на русском языке. Будь полезным, дружелюбным и информативным гидом.
Если информации недостаточно, предложи поискать в интернете.

Вопрос пользователя: {question}
"""

        # Создаем сообщения
        system_message = SystemMessage(content=system_prompt)
        chat_history = build_gigachat_messages(db, user_id, question)
        all_messages = [system_message] + chat_history
        
        # Вызываем модель
        response = ai_assistant.invoke(all_messages)
        response_text = response.content if hasattr(response, "content") else str(response)
        return response_text

    except Exception as e:
        logger.error(f"Ошибка генерации ответа: {e}")
        return "Извините, произошла ошибка при генерации ответа."

def generate_rag_response_with_offer(db: Session, user_id: str, question: str, 
                                   context_docs: List[Document],
                                   conversation_summary: str, user_preferences: Dict) -> str:
    """Генерирует ответ на основе RAG с явным предложением поиска в интернете"""
    try:
        if not token_manager.is_token_valid():
            refresh_models()

        # Форматируем контекст из документов
        context_text = ""
        limited_docs = context_docs[:Config.MAX_PLACES_TO_SHOW]
        
        if limited_docs:
            for i, doc in enumerate(limited_docs, 1):
                name = doc.metadata.get('name', 'Неизвестно')
                address = doc.metadata.get('address', 'Адрес не указан')
                description = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                context_text += f"{i}. {name}\n   Адрес: {address}\n   Описание: {description}\n\n"
        else:
            context_text = "К сожалению, в моей локальной базе нет информации по вашему запросу."

        # Создаем системный промпт с явным указанием предложить поиск
        system_prompt = f"""
Ты виртуальный гид по Суздалю. Твоя задача:
1. Представить информацию из базы данных
2. В КОНЦЕ ответа обязательно предложить поискать более актуальную информацию в интернете
3. Использовать точную фразу: "Хотите, чтобы я поискал более свежую информацию в интернете?"

Информация из базы:
{context_text}

Отвечай на русском языке. Будь полезным, дружелюбным и информативным гидом.
Сначала представь информацию из базы, затем предложи поиск в интернете.

Вопрос пользователя: {question}
"""

        # Создаем сообщения
        system_message = SystemMessage(content=system_prompt)
        chat_history = build_gigachat_messages(db, user_id, question)
        all_messages = [system_message] + chat_history
        
        # Вызываем модель
        response = ai_assistant.invoke(all_messages)
        response_text = response.content if hasattr(response, "content") else str(response)
        
        # Если модель не добавила предложение о поиске, добавляем его вручную
        if "интернете" not in response_text.lower() and "поискал" not in response_text.lower():
            response_text += "\n\nХотите, чтобы я поискал более свежую информацию в интернете?"
            
        return response_text

    except Exception as e:
        logger.error(f"Ошибка генерации ответа: {e}")
        return "Извините, произошла ошибка. На основе моей базы данных я могу предложить некоторую информацию. Хотите, чтобы я поискал более свежую информацию в интернете?"
def search_web(query: str) -> List[Dict]:
    """Поиск информации в интернете"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=Config.SEARCH_RESULTS))
            return results
    except Exception as e:
        logger.error(f"Ошибка веб-поиска: {e}")
        return []

def generate_web_response(db: Session, user_id: str, question: str, 
                        web_results: List[Dict],
                        conversation_summary: str, user_preferences: Dict) -> str:
    """Генерирует ответ на основе веб-поиска"""
    try:
        if not token_manager.is_token_valid():
            refresh_models()

        # Форматируем результаты поиска
        web_context = ""
        if web_results:
            for i, result in enumerate(web_results[:Config.SEARCH_RESULTS], 1):
                web_context += f"{i}. {result.get('title', 'Без названия')}\n   {result.get('body', 'Описание недоступно')}\n   URL: {result.get('href', 'Ссылка недоступна')}\n\n"
        else:
            web_context = "К сожалению, не удалось найти информацию в интернете."

        # Создаем системный промпт для веб-ответа
        system_prompt = f"""
Ты виртуальный гид по Суздалю. Используй информацию из интернета, чтобы ответить на вопрос пользователя.

Результаты поиска из интернета:
{web_context}

Отвечай на русском языке. Будь полезным, дружелюбным и информативным гидом.
Представь информацию кратко и структурированно.

Вопрос пользователя: {question}
"""

        # Создаем сообщения
        system_message = SystemMessage(content=system_prompt)
        chat_history = build_gigachat_messages(db, user_id, question)
        all_messages = [system_message] + chat_history
        
        # Вызываем модель
        response = ai_assistant.invoke(all_messages)
        response_text = response.content if hasattr(response, "content") else str(response)
        return response_text

    except Exception as e:
        logger.error(f"Ошибка генерации веб-ответа: {e}")
        return "Извините, не удалось найти информацию. Попробуйте задать вопрос по-другому."

def handle_question(db: Session, question: str, user_id: str) -> str:
    if not app_initialized:
        return "Приложение еще не инициализировано. Попробуйте позже."
    
    question = question.strip()
    if not question:
        return "Пожалуйста, задайте ваш вопрос о Суздале."

    # Получаем или создаем сессию
    session = db.query(ChatSession).filter(ChatSession.id == user_id).first()
    if not session:
        session = ChatSession(id=user_id, user_preferences={}, conversation_summary="")
        db.add(session)
        db.commit()

    # Сохраняем вопрос пользователя
    user_message = Message(session_id=user_id, role="user", content=question, timestamp=datetime.utcnow())
    db.add(user_message)

    user_preferences = session.user_preferences if session.user_preferences else {}
    conversation_summary = session.conversation_summary if session.conversation_summary else ""

    # 1. Проверяем, является ли вопрос общим
    is_general_question = check_general_question(question)
    
    # 2. Если вопрос общий, задаем уточняющие вопросы
    if is_general_question:
        clarification_response = ask_clarification_questions(question, session)
        if clarification_response:
            # Сохраняем ответ ассистента с уточняющим вопросом
            assistant_message = Message(session_id=user_id, role="assistant", content=clarification_response, timestamp=datetime.utcnow())
            db.add(assistant_message)
            session.updated_at = datetime.utcnow()
            db.commit()
            return clarification_response

    # 3. Сначала ищем в векторной базе
    context_docs = search_in_vector_store(question)
    
    # 4. Если в локальной базе ничего не найдено, используем фаззи-поиск
    if not context_docs and documents:
        context_docs = fuzzy_retrieval(question, documents)

    # 5. Явно ограничиваем количество документов
    context_docs = context_docs[:Config.MAX_PLACES_TO_SHOW]

    # 6. Если информация найдена в RAG, предлагаем поиск в интернете
    if context_docs:
        # Генерируем ответ с предложением поиска в интернете
        ai_answer = generate_rag_response_with_offer(db, user_id, question, context_docs, conversation_summary, user_preferences)
        
        # Сохраняем ответ ассистента с предложением
        assistant_message = Message(session_id=user_id, role="assistant", content=ai_answer, timestamp=datetime.utcnow())
        db.add(assistant_message)
        
        # Сохраняем контекст для возможного последующего поиска
        session.clarification_context = {
            "original_question": question,
            "found_docs": [doc.metadata for doc in context_docs],
            "needs_web_search": True
        }
        
        session.updated_at = datetime.utcnow()
        db.commit()
        
        return ai_answer
    
    else:
        # 7. Если информации в RAG нет, сразу ищем в интернете
        web_results = search_web(question)
        ai_answer = generate_web_response(db, user_id, question, web_results, conversation_summary, user_preferences)
        
        # Сохраняем ответ ассистента
        assistant_message = Message(session_id=user_id, role="assistant", content=ai_answer, timestamp=datetime.utcnow())
        db.add(assistant_message)
        
        session.updated_at = datetime.utcnow()
        db.commit()
        
        return ai_answer
        
def initialize_app():
    global embedding_model, ai_assistant, documents, vector_store, app_initialized, token_manager
    
    try:
        download_certificate()
        token_manager = GigaChatTokenManager()
        embedding_model, ai_assistant = initialize_models()
        documents = load_data()
        
        if documents:
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            vector_store = FAISS.from_texts(texts, embedding_model, metadatas=metadatas)
            logger.info(f"Приложение успешно инициализировано, загружено {len(documents)} документов")
        else:
            logger.warning("Документы не загружены, RAG будет работать в ограниченном режиме")
        
        app_initialized = True
        
    except Exception as e:
        logger.critical(f"Ошибка инициализации: {e}")
        documents = []
        app_initialized = False
        raise

# Инициализация приложения
initialize_app()

app = FastAPI(title="Суздаль Tourism Assistant")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class Question(BaseModel):
    question: str
    user_id: str = "default"

class FollowUpResponse(BaseModel):
    response: str
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

@app.post("/handle_followup")
async def handle_followup(followup: FollowUpResponse, db: Session = Depends(get_db)):
    """Обрабатывает ответы пользователя на предложения поиска в интернете"""
    if not app_initialized:
        raise HTTPException(status_code=503, detail="Сервис временно недоступен.")
    
    session = db.query(ChatSession).filter(ChatSession.id == followup.user_id).first()
    if not session:
        return {"answer": "Сессия не найдена. Начните новый диалог."}
    
    clarification_context = session.clarification_context or {}
    user_response = followup.response.lower().strip()
    
    # Если пользователь согласился на поиск в интернете
    if clarification_context.get('needs_web_search') and user_response in ['да', 'yes', 'конечно', 'ага', 'пожалуйста', 'ищи']:
        original_question = clarification_context.get('original_question', '')
        web_results = search_web(original_question)
        ai_answer = generate_web_response(db, followup.user_id, original_question, web_results, 
                                        session.conversation_summary or "", 
                                        session.user_preferences or {})
        
        # Сохраняем ответ пользователя
        user_message = Message(session_id=followup.user_id, role="user", content=followup.response, timestamp=datetime.utcnow())
        db.add(user_message)
        
        # Сохраняем ответ ассистента
        assistant_message = Message(session_id=followup.user_id, role="assistant", content=ai_answer, timestamp=datetime.utcnow())
        db.add(assistant_message)
        
        # Очищаем контекст
        session.clarification_context = {}
        session.updated_at = datetime.utcnow()
        db.commit()
        
        return {"answer": ai_answer}
    
    # Если пользователь отказался от поиска
    elif clarification_context.get('needs_web_search') and user_response in ['нет', 'no', 'не надо', 'не нужно']:
        # Сохраняем ответ пользователя
        user_message = Message(session_id=followup.user_id, role="user", content=followup.response, timestamp=datetime.utcnow())
        db.add(user_message)
        
        # Отвечаем, что поиск не будет выполнен
        assistant_message = Message(session_id=followup.user_id, role="assistant", 
                                  content="Хорошо, если у вас появятся другие вопросы о Суздале, буду рад помочь!", 
                                  timestamp=datetime.utcnow())
        db.add(assistant_message)
        
        # Очищаем контекст
        session.clarification_context = {}
        session.updated_at = datetime.utcnow()
        db.commit()
        
        return {"answer": "Хорошо, если у вас появятся другие вопросы о Суздале, буду рад помочь!"}
    
    else:
        return {"answer": "Не понял ваш ответ. Пожалуйста, ответьте 'да' или 'нет'."}

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
