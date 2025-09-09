import os
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import JSON
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
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String(10), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    message_metadata = Column(JSON, nullable=True)
    session = relationship("ChatSession", back_populates="messages")

# Индексы
Index('ix_messages_session_id', Message.session_id)
Index('ix_messages_timestamp', Message.timestamp)
Index('ix_messages_session_role', Message.session_id, Message.role)

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
    RETRIEVER_K = 5
    CERT_PATH = os.getenv("CERT_PATH", "./cert.pem")
    CERT_URL = os.getenv("CERT_URL")
    GIGACHAT_AUTH = os.getenv("GIGACHAT_AUTH")
    CSV_DATA_URL = os.getenv(
        "CSV_DATA_URL",
        "https://raw.githubusercontent.com/vuyq/SuzdalAI/main/suzdal_full_guide_refine/attractions.csv"
    )
    TOKEN_EXPIRY_MINUTES = 25  # Токен живет ~30 минут, обновляем заранее

# Глобальные переменные для моделей
embedding_model = None
ai_assistant = None
documents = []
vector_store = None
document_retriever = None
app_initialized = False
token_manager = None

# Менеджер для управления токенами GigaChat
class GigaChatTokenManager:
    def __init__(self):
        self.access_token = None
        self.token_expires = None
        self.lock = False
        
    def get_valid_token(self) -> str:
        """Получает валидный токен, обновляя при необходимости"""
        if self.access_token and self.token_expires and datetime.now() < self.token_expires:
            return self.access_token
        
        # Защита от параллельных запросов
        if self.lock:
            # Ждем, пока другой поток обновит токен
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
        """Проверяет, валиден ли текущий токен"""
        return bool(self.access_token and self.token_expires and datetime.now() < self.token_expires)

# Загрузка сертификата
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

# Получение токена GigaChat
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

# Инициализация моделей с обновляемым токеном
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

# Обновление моделей с новым токеном
def refresh_models():
    """Обновляет модели с новым токеном"""
    global embedding_model, ai_assistant
    try:
        access_token = token_manager.get_valid_token()
        
        # Обновляем embedding модель
        if embedding_model:
            embedding_model.access_token = access_token
        
        # Обновляем AI ассистента
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

# Загрузка CSV данных
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

# Веб-поиск через DuckDuckGo с кэшированием
@lru_cache(maxsize=100)
def perform_web_search(query: str) -> str:
    try:
        if DDGS is None:
            return "Веб-поиск недоступен (модуль ddgs не установлен)"
            
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(f"{query} Суздаль", max_results=Config.SEARCH_RESULTS, timelimit='y'):
                results.append(f"• {r['title']}\n  {r['href']}\n  {r['body'][:200]}...")
        return "\n\n".join(results) if results else "Не найдено результатов"
    except Exception as e:
        logger.error(f"Ошибка веб-поиска: {e}")
        return "Ошибка при выполнении поиска"

# Поиск в векторной базе
def search_in_vector_store(query: str, k: int = None) -> List[Document]:
    """Поиск в векторной базе с использованием FAISS"""
    if not vector_store or not document_retriever:
        return []
    
    try:
        k = k or Config.RETRIEVER_K
        results = vector_store.similarity_search(query, k=k)
        return results
    except Exception as e:
        logger.error(f"Ошибка поиска в векторной базе: {e}")
        return []

# Fuzzy search по RAG (резервный метод)
def fuzzy_retrieval(question: str, docs: List[Document], limit: int = 5) -> List[Document]:
    if not docs:
        return []
    
    # Используем только названия для fuzzy search для производительности
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

# Prompt для AI
TOURISM_PROMPT_TEMPLATE = """
Ты виртуальный гид по Суздалю. Отвечай на вопросы информативно и дружелюбно, учитывая контекст предыдущего диалога.

[История диалога]:
{dialog_context}

[Данные из базы о достопримечательностях]:
{context}

[Веб-результаты]:
{web_search}

[Текущий вопрос]:
{question}

Учти всю историю диалога выше. Отвечай подробно и полезно, предлагая конкретные рекомендации. Если в истории уже обсуждались какие-то места, упомяни это в ответе.
"""
tourism_prompt = PromptTemplate.from_template(TOURISM_PROMPT_TEMPLATE)

def generate_ai_response(question: str, context_docs: List[Document], web_results: str, dialog_context: str) -> str:
    try:
        # Проверяем и обновляем токен при необходимости
        if not token_manager.is_token_valid():
            refresh_models()
        
        prompt_input = {
            "question": question,
            "context": "\n\n".join(d.page_content for d in context_docs) if context_docs else "Нет данных в базе",
            "web_search": web_results,
            "dialog_context": dialog_context
        }
        response = ai_assistant.invoke(tourism_prompt.format(**prompt_input))
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        logger.error(f"Ошибка генерации ответа: {e}")
        # Пытаемся обновить токен и повторить
        try:
            refresh_models()
            response = ai_assistant.invoke(tourism_prompt.format(**prompt_input))
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as retry_error:
            logger.error(f"Повторная ошибка генерации ответа: {retry_error}")
            return "Извините, произошла ошибка при генерации ответа. Попробуйте позже."

# Работа с базой
def update_dialog_context(db: Session, user_id: str, role: str, message: str):
    try:
        session = db.query(ChatSession).filter(ChatSession.id == user_id).first()
        if not session:
            session = ChatSession(id=user_id)
            db.add(session)
            # Не коммитим здесь, коммитим в конце
        db_message = Message(session_id=user_id, role=role, content=message, timestamp=datetime.utcnow())
        db.add(db_message)
        db.commit()  # Один коммит для всего
    except Exception as e:
        logger.error(f"Ошибка обновления контекста диалога: {e}")
        db.rollback()
        raise

def set_last_question(db: Session, user_id: str, question: str):
    try:
        session = db.query(ChatSession).filter(ChatSession.id == user_id).first()
        if session:
            session.last_question = question
            session.updated_at = datetime.utcnow()
            db.commit()
    except Exception as e:
        logger.error(f"Ошибка установки последнего вопроса: {e}")
        db.rollback()

def get_last_question(db: Session, user_id: str) -> Optional[str]:
    try:
        session = db.query(ChatSession).filter(ChatSession.id == user_id).first()
        return session.last_question if session else None
    except Exception as e:
        logger.error(f"Ошибка получения последнего вопроса: {e}")
        return None

def get_dialog_context(db: Session, user_id: str, max_messages: int = 20) -> str:
    try:
        messages = db.query(Message).filter(Message.session_id == user_id)\
                     .order_by(Message.timestamp.asc()).limit(max_messages).all()
        
        dialog_lines = []
        for msg in messages:
            role = "Пользователь" if msg.role == "user" else "Ассистент"
            dialog_lines.append(f"{role}: {msg.content}")
        
        return "\n".join(dialog_lines)
    except Exception as e:
        logger.error(f"Ошибка получения контекста диалога: {e}")
        return ""

def clear_chat_history(db: Session, user_id: str):
    """Очищает историю сообщений для пользователя"""
    try:
        db.query(Message).filter(Message.session_id == user_id).delete()
        session = db.query(ChatSession).filter(ChatSession.id == user_id).first()
        if session:
            session.last_question = None
            session.updated_at = datetime.utcnow()
        db.commit()
        logger.info(f"История чата очищена для пользователя {user_id}")
    except Exception as e:
        logger.error(f"Ошибка очистки истории: {e}")
        db.rollback()
        raise

# Проверка на непонятное сообщение
def is_message_unclear(message: str) -> bool:
    msg = message.strip()
    if len(msg) < 3:
        return True
    if all(c in ".,!? " for c in msg):
        return True
    return False

# Проверка на уточнения
def needs_clarification(question: str) -> Tuple[bool, str]:
    q = question.lower()
    if "рестора" in q or "поесть" in q or "кухн" in q:
        return True, (
            "Я вижу, что Вы ищите ресторан или кофе. Можете уточнить:\n"
            "- Какой тип кухни предпочитаете (русская, японская, китайская, европейская, любая)?\n"
            "- Важно ли расположение (центр, окраина, рядом с достопримечательностями)?\n"
            "- Нужен ли бюджетный вариант или премиум?"
        )
    if any(word in q for word in ["достопримечательности", "музеи", "куда сходить", "что посетить"]):
    if "музе" in q:  # Отлавливаем "музей", "музеи", "музеев" и т.д.
        return True, (
            "В Суздале много интересных музеев. Какой вас интересует больше?\n\n"
            "- Музей деревянного зодчества\n"
            "- Спасо-Евфимиев монастырь (музейный комплекс)\n"
            "- Кремль с его экспозициями\n"
            "- Музей восковых фигур\n"
            "- Или что-то другое?"
        )
    else:
        return True, (
            "Вы ищете достопримечательности. Хотите больше про:\n"
            "- Исторические объекты\n"
            "- Музеи\n"
            "- Храмы и монастыри\n"
            "- Природные места"
        )
    return False, ""

def format_context_docs(docs: List[Document]) -> str:
    if not docs:
        return "В базе данных ничего не найдено."
    lines = []
    for doc in docs:
        meta = doc.metadata
        entry = []
        if "name" in meta:
            entry.append(f"🏷 {meta['name']}")
        if "type" in meta:
            entry.append(f"{meta['type']}")
        if "address" in meta:
            entry.append(f"📍 Адрес: {meta['address']}")
        if "hours" in meta:
            entry.append(f"🕒 Время работы: {meta['hours']}")
        if "price" in meta:
            entry.append(f"💰 Цена: {meta['price']}")
        if "description" in meta:
            entry.append(f"ℹ {meta['description']}")
        lines.append("\n".join(entry))
    return "\n\n".join(lines)

# Основная логика
def handle_question(db: Session, question: str, user_id: str) -> str:
    if not app_initialized:
        return "Приложение еще не инициализировано. Попробуйте позже."
    
    question = question.strip()
    if not question:
        return "Пожалуйста, задайте Ваш вопрос. Я всегда готов помочь!"

    if is_message_unclear(question):
        response = "Не очень понял Ваше сообщение, пожалуйста, напишите ещё раз."
        update_dialog_context(db, user_id, "assistant", response)
        return response

    update_dialog_context(db, user_id, "user", question)
    dialog_context = get_dialog_context(db, user_id)

    if question.lower() in ["да", "расскажи", "расскажи подробнее", "подробнее"]:
        context_docs = search_in_vector_store(last_q)
        if context_docs:
            ai_answer = generate_ai_response(last_q, context_docs, "", get_dialog_context(db, user_id))
            response = f"🤖 Расширенный рассказ:\n\n{ai_answer}"
        else:
            response = "К сожалению, не могу найти информацию для подробного рассказа."
    
        update_dialog_context(db, user_id, "assistant", response)
        return response

    if question.lower() in ["да", "ищи", "да, ищи", "в интернете", "давай интернет"]:
        last_q = get_last_question(db, user_id)
        if last_q:
            web_results = perform_web_search(last_q)
            response = f"🌐 Вот что удалось найти в интернете по Вашему запросу! «{last_q}»:\n\n{web_results}"
            update_dialog_context(db, user_id, "assistant", response)
            return response
        else:
            response = "Простите, не могу найти предыдущий запрос для поиска в интернете."
            update_dialog_context(db, user_id, "assistant", response)
            return response

    # Уточнения
    needs_clarify, clarification_text = needs_clarification(question)
    if needs_clarify:
        update_dialog_context(db, user_id, "assistant", clarification_text)
        return clarification_text

    # Поиск в векторной базе (основной метод)
    context_docs = search_in_vector_store(question)
    
    # Если в векторной базе ничего не найдено, используем fuzzy search как запасной вариант
    if not context_docs and documents:
        context_docs = fuzzy_retrieval(question, documents, limit=Config.RETRIEVER_K)

    if context_docs:
        formatted_context = format_context_docs(context_docs)
        set_last_question(db, user_id, question)
        response = (
            f"Вот, что я могу вам предложить:\n\n{formatted_context}\n\n"
            "Хотите, чтобы я сделал расширенный рассказ об этих местах на основе этой информации? "
            "Просто напишите 'да' или 'расскажи подробнее'."
        )
    else:
        web_results = perform_web_search(question)
        response = f"🌐 В базе ничего не найдено. Вот что мне удалось найти в интернете:\n\n{web_results}"

# Сохраняем context_docs в сессии для возможного последующего использования
# (это потребует изменения модели базы данных или временного хранилища)
    

# Инициализация
def initialize_app():
    global embedding_model, ai_assistant, documents, vector_store, document_retriever, app_initialized, token_manager
    
    try:
        download_certificate()
        
        # Инициализируем менеджер токенов
        token_manager = GigaChatTokenManager()
        
        # Инициализируем модели
        embedding_model, ai_assistant = initialize_models()
        
        # Загружаем документы
        documents = load_data()
        
        if documents:
            # Создаем векторное хранилище
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

# Инициализируем приложение
initialize_app()

# FastAPI
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

@app.get("/history/{user_id}")
async def get_history(user_id: str, db: Session = Depends(get_db)):
    """Получить историю сообщений пользователя"""
    try:
        messages = db.query(Message).filter(Message.session_id == user_id)\
                     .order_by(Message.timestamp.asc()).all()
        
        history = []
        for msg in messages:
            history.append({
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            })
        
        return {"user_id": user_id, "history": history}
    except Exception as e:
        logger.error(f"Ошибка получения истории: {e}")
        raise HTTPException(status_code=500, detail="Ошибка получения истории")

@app.post("/clear_history/{user_id}")
async def clear_history(user_id: str, db: Session = Depends(get_db)):
    """Очищает историю сообщений для указанного пользователя"""
    try:
        clear_chat_history(db, user_id)
        return {"status": "history cleared", "user_id": user_id}
    except Exception as e:
        logger.error(f"Ошибка очистки истории: {e}")
        raise HTTPException(status_code=500, detail="Ошибка очистки истории")

@app.post("/reinitialize")
async def reinitialize():
    """Переинициализирует приложение (для административных целей)"""
    try:
        initialize_app()
        return {"status": "reinitialized", "success": app_initialized}
    except Exception as e:
        logger.error(f"Ошибка переинициализации: {e}")
        raise HTTPException(status_code=500, detail="Ошибка переинициализации")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
