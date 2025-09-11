import os
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, ForeignKey, JSON, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy import inspect
from datetime import datetime, timedelta
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_gigachat import GigaChat, GigaChatEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
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
import traceback

# Логирование
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
            logger.warning(f"Ошибка загрузки сертификата: {e}")
            # Создаем пустой файл для продолжения работы
            Path(Config.CERT_PATH).touch()
            logger.info("Создан пустой файл сертификата для продолжения работы")

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
    
    verify_ssl = Path(Config.CERT_PATH).exists() and os.path.getsize(Config.CERT_PATH) > 0
    
    response = requests.post(
        url, headers=headers, data=payload,
        verify=verify_ssl, timeout=Config.REQUEST_TIMEOUT
    )
    response.raise_for_status()
    return response.json().get("access_token")

def initialize_models() -> Tuple[GigaChatEmbeddings, GigaChat]:
    try:
        access_token = token_manager.get_valid_token()
        
        verify_ssl = Path(Config.CERT_PATH).exists() and os.path.getsize(Config.CERT_PATH) > 0
        ca_bundle = Config.CERT_PATH if verify_ssl else None
        
        embedding_model = GigaChatEmbeddings(
            access_token=access_token, 
            model="Embeddings", 
            scope="GIGACHAT_API_PERS",
            verify_ssl_certs=verify_ssl,
            ca_bundle_file=ca_bundle,
            timeout=Config.REQUEST_TIMEOUT
        )
        
        ai_assistant = GigaChat(
            access_token=access_token, 
            model="GigaChat", 
            temperature=0.2,
            verify_ssl_certs=verify_ssl,
            ca_bundle_file=ca_bundle,
            timeout=Config.REQUEST_TIMEOUT, 
            verbose=False
        )
        
        return embedding_model, ai_assistant
        
    except Exception as e:
        logger.error(f"Ошибка инициализации моделей: {e}")
        raise

def refresh_models():
    global embedding_model, ai_assistant
    try:
        access_token = token_manager.get_valid_token()
        
        verify_ssl = Path(Config.CERT_PATH).exists() and os.path.getsize(Config.CERT_PATH) > 0
        ca_bundle = Config.CERT_PATH if verify_ssl else None
        
        if embedding_model:
            embedding_model.access_token = access_token
        
        if ai_assistant:
            ai_assistant.access_token = access_token
        else:
            ai_assistant = GigaChat(
                access_token=access_token, 
                model="GigaChat", 
                temperature=0.2,
                verify_ssl_certs=verify_ssl,
                ca_bundle_file=ca_bundle,
                timeout=Config.REQUEST_TIMEOUT, 
                verbose=False
            )
            
        logger.info("Модели успешно обновлены с новым токеном")
    except Exception as e:
        logger.error(f"Ошибка обновления моделей: {e}")
        # Не падаем, продолжаем работу в degraded mode

def load_data() -> List[Document]:
    try:
        try:
            df = pd.read_csv(Config.CSV_DATA_URL, sep=';', on_bad_lines='skip')
        except Exception:
            try:
                df = pd.read_csv(Config.CSV_DATA_URL, on_bad_lines='skip')
            except Exception as e:
                logger.error(f"Ошибка загрузки CSV: {e}")
                # Пробуем создать минимальный набор данных
                return create_fallback_data()

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
            
            if not metadata.get('name'):
                continue
                
            doc = Document(
                page_content="\n".join(content) if content else metadata.get('description', ''),
                metadata=metadata
            )
            documents.append(doc)
            
        logger.info(f"Загружено {len(documents)} документов из CSV")
        return documents
        
    except Exception as e:
        logger.error(f"Критическая ошибка загрузки данных: {e}")
        return create_fallback_data()

def create_fallback_data() -> List[Document]:
    """Создает fallback данные о Суздале"""
    fallback_data = [
        {
            'name': 'Суздальский кремль',
            'type': 'достопримечательность',
            'address': 'ул. Кремлевская, д. 1',
            'description': 'Исторический центр Суздаля, древнейшее сооружение города'
        },
        {
            'name': 'Ресторан Русская изба',
            'type': 'ресторан',
            'address': 'ул. Ленина, д. 15',
            'description': 'Традиционная русская кухня в аутентичной обстановке'
        },
        {
            'name': 'Гостиница Горячие ключи',
            'type': 'отель',
            'address': 'ул. Коровники, д. 45',
            'description': 'Комфортабельный отель с бассейном и спа'
        },
        {
            'name': 'Кафе Улей',
            'type': 'кафе',
            'address': 'ул. Васильевская, д. 27',
            'description': 'Уютное кафе с домашней кухней и выпечкой'
        },
        {
            'name': 'Трапезная палата',
            'type': 'ресторан',
            'address': 'ул. Кремлевская, д. 10',
            'description': 'Ресторан в историческом здании с русской кухней'
        }
    ]
    
    documents = []
    for data in fallback_data:
        doc = Document(
            page_content=data['description'],
            metadata=data
        )
        documents.append(doc)
    
    logger.info(f"Создано {len(documents)} fallback документов")
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
    if not vector_store:
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

TOURISM_PROMPT_TEMPLATE = """
Ты виртуальный гид по Суздалю. Отвечай на русском языке кратко и информативно.

Информация о достопримечательностях:
{context}

Вопрос пользователя: {question}

Ответь максимально полезно и точно на основе предоставленной информации.
Если информации недостаточно, вежливо сообщи об этом.
"""
tourism_prompt = PromptTemplate.from_template(TOURISM_PROMPT_TEMPLATE)

def generate_ai_response(db: Session, user_id: str, question: str, 
                         context_docs: List[Document], web_results: str,
                         conversation_summary: str, user_preferences: Dict) -> str:
    try:
        # Проверяем доступность GigaChat
        if not ai_assistant or not token_manager or not token_manager.is_token_valid():
            refresh_models()
            if not ai_assistant:
                return generate_fallback_response(question, context_docs)

        # Формируем контекст из найденных документов
        context_text = ""
        if context_docs:
            for i, doc in enumerate(context_docs[:3], 1):
                context_text += f"\n{i}. {doc.metadata.get('name', 'Объект')}: "
                if doc.metadata.get('description'):
                    context_text += f"{doc.metadata['description'][:200]}... "
                if doc.metadata.get('address'):
                    context_text += f"Адрес: {doc.metadata['address']} "
                if doc.metadata.get('hours'):
                    context_text += f"Часы: {doc.metadata['hours']}"
        else:
            context_text = "Информация не найдена в базе данных"

        # Формируем системный промпт
        system_prompt = tourism_prompt.format(
            question=question,
            context=context_text or "Нет данных в базе",
        )

        # Исправленный вызов модели - используем правильный формат
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ]

        # Правильный вызов модели
        response = ai_assistant.invoke(messages)
        
        response_text = response.content if hasattr(response, "content") else str(response)
        return response_text

    except Exception as e:
        logger.error(f"Ошибка генерации ответа: {e}")
        logger.error(traceback.format_exc())
        return generate_fallback_response(question, context_docs)

def generate_fallback_response(question: str, context_docs: List[Document]) -> str:
    """Fallback ответ когда GigaChat недоступен"""
    question_lower = question.lower()
    
    # Ответы про еду
    if any(keyword in question_lower for keyword in ['где поесть', 'еда', 'ресторан', 'кафе', 'столовая', 'питание']):
        restaurants = [doc for doc in context_docs if doc.metadata.get('type', '').lower() in 
                      ['ресторан', 'кафе', 'столовая', 'еда', 'питание', 'food']]
        
        if restaurants:
            response = "🍽️ **Где поесть в Суздале:**\n\n"
            for i, rest in enumerate(restaurants[:5], 1):
                response += f"**{i}. {rest.metadata.get('name', 'Заведение')}**\n"
                if 'address' in rest.metadata:
                    response += f"   📍 Адрес: {rest.metadata['address']}\n"
                if 'hours' in rest.metadata:
                    response += f"   🕒 Часы работы: {rest.metadata['hours']}\n"
                if 'description' in rest.metadata:
                    response += f"   📝 {rest.metadata['description'][:100]}...\n"
                response += "\n"
            return response + "\nРекомендую уточнить актуальный режим работы у администрации заведений."
        else:
            return "🍽️ В Суздале есть множество кафе и ресторанов с традиционной русской кухней. Популярные места:\n\n• **Рестораны в центре города** - предлагают блюда русской кухни\n• **Кафе на улице Ленина** - уютные места с домашней атмосферой\n• **Трапезные при монастырях** - аутентичная атмосфера\n\nРекомендую прогуляться по центру города - там вы найдете множество вариантов!"

    # Ответы про достопримечательности
    elif any(keyword in question_lower for keyword in ['достопримечательность', 'что посмотреть', 'куда сходить', 'музей', 'кремль']):
        attractions = [doc for doc in context_docs if doc.metadata.get('type', '').lower() in 
                      ['достопримечательность', 'музей', 'памятник', 'attraction']]
        
        if attractions:
            response = "🏛️ **Достопримечательности Суздаля:**\n\n"
            for i, attr in enumerate(attractions[:5], 1):
                response += f"**{i}. {attr.metadata.get('name', 'Достопримечательность')}**\n"
                if 'description' in attr.metadata:
                    response += f"   📝 {attr.metadata['description'][:100]}...\n"
                if 'address' in attr.metadata:
                    response += f"   📍 Адрес: {attr.metadata['address']}\n"
                response += "\n"
            return response
        else:
            return "🏛️ Суздаль богат достопримечательностями! Обязательно посетите:\n\n• **Суздальский кремль** - исторический центр города\n• **Музей деревянного зодчества** - уникальные памятники архитектуры\n• **Покровский монастырь** - древняя обитель с богатой историей\n• **Торговые ряды** - архитектурный памятник XIX века\n• **Многочисленные церкви и храмы** - более 30 культовых сооружений"

    # Ответы про жилье
    elif any(keyword in question_lower for keyword in ['отель', 'гостиница', 'жилье', 'где остановиться', 'ночлег']):
        hotels = [doc for doc in context_docs if doc.metadata.get('type', '').lower() in 
                 ['отель', 'гостиница', 'hotel', 'гостевой дом']]
        
        if hotels:
            response = "🏨 **Где остановиться в Суздале:**\n\n"
            for i, hotel in enumerate(hotels[:5], 1):
                response += f"**{i}. {hotel.metadata.get('name', 'Отель')}**\n"
                if 'address' in hotel.metadata:
                    response += f"   📍 Адрес: {hotel.metadata['address']}\n"
                if 'description' in hotel.metadata:
                    response += f"   📝 {hotel.metadata['description'][:100]}...\n"
                response += "\n"
            return response
        else:
            return "🏨 В Суздале есть различные варианты размещения:\n\n• **Гостиницы в центре города** - удобное расположение\n• **Гостевые дома** - уютная атмосфера\n• **Загородные отели** - тишина и природа\n• **Гостиницы при монастырях** - уникальный опыт\n\nРекомендую бронировать заранее, особенно в туристический сезон."

    # Общий ответ
    return "Привет! Я виртуальный гид по Суздалю. 🏛️\n\nЧем могу помочь?\n• Подсказать где поесть 🍽️\n• Посоветовать достопримечательности 🏛️\n• Помочь с выбором жилья 🏨\n• Рассказать об истории города 📖\n\nЗадайте ваш вопрос, и я постараюсь помочь!"

def handle_question(db: Session, question: str, user_id: str) -> str:
    if not app_initialized:
        return "Приложение еще не инициализировано. Попробуйте позже."
    
    question = question.strip()
    if not question:
        return "Пожалуйста, задайте ваш вопрос о Суздале."

    # Получаем или создаем сессию
    session = db.query(ChatSession).filter(ChatSession.id == user_id).first()
    if not session:
        session = ChatSession(id=user_id)
        db.add(session)
        db.commit()

    # Сохраняем вопрос пользователя
    db.add(Message(session_id=user_id, role="user", content=question, timestamp=datetime.utcnow()))
    db.commit()

    user_preferences = session.user_preferences if session and session.user_preferences else {}
    conversation_summary = session.conversation_summary if session else ""

    # Ищем релевантную информацию
    context_docs = []
    if vector_store:
        context_docs = search_in_vector_store(question)
    if not context_docs and documents:
        context_docs = fuzzy_retrieval(question, documents, limit=Config.RETRIEVER_K)

    # Генерируем ответ
    ai_answer = generate_ai_response(db, user_id, question, context_docs, "", conversation_summary, user_preferences)

    # Сохраняем ответ ассистента
    db.add(Message(session_id=user_id, role="assistant", content=ai_answer, timestamp=datetime.utcnow()))
    db.commit()

    return ai_answer

def safe_init_db():
    """Безопасная инициализация базы данных без удаления таблиц"""
    try:
        inspector = inspect(engine)
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
        
        # Безопасно добавляем недостающие колонки
        try:
            if 'chat_sessions' in existing_tables:
                chat_columns = [col['name'] for col in inspector.get_columns('chat_sessions')]
                
                if 'user_preferences' not in chat_columns:
                    with engine.connect() as conn:
                        conn.execute(text("ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS user_preferences JSONB"))
                        conn.commit()
                        
                if 'conversation_summary' not in chat_columns:
                    with engine.connect() as conn:
                        conn.execute(text("ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS conversation_summary TEXT"))
                        conn.commit()
                        
                if 'clarification_context' not in chat_columns:
                    with engine.connect() as conn:
                        conn.execute(text("ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS clarification_context JSONB"))
                        conn.commit()
            
            if 'messages' in existing_tables:
                messages_columns = [col['name'] for col in inspector.get_columns('messages')]
                if 'embeddings' not in messages_columns:
                    with engine.connect() as conn:
                        conn.execute(text("ALTER TABLE messages ADD COLUMN IF NOT EXISTS embeddings JSONB"))
                        conn.commit()
                        
        except Exception as e:
            logger.warning(f"Ошибка при добавлении колонок: {e}")
            
    except Exception as e:
        logger.error(f"Ошибка инициализации базы: {e}")

# Инициализация базы данных
safe_init_db()

def initialize_app():
    global embedding_model, ai_assistant, documents, vector_store, document_retriever, app_initialized, token_manager
    
    try:
        # Загружаем сертификат
        download_certificate()
        
        # Инициализируем токен менеджер
        token_manager = GigaChatTokenManager()
        
        # Пытаемся инициализировать модели
        try:
            embedding_model, ai_assistant = initialize_models()
            logger.info("Модели GigaChat успешно инициализированы")
        except Exception as model_error:
            logger.error(f"Ошибка инициализации моделей GigaChat: {model_error}")
            ai_assistant = None
            embedding_model = None
        
        # Загружаем данные
        documents = load_data()
        
        # Создаем векторное хранилище если есть модели и данные
        if documents and embedding_model:
            try:
                vector_store = FAISS.from_documents(documents, embedding_model)
                document_retriever = vector_store.as_retriever(search_kwargs={"k": Config.RETRIEVER_K})
                logger.info(f"Векторное хранилище создано, {len(documents)} документов")
            except Exception as vector_error:
                logger.error(f"Ошибка создания векторного хранилища: {vector_error}")
                vector_store = None
                document_retriever = None
        else:
            logger.warning("Векторное хранилище не создано (нет данных или моделей)")
        
        app_initialized = True
        logger.info("Приложение инициализировано")
        
    except Exception as e:
        logger.critical(f"Критическая ошибка инициализации: {e}")
        logger.error(traceback.format_exc())
        documents = create_fallback_data()
        app_initialized = True  # Все равно запускаем в fallback режиме

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
        logger.error(traceback.format_exc())
        return {"answer": "Извините, произошла ошибка при обработке вашего запроса. Попробуйте еще раз."}

@app.get("/health")
async def health_check():
    status = "healthy" if app_initialized else "degraded"
    return {
        "status": status, 
        "initialized": app_initialized,
        "timestamp": datetime.utcnow(),
        "documents_loaded": len(documents),
        "gigachat_available": ai_assistant is not None,
        "vector_store_available": vector_store is not None
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
