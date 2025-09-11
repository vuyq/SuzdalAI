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
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
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
    MAX_RETRIES = 2
    REQUEST_TIMEOUT = 10
    SEARCH_RESULTS = 3
    RETRIEVER_K = 5
    CSV_DATA_URL = os.getenv(
        "CSV_DATA_URL",
        "https://raw.githubusercontent.com/vuyq/SuzdalAI/main/suzdal_full_guide_refine/attractions.csv"
    )
    MEMORY_CONTEXT_SIZE = 10
    PREFERENCES_UPDATE_INTERVAL = 5
    SEMANTIC_SEARCH_K = 3

# Глобальные переменные
documents = []
vector_store = None
app_initialized = False

def load_data() -> List[Document]:
    try:
        try:
            df = pd.read_csv(Config.CSV_DATA_URL, sep=';', on_bad_lines='skip')
        except Exception:
            try:
                df = pd.read_csv(Config.CSV_DATA_URL, on_bad_lines='skip')
            except Exception as e:
                logger.error(f"Ошибка загрузки CSV: {e}")
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
            'description': 'Исторический центр Суздаля, древнейшее сооружение города с музеями и экспозициями'
        },
        {
            'name': 'Ресторан Русская изба',
            'type': 'ресторан',
            'address': 'ул. Ленина, д. 15',
            'description': 'Традиционная русская кухня в аутентичной обстановке, щи, пироги, медовуха'
        },
        {
            'name': 'Гостиница Горячие ключи',
            'type': 'отель',
            'address': 'ул. Коровники, д. 45',
            'description': 'Комфортабельный отель с бассейном, спа и рестораном'
        },
        {
            'name': 'Кафе Улей',
            'type': 'кафе',
            'address': 'ул. Васильевская, д. 27',
            'description': 'Уютное кафе с домашней кухней, свежей выпечкой и кофе'
        },
        {
            'name': 'Трапезная палата',
            'type': 'ресторан',
            'address': 'ул. Кремлевская, д. 10',
            'description': 'Ресторан в историческом здании с блюдами русской кухни и европейским меню'
        },
        {
            'name': 'Музей деревянного зодчества',
            'type': 'музей',
            'address': 'ул. Пушкарская, д. 27Б',
            'description': 'Под открытым небом представлены уникальные памятники деревянной архитектуры'
        },
        {
            'name': 'Покровский монастырь',
            'type': 'монастырь',
            'address': 'ул. Покровская, д. 76',
            'description': 'Действующий женский монастырь XIV века с богатой историей'
        },
        {
            'name': 'Спасо-Евфимиев монастырь',
            'type': 'монастырь',
            'address': 'ул. Ленина, д. 135',
            'description': 'Мужской монастырь-крепость с музеями и колокольными звонами'
        },
        {
            'name': 'Торговые ряды',
            'type': 'достопримечательность',
            'address': 'ул. Ленина, д. 63А',
            'description': 'Архитектурный комплекс XIX века с сувенирными лавками и кафе'
        },
        {
            'name': 'Ризоположенский монастырь',
            'type': 'монастырь',
            'address': 'ул. Ленина, д. 79',
            'description': 'Один из древнейших монастырей России с Преподобенской колокольней'
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

def search_in_documents(query: str, docs: List[Document], k: int = 5) -> List[Document]:
    """Простой поиск по документам на основе ключевых слов"""
    if not docs:
        return []
    
    query_lower = query.lower()
    results = []
    
    for doc in docs:
        score = 0
        doc_content = f"{doc.metadata.get('name', '')} {doc.metadata.get('type', '')} {doc.metadata.get('description', '')} {doc.page_content}".lower()
        
        # Проверяем совпадение ключевых слов
        keywords = query_lower.split()
        for keyword in keywords:
            if keyword in doc_content:
                score += 1
        
        # Особые случаи для популярных запросов
        if 'есть' in query_lower or 'питание' in query_lower or 'ресторан' in query_lower or 'кафе' in query_lower:
            if any(food_type in doc.metadata.get('type', '').lower() for food_type in ['ресторан', 'кафе', 'столовая', 'еда']):
                score += 3
        
        if score > 0:
            results.append((doc, score))
    
    # Сортируем по релевантности
    results.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in results[:k]]

def generate_ai_response(question: str, context_docs: List[Document]) -> str:
    """Генерация ответа на основе контекста без использования GigaChat"""
    try:
        question_lower = question.lower()
        
        # Формируем контекст из найденных документов
        context_text = ""
        if context_docs:
            for i, doc in enumerate(context_docs[:3], 1):
                context_text += f"\n{i}. {doc.metadata.get('name', 'Объект')}: "
                if doc.metadata.get('description'):
                    context_text += f"{doc.metadata['description']} "
                if doc.metadata.get('address'):
                    context_text += f"Адрес: {doc.metadata['address']} "
                if doc.metadata.get('hours'):
                    context_text += f"Часы: {doc.metadata['hours']}"
        
        # Специализированные ответы для популярных запросов
        if any(keyword in question_lower for keyword in ['где поесть', 'еда', 'ресторан', 'кафе', 'столовая', 'питание']):
            restaurants = [doc for doc in context_docs if any(food_type in doc.metadata.get('type', '').lower() 
                          for food_type in ['ресторан', 'кафе', 'столовая', 'еда'])]
            
            if restaurants:
                response = "🍽️ **Где поесть в Суздале:**\n\n"
                for i, rest in enumerate(restaurants[:5], 1):
                    response += f"**{i}. {rest.metadata.get('name', 'Заведение')}**\n"
                    if 'address' in rest.metadata:
                        response += f"   📍 Адрес: {rest.metadata['address']}\n"
                    if 'description' in rest.metadata:
                        response += f"   📝 {rest.metadata['description']}\n"
                    response += "\n"
                response += "\n💡 **Совет:** Большинство заведений сосредоточено в центре города вдоль улицы Ленина и вокруг Кремля. Рекомендую уточнять актуальный режим работы."
                return response
            else:
                return "🍽️ В Суздале множество мест где можно вкусно поесть:\n\n• **Рестораны русской кухни** - предлагают традиционные блюда: щи, пироги, блины, медовуху\n• **Кафе в центре города** - уютные места с домашней атмосферой\n• **Трапезные при монастырях** - аутентичная атмосфера и постная кухня\n• **Гостиничные рестораны** - обычно работают до позднего вечера\n\n📍 **Район поиска:** Центр города, улица Ленина, Торговые ряды"

        elif any(keyword in question_lower for keyword in ['достопримечательность', 'что посмотреть', 'куда сходить', 'музей', 'кремль']):
            attractions = [doc for doc in context_docs if any(attr_type in doc.metadata.get('type', '').lower() 
                            for attr_type in ['достопримечательность', 'музей', 'монастырь', 'кремль'])]
            
            if attractions:
                response = "🏛️ **Достопримечательности Суздаля:**\n\n"
                for i, attr in enumerate(attractions[:5], 1):
                    response += f"**{i}. {attr.metadata.get('name', 'Достопримечательность')}**\n"
                    if 'description' in attr.metadata:
                        response += f"   📝 {attr.metadata['description']}\n"
                    if 'address' in attr.metadata:
                        response += f"   📍 Адрес: {attr.metadata['address']}\n"
                    response += "\n"
                response += "\n🎯 **Маршрут для осмотра:** Начните с Суздальского кремля, затем посетите Музей деревянного зодчества, завершите день в одном из монастырей."
                return response
            else:
                return "🏛️ Суздаль - музей под открытым небом! Основные достопримечательности:\n\n• **Суздальский кремль** - сердце города с древними соборами\n• **Музей деревянного зодчества** - уникальные памятники архитектуры\n• **Покровский монастырь** - место ссылки знатных женщин\n• **Спасо-Евфимиев монастырь** - монастырь-крепость с богатой историей\n• **Торговые ряды** - архитектурный ансамбль XIX века\n• **Ризоположенский монастырь** - один из древнейших в России\n\n🚶 **Совет:** Город компактный, все主要 достопримечательности в пешей доступности."

        elif any(keyword in question_lower for keyword in ['отель', 'гостиница', 'жилье', 'где остановиться', 'ночлев']):
            hotels = [doc for doc in context_docs if any(hotel_type in doc.metadata.get('type', '').lower() 
                       for hotel_type in ['отель', 'гостиница', 'гостевой дом'])]
            
            if hotels:
                response = "🏨 **Где остановиться в Суздале:**\n\n"
                for i, hotel in enumerate(hotels[:5], 1):
                    response += f"**{i}. {hotel.metadata.get('name', 'Отель')}**\n"
                    if 'address' in hotel.metadata:
                        response += f"   📍 Адрес: {hotel.metadata['address']}\n"
                    if 'description' in hotel.metadata:
                        response += f"   📝 {hotel.metadata['description']}\n"
                    response += "\n"
                response += "\n📞 **Рекомендация:** Бронируйте заранее, особенно в выходные и праздники. Многие отели предлагают экскурсии и трансфер."
                return response
            else:
                return "🏨 Варианты размещения в Суздале:\n\n• **Гостиницы в центре** - удобное расположение, но может быть шумно\n• **Загородные отели** - тишина и природа, нужен транспорт\n• **Гостевые дома** - уютная атмосфера, часто с домашней кухней\n• **Гостиницы при монастырях** - уникальный духовный опыт\n• **Частный сектор** - экономичный вариант с местным колоритом\n\n💰 **Цены:** Средняя стоимость номера 2000-5000 руб/ночь в зависимости от сезона"

        # Общий интеллектуальный ответ
        if context_docs:
            response = f"🤔 По вашему запросу \"{question}\" я нашел следующую информацию:\n\n"
            for i, doc in enumerate(context_docs[:3], 1):
                response += f"**{i}. {doc.metadata.get('name', 'Объект')}**\n"
                if 'description' in doc.metadata:
                    response += f"   {doc.metadata['description']}\n"
                if 'address' in doc.metadata:
                    response += f"   📍 Адрес: {doc.metadata['address']}\n"
                response += "\n"
            response += "\n💡 Если вам нужна более конкретная информация, уточните пожалуйста ваш вопрос."
            return response
        
        # Общий ответ если ничего не найдено
        return "Привет! Я виртуальный гид по Суздалю. 🏛️\n\nЧем могу помочь?\n• Подсказать где поесть 🍽️\n• Посоветовать достопримечательности 🏛️\n• Помочь с выбором жилья 🏨\n• Рассказать об истории города 📖\n\nЗадайте ваш вопрос подробнее, и я постараюсь помочь!"

    except Exception as e:
        logger.error(f"Ошибка генерации ответа: {e}")
        return "Извините, возникла техническая ошибка. Попробуйте задать вопрос еще раз."

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

    # Ищем релевантную информацию
    context_docs = search_in_documents(question, documents, k=Config.RETRIEVER_K)

    # Генерируем ответ
    ai_answer = generate_ai_response(question, context_docs)

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
    global documents, app_initialized
    
    try:
        # Загружаем данные
        documents = load_data()
        
        app_initialized = True
        logger.info(f"Приложение инициализировано, загружено {len(documents)} документов")
        
    except Exception as e:
        logger.critical(f"Критическая ошибка инициализации: {e}")
        logger.error(traceback.format_exc())
        documents = create_fallback_data()
        app_initialized = True

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
        "service": "suzdal_tourism_assistant"
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
