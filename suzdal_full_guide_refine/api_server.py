import os
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, ForeignKey, JSON, text, Boolean
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
    waiting_for_clarification = Column(JSON, nullable=True)
    waiting_for_internet_search = Column(Boolean, default=False)
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
    SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")
    SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

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

def search_internet(query: str) -> Optional[str]:
    """Поиск информации в интернете с использованием Google Search API"""
    try:
        if not Config.SEARCH_API_KEY or not Config.SEARCH_ENGINE_ID:
            logger.warning("API ключи для поиска не настроены")
            return None
            
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': Config.SEARCH_API_KEY,
            'cx': Config.SEARCH_ENGINE_ID,
            'q': f"{query} Суздаль",
            'num': 3
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if 'items' in data and data['items']:
            results = []
            for item in data['items'][:3]:
                title = item.get('title', '')
                snippet = item.get('snippet', '')
                results.append(f"{title}: {snippet}")
            
            return "\n\n".join(results)
        
        return None
        
    except Exception as e:
        logger.error(f"Ошибка поиска в интернете: {e}")
        return None

def needs_clarification(question: str) -> Tuple[bool, Optional[str]]:
    """Определяет, нуждается ли вопрос в уточнении"""
    question_lower = question.lower()
    
    # Общие вопросы, требующие уточнения
    vague_questions = [
        'где поесть', 'что посмотреть', 'куда сходить', 
        'где остановиться', 'что делать', 'что интересного',
        'рекомендации', 'советы', 'посоветуйте'
    ]
    
    for vague_q in vague_questions:
        if vague_q in question_lower:
            if 'где поесть' in question_lower:
                return True, "Какую кухню вы предпочитаете? Или может быть у вас есть предпочтения по бюджету?"
            elif 'что посмотреть' in question_lower or 'куда сходить' in question_lower:
                return True, "Вас больше интересуют исторические достопримечательности, музеи или, может быть, природные красоты?"
            elif 'где остановиться' in question_lower:
                return True, "Какой тип размещения вы предпочитаете? Отель, гостевой дом или что-то другое? И какой у вас бюджет?"
            else:
                return True, "Не могли бы вы уточнить, что именно вас интересует? Это поможет мне дать более точный ответ."
    
    return False, None

def generate_ai_response(question: str, context_docs: List[Document], session_data: Optional[Dict] = None) -> str:
    """Генерация ответа на основе контекста"""
    try:
        question_lower = question.lower()
        
        # Проверяем, ждем ли мы ответ о поиске в интернете
        if session_data and session_data.get('waiting_for_internet_search'):
            if any(word in question_lower for word in ['да', 'yes', 'конечно', 'пожалуйста', 'ищи']):
                # Ищем в интернете
                internet_results = search_internet(session_data.get('last_question', ''))
                if internet_results:
                    return f"🔍 **Результаты поиска в интернете:**\n\n{internet_results}\n\nНадеюсь, эта информация была полезной! 😊"
                else:
                    return "К сожалению, мне не удалось найти дополнительную информацию в интернете по вашему запросу. 😔"
            else:
                return "Хорошо! Я рад, что смог помочь вам с имеющейся информацией. Если у вас будут еще вопросы о Суздале - обращайтесь! 🏛️"
        
        # Проверяем, ждем ли мы уточнения
        if session_data and session_data.get('waiting_for_clarification'):
            # Обрабатываем уточняющий ответ пользователя
            clarified_question = f"{session_data.get('last_question', '')} {question}"
            context_docs = search_in_documents(clarified_question, documents, k=Config.RETRIEVER_K)
            return generate_final_response(clarified_question, context_docs)
        
        # Проверяем, нуждается ли вопрос в уточнении
        needs_clarify, clarification_msg = needs_clarification(question)
        if needs_clarify:
            return f"{generate_preliminary_response(question, context_docs)}\n\n{clarification_msg}"
        
        # Формируем окончательный ответ
        response = generate_final_response(question, context_docs)
        
        # Предлагаем поиск в интернете, если ответ неполный
        if len(context_docs) < 2 and not any(keyword in question_lower for keyword in ['привет', 'hello', 'hi', 'начать']):
            response += "\n\n🤔 Хотите, чтобы я поискал дополнительную информацию в интернете?"
        
        return response
        
    except Exception as e:
        logger.error(f"Ошибка генерации ответа: {e}")
        return "Извините, возникла техническая ошибка. Попробуйте задать вопрос еще раз."

def generate_preliminary_response(question: str, context_docs: List[Document]) -> str:
    """Генерация предварительного ответа"""
    question_lower = question.lower()
    
    if any(keyword in question_lower for keyword in ['где поесть', 'еда', 'ресторан', 'кафе']):
        return "🍽️ В Суздале есть множество прекрасных мест где можно поесть!"
    elif any(keyword in question_lower for keyword in ['достопримечательность', 'что посмотреть', 'музей']):
        return "🏛️ Суздаль богат интересными достопримечательностями!"
    elif any(keyword in question_lower for keyword in ['отель', 'гостиница', 'жилье']):
        return "🏨 В Суздале есть различные варианты размещения!"
    else:
        return "У меня есть информация по вашему запросу!"

def generate_final_response(question: str, context_docs: List[Document]) -> str:
    """Генерация окончательного ответа на основе контекста"""
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
    
    # Специализированные ответы
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
            return response
        else:
            return "🍽️ В Суздале множество мест где можно вкусно поесть. Рекомендую уточнить ваши предпочтения по кухне или бюджету."

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
            return response
        else:
            return "🏛️ Суздаль богат достопримечательностями! Уточните, что именно вас интересует: история, архитектура, музеи?"

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
            return response
        else:
            return "🏨 В Суздале есть различные варианты размещения. Уточните ваш бюджет или предпочтения по типу жилья."

    # Общий ответ
    if context_docs:
        response = f"🤔 По вашему запросу \"{question}\" я нашел:\n\n"
        for i, doc in enumerate(context_docs[:3], 1):
            response += f"**{i}. {doc.metadata.get('name', 'Объект')}**\n"
            if 'description' in doc.metadata:
                response += f"   {doc.metadata['description']}\n"
            if 'address' in doc.metadata:
                response += f"   📍 Адрес: {doc.metadata['address']}\n"
            response += "\n"
        return response
    
    # Общий ответ если ничего не найдено
    return "Привет! Я виртуальный гид по Суздалю. 🏛️\n\nЧем могу помочь?\n• Подсказать где поесть 🍽️\n• Посоветовать достопримечательности 🏛️\n• Помочь с выбором жилья 🏨\n• Рассказать об истории города 📖\n\nЗадайте ваш вопрос подробнее, и я постараюсь помочь!"

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

    # Проверяем, ждем ли мы ответа о поиске в интернете
    if session.waiting_for_internet_search:
        session.waiting_for_internet_search = False
        db.commit()
        
        # Сохраняем ответ пользователя
        db.add(Message(session_id=user_id, role="user", content=question, timestamp=datetime.utcnow()))
        db.commit()
        
        # Генерируем ответ на основе предыдущего вопроса
        context_docs = search_in_documents(session.last_question or "", documents, k=Config.RETRIEVER_K)
        ai_answer = generate_ai_response(question, context_docs, {
            'waiting_for_internet_search': True,
            'last_question': session.last_question
        })
        
        # Сохраняем ответ ассистента
        db.add(Message(session_id=user_id, role="assistant", content=ai_answer, timestamp=datetime.utcnow()))
        db.commit()
        
        return ai_answer

    # Проверяем, ждем ли мы уточнения
    if session.waiting_for_clarification:
        session.waiting_for_clarification = None
        db.commit()
        
        # Сохраняем уточняющий ответ пользователя
        db.add(Message(session_id=user_id, role="user", content=question, timestamp=datetime.utcnow()))
        db.commit()
        
        # Объединяем с предыдущим вопросом
        full_question = f"{session.last_question} {question}"
        context_docs = search_in_documents(full_question, documents, k=Config.RETRIEVER_K)
        ai_answer = generate_ai_response(full_question, context_docs, {
            'waiting_for_clarification': True
        })
        
        # Сохраняем ответ ассистента
        db.add(Message(session_id=user_id, role="assistant", content=ai_answer, timestamp=datetime.utcnow()))
        db.commit()
        
        # Предлагаем поиск в интернете, если ответ неполный
        if len(context_docs) < 2:
            session.waiting_for_internet_search = True
            session.last_question = full_question
            db.commit()
            ai_answer += "\n\n🤔 Хотите, чтобы я поискал дополнительную информацию в интернете?"
            
        return ai_answer

    # Обычная обработка нового вопроса
    session.last_question = question
    db.commit()

    # Сохраняем вопрос пользователя
    db.add(Message(session_id=user_id, role="user", content=question, timestamp=datetime.utcnow()))
    db.commit()

    # Ищем релевантную информацию
    context_docs = search_in_documents(question, documents, k=Config.RETRIEVER_K)

    # Проверяем, нуждается ли вопрос в уточнении
    needs_clarify, clarification_msg = needs_clarification(question)
    if needs_clarify:
        session.waiting_for_clarification = {"original_question": question}
        db.commit()
        
        # Генерируем ответ с просьбой об уточнении
        preliminary_response = generate_preliminary_response(question, context_docs)
        ai_answer = f"{preliminary_response}\n\n{clarification_msg}"
    else:
        # Генерируем обычный ответ
        ai_answer = generate_ai_response(question, context_docs)
        
        # Предлагаем поиск в интернете, если ответ неполный
        if len(context_docs) < 2 and not any(keyword in question.lower() for keyword in ['привет', 'hello', 'hi', 'начать']):
            session.waiting_for_internet_search = True
            db.commit()
            ai_answer += "\n\n🤔 Хотите, чтобы я поискал дополнительную информацию в интернете?"

    # Сохраняем ответ ассистента
    db.add(Message(session_id=user_id, role="assistant", content=ai_answer, timestamp=datetime.utcnow()))
    db.commit()

    return ai_answer

def safe_init_db():
    """Безопасная инициализация базы данных"""
    try:
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        
        if 'chat_sessions' not in existing_tables:
            ChatSession.__table__.create(engine)
            logger.info("Таблица chat_sessions создана")
        
        if 'messages' not in existing_tables:
            Message.__table__.create(engine)
            logger.info("Таблица messages создана")
        
        # Добавляем новые колонки если нужно
        try:
            with engine.connect() as conn:
                # Проверяем и добавляем новые колонки для chat_sessions
                chat_columns = [col['name'] for col in inspector.get_columns('chat_sessions')]
                
                if 'waiting_for_clarification' not in chat_columns:
                    conn.execute(text("ALTER TABLE chat_sessions ADD COLUMN waiting_for_clarification JSONB"))
                
                if 'waiting_for_internet_search' not in chat_columns:
                    conn.execute(text("ALTER TABLE chat_sessions ADD COLUMN waiting_for_internet_search BOOLEAN DEFAULT FALSE"))
                
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
