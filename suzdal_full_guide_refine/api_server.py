import os
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import JSON
from datetime import datetime
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_gigachat import GigaChat, GigaChatEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential
from ddgs import DDGS
import logging
from typing import Dict, List, Optional, Tuple
import uuid
import re

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

# Настройки базы данных
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
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    feedbacks = relationship("Feedback", back_populates="session", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String(10), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    message_metadata = Column(JSON, nullable=True)
    session = relationship("ChatSession", back_populates="messages")

class Feedback(Base):
    __tablename__ = "feedbacks"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), ForeignKey("chat_sessions.id"), nullable=False)
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=False)
    is_helpful = Column(Boolean, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    session = relationship("ChatSession", back_populates="feedbacks")
    message = relationship("Message")

# Создание таблиц
Base.metadata.create_all(bind=engine)

# Dependency для получения сессии БД
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class Config:
    # Основные настройки
    MAX_RETRIES = 3
    REQUEST_TIMEOUT = 15
    MIN_QUESTION_LENGTH = 2
    SEARCH_RESULTS = 3
    RETRIEVER_K = 10  # Увеличиваем количество извлекаемых документов
    MAX_CONTEXT_LENGTH = 20
    MAX_MESSAGES_TO_KEEP = 50
    
    # Пути и URL
    CERT_PATH = os.getenv("CERT_PATH", "./cert.pem")
    CERT_URL = os.getenv("CERT_URL")
    GIGACHAT_AUTH = os.getenv("GIGACHAT_AUTH")
    CSV_DATA_URL = os.getenv("CSV_DATA_URL", "https://raw.githubusercontent.com/vuyq/SuzdalAI/main/suzdal_full_guide_refine/attractions.csv")
    
    # Ключевые слова
    FOOD_KEYWORDS = ["еда", "поесть", "кафе", "ресторан", "перекусить", "кухня", "столовая", "меню", "завтрак", "обед", "ужин"]
    MUSEUM_KEYWORDS = ["музей", "музеи", "экспозиция", "выставка", "галерея", "коллекция"]
    ATTRACTION_KEYWORDS = ["достопримечательность", "посмотреть", "посетить", "интересное", "место", "архитектура", "памятник"]
    ACCOMMODATION_KEYWORDS = ["отель", "гостиница", "хостел", "номер", "жилье", "размещение", "ночлег"]
    TRANSPORT_KEYWORDS = ["транспорт", "добраться", "автобус", "поезд", "такси", "маршрут"]
    
    # Фразы для определения уточнений
    CLARIFICATION_PHRASES = [
        "что для вас важнее",
        "уточните пожалуйста",
        "по каким критериям",
        "что предпочитаете",
        "какой вариант выбрать",
        "какая кухня",
        "какой бюджет",
        "где расположение"
    ]

def download_certificate():
    """Загрузка SSL-сертификата при необходимости"""
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

# ... (остальные функции остаются без изменений до функции handle_question)

def format_detailed_response(docs: List[Document], query: str) -> str:
    """Форматирование подробного ответа с максимальной информацией из базы"""
    if not docs:
        return f"К сожалению, не нашел информации по запросу '{query}'."
    
    response = [f"**Вот подробная информация по вашему запросу '{query}':**\n\n"]
    
    for i, doc in enumerate(docs[:7], 1):  # Увеличиваем количество выводимых мест
        name = doc.metadata.get("name", "Место")
        type_info = doc.metadata.get("type", "")
        address = doc.metadata.get("address", "")
        price = doc.metadata.get("price", "")
        hours = doc.metadata.get("hours", "")
        description = doc.page_content
        
        response.append(f"🏛️ **{name}**")
        if type_info:
            response.append(f"   • Тип: {type_info}")
        if description and len(description) > 50:
            response.append(f"   • Описание: {description[:250]}...")  # Увеличиваем длину описания
        elif description:
            response.append(f"   • Описание: {description}")
        if address:
            response.append(f"   • Адрес: {address}")
        if price:
            response.append(f"   • Цены: {price}")
        if hours:
            response.append(f"   • Часы работы: {hours}")
        response.append("")
    
    response.append("ℹ️ *Рекомендую уточнить актуальную информацию перед посещением!*")
    return "\n".join(response)

def generate_ai_response(question: str, context_docs: List[Document], 
                        web_results: str, dialog_context: str) -> str:
    """Генерация ответа с помощью GigaChat"""
    try:
        # Сначала показываем подробную информацию из базы
        detailed_info = format_detailed_response(context_docs, question)
        
        # Затем добавляем AI-ответ если есть дополнительные данные
        if web_results and "Не найдено" not in web_results:
            prompt_input = {
                "question": question,
                "context": detailed_info,
                "web_search": web_results,
                "dialog_context": dialog_context
            }
            
            ai_response = ai_assistant.invoke(tourism_prompt.format(**prompt_input))
            return f"{detailed_info}\n\n🔍 **Дополнительная информация из интернета:**\n{ai_response.content if hasattr(ai_response, 'content') else str(ai_response)}"
        
        return detailed_info
    
    except Exception as e:
        logger.error(f"Ошибка генерации ответа: {e}")
        return format_detailed_response(context_docs, question)

def handle_feedback(db: Session, user_id: str, message_id: int, is_helpful: bool) -> str:
    """Обработка фидбека от пользователя"""
    try:
        # Проверяем существование сессии и сообщения
        session = db.query(ChatSession).filter(ChatSession.id == user_id).first()
        if not session:
            return "Сессия не найдена"
        
        message = db.query(Message).filter(
            Message.id == message_id,
            Message.session_id == user_id,
            Message.role == "assistant"
        ).first()
        
        if not message:
            return "Сообщение не найдено"
        
        # Сохраняем фидбек
        feedback = Feedback(
            session_id=user_id,
            message_id=message_id,
            is_helpful=is_helpful
        )
        db.add(feedback)
        db.commit()
        
        # Формируем ответ в зависимости от фидбека
        if is_helpful:
            return "Рад, что смог помочь! 😊 Если у вас есть еще вопросы о Суздале - с удовольствием на них отвечу!"
        else:
            return "Извините, что ответ не был полезен. 😔 Пожалуйста, уточните ваш вопрос или задайте его по-другому, и я постараюсь помочь лучше!"
    
    except Exception as e:
        logger.error(f"Ошибка обработки фидбека: {e}")
        db.rollback()
        return "Ошибка обработки отзыва"

def handle_question(db: Session, question: str, user_id: str) -> str:
    """Основная обработка вопроса"""
    try:
        question = question.strip()
        if not question:
            return "Пожалуйста, задайте ваш вопрос о Суздале."
        
        # Проверяем, является ли вопрос фидбеком
        if question.lower() in ['да', 'нет', 'yes', 'no']:
            # Ищем последнее сообщение ассистента
            last_assistant_msg = db.query(Message).filter(
                Message.session_id == user_id,
                Message.role == "assistant"
            ).order_by(Message.timestamp.desc()).first()
            
            if last_assistant_msg:
                is_helpful = question.lower() in ['да', 'yes']
                feedback_response = handle_feedback(db, user_id, last_assistant_msg.id, is_helpful)
                return feedback_response
        
        # Сохраняем вопрос пользователя
        update_dialog_context(db, user_id, "user", question)
        
        # Проверяем, является ли это ответом на уточнение
        if is_user_response_to_clarification(db, user_id, question):
            response = generate_clarified_response(db, user_id, question)
            update_dialog_context(db, user_id, "assistant", response)
            cleanup_old_messages(db, user_id)
            return response
        
        # Проверяем, нуждается ли вопрос в уточнении
        needs_clarify, clarification_text = needs_clarification(question)
        if needs_clarify:
            update_dialog_context(db, user_id, "assistant", clarification_text)
            cleanup_old_messages(db, user_id)
            return clarification_text
        
        # Получаем предыдущий контекст
        dialog_context = get_dialog_context(db, user_id)
        logger.info(f"Диалоговый контекст для {user_id}: {len(dialog_context.splitlines())} сообщений")
        
        # Поиск в базе знаний
        context_docs = document_retriever.invoke(question)
        
        # Формирование ответа
        if context_docs:
            web_results = perform_web_search(question) if len(context_docs) < 3 else ""
            response = generate_ai_response(question, context_docs, web_results, dialog_context)
        else:
            web_results = perform_web_search(question)
            response = generate_ai_response(question, [], web_results, dialog_context)
        
        # Добавляем запрос фидбека к ответу
        response_with_feedback = f"{response}\n\n---\n*Был ли этот ответ полезен? (да/нет)*"
        
        # Сохраняем ответ ассистента
        update_dialog_context(db, user_id, "assistant", response_with_feedback)
        cleanup_old_messages(db, user_id)
        
        return response_with_feedback
    
    except Exception as e:
        logger.error(f"Ошибка обработки вопроса: {e}")
        return "Извините, возникла техническая ошибка. Пожалуйста, попробуйте задать вопрос позже."

# ... (остальной код остается без изменений)

class FeedbackRequest(BaseModel):
    message_id: int
    is_helpful: bool
    user_id: str = "default"

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest, db: Session = Depends(get_db)):
    """Эндпоинт для отправки фидбека"""
    try:
        response = handle_feedback(db, feedback.user_id, feedback.message_id, feedback.is_helpful)
        return {"answer": response}
    
    except Exception as e:
        logger.error(f"API feedback error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Ошибка обработки отзыва"
        )

# ... (остальной код приложения)
