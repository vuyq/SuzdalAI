import os
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, ForeignKey, Index
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

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String(10), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    message_metadata = Column(JSON, nullable=True)
    session = relationship("ChatSession", back_populates="messages")

# Создание индексов для улучшения производительности
Index('ix_messages_session_id', Message.session_id)
Index('ix_messages_timestamp', Message.timestamp)
Index('ix_messages_session_role', Message.session_id, Message.role)

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
    RETRIEVER_K = 5
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
    
    # Категории для правильного отображения
    FOOD_CATEGORIES = ["ресторан", "кафе", "столовая", "паб", "бар", "трактир", "чайная", "кофейня"]
    MUSEUM_CATEGORIES = ["музей", "экспозиция", "галерея", "выставка"]
    ACCOMMODATION_CATEGORIES = ["отель", "гостиница", "хостел", "гостевой дом", "апартаменты", "номер"]
    ATTRACTION_CATEGORIES = ["смотровая площадка", "парк", "монастырь", "церковь", "собор", "кремль", "памятник"]
    
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

@retry(stop=stop_after_attempt(Config.MAX_RETRIES), 
      wait=wait_exponential(multiplier=1, min=2, max=10))
def get_gigachat_token() -> str:
    """Получение токена доступа GigaChat"""
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': str(uuid.uuid4()),
        'Authorization': f'Basic {Config.GIGACHAT_AUTH}'
    }
    payload = {'scope': 'GIGACHAT_API_PERS'}
    
    try:
        response = requests.post(
            url, 
            headers=headers, 
            data=payload, 
            verify=Config.CERT_PATH,
            timeout=Config.REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return response.json().get("access_token")
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка получения токена: {e}")
        raise

def initialize_models() -> Tuple[GigaChatEmbeddings, GigaChat]:
    """Инициализация моделей GigaChat"""
    try:
        access_token = get_gigachat_token()
        logger.info("Токен GigaChat получен успешно")
        
        embedding_model = GigaChatEmbeddings(
            access_token=access_token,
            model="Embeddings",
            scope="GIGACHAT_API_PERS",
            verify_ssl_certs=bool(Config.CERT_PATH),
            ca_bundle_file=Config.CERT_PATH if Config.CERT_PATH and Path(Config.CERT_PATH).exists() else None,
            timeout=Config.REQUEST_TIMEOUT
        )
        
        ai_assistant = GigaChat(
            access_token=access_token,
            model="GigaChat",
            temperature=0.2,
            verify_ssl_certs=bool(Config.CERT_PATH),
            ca_bundle_file=Config.CERT_PATH if Config.CERT_PATH and Path(Config.CERT_PATH).exists() else None,
            timeout=Config.REQUEST_TIMEOUT,
            verbose=True
        )
        
        return embedding_model, ai_assistant
    except Exception as e:
        logger.error(f"Ошибка инициализации моделей: {e}")
        raise

def load_data() -> List[Document]:
    """Загрузка и обработка данных о достопримечательностях"""
    try:
        df = pd.read_csv(Config.CSV_DATA_URL, sep=';', on_bad_lines='skip')
        logger.info(f"Загружен CSV с {len(df)} строками и {len(df.columns)} колонками")
        logger.info(f"Колонки: {list(df.columns)}")
    except Exception as e:
        logger.error(f"Ошибка загрузки CSV с разделителем ';': {e}")
        try:
            df = pd.read_csv(Config.CSV_DATA_URL, on_bad_lines='skip')
            logger.info(f"Загружен CSV с разделителем ',' с {len(df)} строками")
        except Exception as e:
            logger.error(f"Критическая ошибка загрузки данных: {e}")
            return []

    documents = []
    for _, row in df.iterrows():
        try:
            content = []
            metadata = {}
            
            # Логируем первую строку для отладки
            if _ == 0:
                logger.info(f"Первая строка данных: {dict(row)}")
            
            for col, val in row.items():
                if pd.isna(val) or str(val).strip() == '':
                    continue
                    
                # Сохраняем все основные метаданные
                col_lower = col.lower()
                if col_lower in ['name', 'title', 'название']:
                    metadata['name'] = str(val)
                elif col_lower in ['type', 'тип', 'category', 'категория']:
                    metadata['type'] = str(val)
                elif col_lower in ['address', 'адрес', 'location', 'локация']:
                    metadata['address'] = str(val)
                elif col_lower in ['price', 'цена', 'стоимость']:
                    metadata['price'] = str(val)
                elif col_lower in ['hours', 'время', 'работа', 'часы']:
                    metadata['hours'] = str(val)
                elif col_lower in ['description', 'описание', 'info', 'информация']:
                    metadata['description'] = str(val)
                elif col_lower in ['tags', 'теги']:
                    metadata['tags'] = str(val)
                else:
                    content.append(f"{col}: {val}")
            
            # Если нет имени, пропускаем
            if 'name' not in metadata:
                continue
                
            doc = Document(
                page_content="\n".join(content) if content else metadata.get('description', ''),
                metadata=metadata
            )
            documents.append(doc)
        except Exception as e:
            logger.warning(f"Ошибка обработки строки {_}: {e}")
    
    logger.info(f"Создано {len(documents)} документов")
    return documents

def perform_web_search(query: str) -> str:
    """Выполнение веб-поиска через DuckDuckGo"""
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(
                f"{query} Суздаль", 
                max_results=Config.SEARCH_RESULTS,
                timelimit='y'
            ):
                results.append(f"• {r['title']}\n  {r['href']}\n  {r['body'][:200]}...")
        
        return "\n\n".join(results) if results else "Не найдено результатов"
    except Exception as e:
        logger.error(f"Ошибка веб-поиска: {e}")
        return "Ошибка при выполнении поиска"

def update_dialog_context(db: Session, user_id: str, role: str, message: str):
    """Обновление контекста диалога в базе данных"""
    try:
        # Проверяем существование сессии
        session = db.query(ChatSession).filter(ChatSession.id == user_id).first()
        if not session:
            session = ChatSession(id=user_id)
            db.add(session)
            db.commit()
            db.refresh(session)
        
        # Добавляем сообщение
        db_message = Message(
            session_id=user_id,
            role=role,
            content=message,
            timestamp=datetime.utcnow()
        )
        db.add(db_message)
        db.commit()
        db.refresh(db_message)
        
        logger.info(f"Сохранили сообщение для сессии {user_id}: {role} - {message[:100]}...")
        
    except Exception as e:
        logger.error(f"Ошибка обновления контекста: {e}")
        db.rollback()
        raise

def get_dialog_context(db: Session, user_id: str, max_messages: int = 10) -> str:
    """Получение форматированного контекста диалога из базы данных"""
    try:
        # Проверяем существование сессии
        session = db.query(ChatSession).filter(ChatSession.id == user_id).first()
        if not session:
            return ""
        
        messages = db.query(Message).filter(
            Message.session_id == user_id
        ).order_by(
            Message.timestamp.asc()  # Хронологический порядок
        ).limit(max_messages).all()
        
        if not messages:
            return ""
        
        context = "\n".join(
            f"{msg.role}: {msg.content}" 
            for msg in messages
        )
        
        logger.info(f"Получен контекст для {user_id}: {len(messages)} сообщений")
        return context
    
    except Exception as e:
        logger.error(f"Ошибка получения контекста: {e}")
        return ""

def cleanup_old_messages(db: Session, user_id: str, keep_last: int = Config.MAX_MESSAGES_TO_KEEP):
    """Очистка старых сообщений для предотвращения переполнения"""
    try:
        # Получаем общее количество сообщений
        total_count = db.query(Message).filter(Message.session_id == user_id).count()
        
        if total_count > keep_last:
            # Находим ID сообщений, которые нужно сохранить (последние keep_last)
            recent_messages = db.query(Message.id).filter(
                Message.session_id == user_id
            ).order_by(
                Message.timestamp.desc()
            ).limit(keep_last).all()
            
            recent_ids = [msg.id for msg in recent_messages]
            
            # Удаляем старые сообщения
            deleted_count = db.query(Message).filter(
                Message.session_id == user_id,
                Message.id.notin_(recent_ids)
            ).delete(synchronize_session=False)
            
            db.commit()
            logger.info(f"Удалено {deleted_count} старых сообщений для сессии {user_id}")
    
    except Exception as e:
        logger.error(f"Ошибка очистки сообщений: {e}")
        db.rollback()

def get_last_assistant_message(db: Session, user_id: str) -> Optional[str]:
    """Получение последнего сообщения ассистента"""
    try:
        last_message = db.query(Message).filter(
            Message.session_id == user_id,
            Message.role == "assistant"
        ).order_by(Message.timestamp.desc()).first()
        
        return last_message.content if last_message else None
    except Exception as e:
        logger.error(f"Ошибка получения последнего сообщения: {e}")
        return None

def is_clarification_request(text: str) -> bool:
    """Проверяет, является ли текст уточняющим вопросом"""
    if not text:
        return False
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in Config.CLARIFICATION_PHRASES)

def classify_question(question: str) -> str:
    """Классификация вопроса по категориям"""
    question_lower = question.lower()
    
    if any(keyword in question_lower for keyword in Config.FOOD_KEYWORDS):
        return "food"
    elif any(keyword in question_lower for keyword in Config.MUSEUM_KEYWORDS):
        return "museum"
    elif any(keyword in question_lower for keyword in Config.ATTRACTION_KEYWORDS):
        return "attraction"
    elif any(keyword in question_lower for keyword in Config.ACCOMMODATION_KEYWORDS):
        return "accommodation"
    elif any(keyword in question_lower for keyword in Config.TRANSPORT_KEYWORDS):
        return "transport"
    return "general"

def get_category_from_type(type_str: str) -> str:
    """Определяет категорию из типа заведения"""
    if not type_str:
        return "Место"
    
    type_lower = type_str.lower()
    
    if any(cat in type_lower for cat in Config.FOOD_CATEGORIES):
        return "🍽️ Ресторан/Кафе"
    elif any(cat in type_lower for cat in Config.MUSEUM_CATEGORIES):
        return "🏛️ Музей"
    elif any(cat in type_lower for cat in Config.ACCOMMODATION_CATEGORIES):
        return "🏨 Гостиница"
    elif any(cat in type_lower for cat in Config.ATTRACTION_CATEGORIES):
        return "🏰 Достопримечательность"
    else:
        return "📍 Место"

def needs_clarification(question: str) -> Tuple[bool, Optional[str]]:
    """Проверяет, нуждается ли вопрос в уточнении"""
    question = question.strip()
    question_lower = question.lower()
    
    # Очень короткие вопросы
    if len(question) < 3:
        return True, "Пожалуйста, уточните ваш вопрос. Например:\n- Какие музеи стоит посетить?\n- Где можно попробовать медовуху?"
    
    # Однословные запросы
    words = question.split()
    if len(words) == 1:
        word = words[0].lower()
        if word in ["еда", "кафе", "ресторан"]:
            return True, get_food_clarification()
        elif word in ["музей", "музеи"]:
            return True, get_museum_clarification()
        elif word in ["достопримечательность", "посмотреть"]:
            return True, get_attraction_clarification()
        elif word in ["отель", "гостиница"]:
            return True, get_accommodation_clarification()
        elif word in ["транспорт", "добраться"]:
            return True, get_transport_clarification()
    
    # Короткие вопросы без деталей
    if len(words) <= 3:
        question_type = classify_question(question)
        if question_type == "food":
            return True, get_food_clarification()
        elif question_type == "museum":
            return True, get_museum_clarification()
        elif question_type == "attraction":
            return True, get_attraction_clarification()
        elif question_type == "accommodation":
            return True, get_accommodation_clarification()
        elif question_type == "transport":
            return True, get_transport_clarification()
    
    return False, None

def get_food_clarification() -> str:
    """Уточняющий вопрос для еды"""
    return (
        "🍽️ Чтобы порекомендовать подходящие места, уточните:\n\n"
        "• **Тип кухни**: русская, итальянская, европейская, азиатская?\n"
        "• **Бюджет**: экономный, средний, премиум?\n"
        "• **Расположение**: в центре, рядом с кремлем, на окраине?\n"
        "• **Тип заведения**: кафе, ресторан, столовая, паб?\n\n"
        "Что для вас важнее в первую очередь?"
    )

def get_museum_clarification() -> str:
    """Уточняющий вопрос для музеев"""
    return (
        "🏛️ Уточните, какие музеи вас интересуют:\n\n"
        "• **Исторические** - Суздальский кремль, музей деревянного зодчества\n"
        "• **Художественные** - галереи, иконопись\n"
        "• **Тематические** - медовухи, огурца, купеческого быта\n"
        "• **Архитектурные** - монастыри, церкви\n\n"
        "Что вас больше привлекает?"
    )

def get_attraction_clarification() -> str:
    """Уточняющий вопрос для достопримечательности"""
    return (
        "🏰 Уточните, что хотите посмотреть:\n\n"
        "• **Архитектура** - кремль, монастыри, церкви\n"
        "• **История** - древние памятники, музеи\n"
        "• **Природа** - парки, река Каменка\n"
        "• **Развлечения** - festivals, мастер-классы\n\n"
        "Что вас интересует больше всего?"
    )

def get_accommodation_clarification() -> str:
    """Уточняющий вопрос для размещения"""
    return (
        "🏨 Уточните параметры размещения:\n\n"
        "• **Бюджет**: эконом, средний, luxury?\n"
        "• **Тип**: отель, гостиница, хостел, квартира?\n"
        "• **Расположение**: центр, тихий район, рядом с достопримечательностями?\n"
        "• **Удобства**: WiFi, парковка, завтрак?\n\n"
        "Что для вас наиболее важно?"
    )

def get_transport_clarification() -> str:
    """Уточняющий вопрос для транспорта"""
    return (
        "🚗 Уточните, о каком транспорте:\n\n"
        "• **До Суздаля** - из Москвы, из Владимира\n"
        "• **Внутри города** - такси, автобусы, пешие маршруты\n"
        "• **Аренда** - автомобили, велосипеды\n"
        "• **Экскурсии** - организованные туры\n\n"
        "Что именно вас интересует?"
    )

def is_user_response_to_clarification(db: Session, user_id: str, current_question: str) -> bool:
    """Проверяет, является ли текущий вопрос ответом на уточнение"""
    last_assistant_msg = get_last_assistant_message(db, user_id)
    if not last_assistant_msg or not is_clarification_request(last_assistant_msg):
        return False
    
    # Проверяем, что пользователь отвечает на уточнение, а не задает новый вопрос
    current_lower = current_question.lower()
    
    # Если пользователь просто повторяет тот же короткий вопрос
    if len(current_question.split()) <= 2 and classify_question(current_question) == classify_question(last_assistant_msg):
        return True
    
    # Если пользователь дает конкретный ответ на уточнение
    clarification_patterns = [
        r"(русск|итальянск|европейск|азиатск)[а-я]* кухн",
        r"(эконом|средн|премиум|деш[её]в|дорог)[а-я]*",
        r"(центр|кремл|окраин)[а-я]*",
        r"(кафе|ресторан|столов|паб|бар)[а-я]*",
        r"(историческ|художествен|тематическ|архитектурн)[а-я]*",
        r"(отель|гостиниц|хостел|квартир)[а-я]*",
        r"(москв|владимир|поезд|автобус|такси|машин)[а-я]*"
    ]
    
    return any(re.search(pattern, current_lower) for pattern in clarification_patterns)

def generate_clarified_response(db: Session, user_id: str, clarification: str) -> str:
    """Генерация ответа на уточняющую информацию"""
    try:
        # Получаем последний уточняющий вопрос
        last_clarification = get_last_assistant_message(db, user_id)
        
        # Ищем первоначальный запрос (перед уточнением)
        user_messages = db.query(Message).filter(
            Message.session_id == user_id,
            Message.role == "user"
        ).order_by(Message.timestamp.desc()).all()
        
        original_question = None
        for msg in user_messages:
            if not is_clarification_request(msg.content):
                original_question = msg.content
                break
        
        if not original_question:
            # Если не нашли оригинальный вопрос, используем уточнение как основной запрос
            combined_query = clarification
        else:
            # Комбинируем оригинальный вопрос и уточнение
            combined_query = f"{original_question} {clarification}"
        
        # Поиск в базе знаний
        docs = document_retriever.invoke(combined_query)
        
        if docs:
            question_type = classify_question(combined_query)
            if question_type == "food":
                return format_food_response(docs)
            elif question_type == "museum":
                return format_museum_response(docs)
            elif question_type == "attraction":
                return format_attraction_response(docs)
            elif question_type == "accommodation":
                return format_accommodation_response(docs)
            else:
                return format_general_response(docs, combined_query)
        
        # Если в базе нет результатов - ищем в интернете
        web_results = perform_web_search(combined_query)
        if "Не найдено" not in web_results:
            return f"Вот что я нашел в интернете по вашему запросу:\n\n{web_results}"
        
        return "К сожалению, не нашел конкретной информации по вашим критериям. Попробуйте изменить параметры поиска или задайте вопрос по-другому."
    
    except Exception as e:
        logger.error(f"Ошибка генерации уточненного ответа: {e}")
        return "Не удалось обработать ваш запрос. Пожалуйста, попробуйте еще раз."

def format_food_response(docs: List[Document]) -> str:
    """Форматирование ответа о местах питания"""
    if not docs:
        return "К сожалению, не нашел подходящих мест по вашему запросу."
    
    response = ["🍽️ **Рестораны и кафе Суздаля:**\n"]
    for i, doc in enumerate(docs[:5], 1):
        name = doc.metadata.get("name", "Заведение")
        place_type = doc.metadata.get("type", "тип не указан")
        address = doc.metadata.get("address", "адрес не указан")
        price = doc.metadata.get("price", "")
        description = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
        
        response.append(f"{i}. **{name}**")
        response.append(f"   🏷️ Тип: {place_type}")
        if description and description.strip():
            response.append(f"   📖 {description}")
        if price:
            response.append(f"   💰 Цены: {price}")
        if address:
            response.append(f"   📍 Адрес: {address}")
        response.append("")
    
    response.append("💡 Рекомендую уточнить часы работы по телефону перед посещением!")
    return "\n".join(response)

def format_museum_response(docs: List[Document]) -> str:
    """Форматирование ответа о музеях"""
    if not docs:
        return "К сожалению, не нашел информации о музеях по вашему запросу."
    
    response = ["🏛️ **Музеи Суздаля:**\n"]
    for i, doc in enumerate(docs[:5], 1):
        name = doc.metadata.get("name", "Музей")
        museum_type = doc.metadata.get("type", "музей")
        description = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
        address = doc.metadata.get("address", "")
        hours = doc.metadata.get("hours", "")
        price = doc.metadata.get("price", "")
        
        response.append(f"{i}. **{name}**")
        response.append(f"   🏷️ Тип: {museum_type}")
        if description and description.strip():
            response.append(f"   📖 {description}")
        if address:
            response.append(f"   📍 Адрес: {address}")
        if hours:
            response.append(f"   🕒 Часы работы: {hours}")
        if price:
            response.append(f"   💰 Стоимость: {price}")
        response.append("")
    
    return "\n".join(response)

def format_attraction_response(docs: List[Document]) -> str:
    """Форматирование ответа о достопримечательностях"""
    if not docs:
        return "К сожалению, не нашел достопримечательностей по вашему запросу."
    
    response = ["🏰 **Достопримечательности Суздаля:**\n"]
    for i, doc in enumerate(docs[:5], 1):
        name = doc.metadata.get("name", "Достопримечательность")
        attraction_type = doc.metadata.get("type", "достопримечательность")
        description = doc.page_content[:120] + "..." if len(doc.page_content) > 120 else doc.page_content
        address = doc.metadata.get("address", "")
        hours = doc.metadata.get("hours", "")
        price = doc.metadata.get("price", "")
        
        response.append(f"{i}. **{name}**")
        response.append(f"   🏷️ Тип: {attraction_type}")
        if description and description.strip():
            response.append(f"   📖 {description}")
        if address:
            response.append(f"   📍 Адрес: {address}")
        if hours:
            response.append(f"   🕒 Часы работы: {hours}")
        if price:
            response.append(f"   💰 Стоимость: {price}")
        response.append("")
    
    return "\n".join(response)

def format_accommodation_response(docs: List[Document]) -> str:
    """Форматирование ответа о размещении"""
    if not docs:
        return "К сожалению, не нашел вариантов размещения по вашему запросу."
    
    response = ["🏨 **Варианты размещения в Суздале:**\n"]
    for i, doc in enumerate(docs[:5], 1):
        name = doc.metadata.get("name", "Гостиница")
        accommodation_type = doc.metadata.get("type", "гостиница")
        description = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
        address = doc.metadata.get("address", "")
        price = doc.metadata.get("price", "")
        
        response.append(f"{i}. **{name}**")
        response.append(f"   🏷️ Тип: {accommodation_type}")
        if description and description.strip():
            response.append(f"   📖 {description}")
        if address:
            response.append(f"   📍 Адрес: {address}")
        if price:
            response.append(f"   💰 Стоимость: {price}")
        response.append("")
    
    return "\n".join(response)

def format_general_response(docs: List[Document], query: str) -> str:
    """Форматирование общего ответа"""
    if not docs:
        return f"К сожалению, не нашел информации по запросу '{query}'."
    
    response = [f"**Результаты по запросу '{query}':**\n"]
    for i, doc in enumerate(docs[:5], 1):
        name = doc.metadata.get("name", "Место")
        place_type = doc.metadata.get("type", "тип не указан")
        description = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
        address = doc.metadata.get("address", "")
        
        category = get_category_from_type(place_type)
        
        response.append(f"{i}. **{name}**")
        response.append(f"   {category}")
        if description and description.strip():
            response.append(f"   📖 {description}")
        if address:
            response.append(f"   📍 Адрес: {address}")
        response.append("")
    
    return "\n".join(response)

TOURISM_PROMPT_TEMPLATE = """
Ты виртуальный гид по Суздалю. Отвечай на вопросы информативно и дружелюбно.

[Контекст диалога]:
{dialog_context}

[Данные из базы]:
{context}

[Веб-результаты]:
{web_search}

[Вопрос]:
{question}

Ответь подробно и полезно, предлагая конкретные рекомендации.
Если информации недостаточно - вежливо предложи уточнить запрос.
"""

tourism_prompt = PromptTemplate.from_template(TOURISM_PROMPT_TEMPLATE)

def generate_ai_response(question: str, context_docs: List[Document], 
                        web_results: str, dialog_context: str) -> str:
    """Генерация ответа с помощью GigaChat"""
    try:
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
        return "Не удалось обработать запрос с помощью AI. Вот что я нашел:\n" + format_general_response(context_docs, question)

def handle_question(db: Session, question: str, user_id: str) -> str:
    """Основная обработка вопроса"""
    try:
        question = question.strip()
        if not question:
            return "Пожалуйста, задайте ваш вопрос о Суздале."
        
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
        
        # Сохраняем ответ ассистента
        update_dialog_context(db, user_id, "assistant", response)
        cleanup_old_messages(db, user_id)
        
        return response
    
    except Exception as e:
        logger.error(f"Ошибка обработки вопроса: {e}")
        return "Извините, возникла техническая ошибка. Пожалуйста, попробуйте задать вопрос позже."

# Инициализация приложения
try:
    download_certificate()
    embedding_model, ai_assistant = initialize_models()
    documents = load_data()
    
    if not documents:
        raise Exception("Не удалось загрузить данные о достопримечательностях")
    
    vector_store = FAISS.from_documents(documents, embedding_model)
    document_retriever = vector_store.as_retriever(
        search_kwargs={"k": Config.RETRIEVER_K}
    )
    logger.info("Модели и данные успешно инициализированы")
    
except Exception as e:
    logger.critical(f"Ошибка инициализации: {e}")
    # Создаем заглушки для продолжения работы
    documents = []

app = FastAPI(title="Суздаль Tourism Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str
    user_id: str = "default"

@app.post("/ask")
async def ask(item: Question, db: Session = Depends(get_db)):
    try:
        if not item.question.strip():
            return {"answer": "Пожалуйста, задайте ваш вопрос."}
        
        response = handle_question(db, item.question, item.user_id)
        return {"answer": response}
    
    except Exception as e:
        logger.error(f"API error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Внутренняя ошибка сервера. Пожалуйста, попробуйте позже."
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.get("/history/{user_id}")
async def get_history(user_id: str, db: Session = Depends(get_db)):
    """Получить историю сообщений для пользователя"""
    try:
        messages = db.query(Message).filter(
            Message.session_id == user_id
        ).order_by(Message.timestamp.asc()).all()
        
        return {
            "user_id": user_id,
            "message_count": len(messages),
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat()
                }
                for msg in messages
            ]
        }
    
    except Exception as e:
        logger.error(f"Ошибка получения истории: {e}")
        raise HTTPException(status_code=500, detail="Ошибка получения истории")

@app.delete("/history/{user_id}")
async def clear_history(user_id: str, db: Session = Depends(get_db)):
    """Очистить историю сообщений для пользователя"""
    try:
        # Удаляем все сообщения пользователя
        message_count = db.query(Message).filter(Message.session_id == user_id).delete()
        
        # Удаляем сессию
        session_count = db.query(ChatSession).filter(ChatSession.id == user_id).delete()
        
        db.commit()
        
        return {
            "user_id": user_id,
            "deleted_messages": message_count,
            "deleted_sessions": session_count
        }
    
    except Exception as e:
        logger.error(f"Ошибка очистки истории: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Ошибка очистки истории")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
