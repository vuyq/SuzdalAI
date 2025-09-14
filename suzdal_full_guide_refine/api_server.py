import os
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
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

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False)
    role = Column(String(10), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    message_metadata = Column(JSON, nullable=True)

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
    MAX_CONTEXT_LENGTH = 10
    
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
    
    # Минимальное количество букв для осмысленного слова
    MIN_WORD_LENGTH = 3
    
    # Максимальное количество опечаток в слове
    MAX_TYPOS_PER_WORD = 2

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
    except Exception as e:
        logger.error(f"Ошибка загрузки CSV: {e}")
        try:
            df = pd.read_csv(Config.CSV_DATA_URL, on_bad_lines='skip')
        except Exception as e:
            logger.error(f"Критическая ошибка загрузки данных: {e}")
            return []

    documents = []
    for _, row in df.iterrows():
        try:
            content = []
            metadata = {}
            
            for col, val in row.items():
                if pd.isna(val):
                    continue
                    
                if col.lower() in ['name', 'type', 'tags', 'address', 'price', 'hours']:
                    metadata[col.lower()] = str(val)
                else:
                    content.append(f"{col}: {val}")
            
            if not content:
                continue
                
            doc = Document(
                page_content="\n".join(content),
                metadata=metadata
            )
            documents.append(doc)
        except Exception as e:
            logger.warning(f"Ошибка обработки строки {_}: {e}")
    
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
    # Проверяем существование сессии
    session = db.query(ChatSession).filter(ChatSession.id == user_id).first()
    if not session:
        session = ChatSession(id=user_id)
        db.add(session)
        db.commit()
    
    # Добавляем сообщение
    db_message = Message(
        session_id=user_id,
        role=role,
        content=message,
        timestamp=datetime.utcnow()
    )
    db.add(db_message)
    db.commit()

def get_dialog_context(db: Session, user_id: str, max_messages: int = 10) -> str:
    """Получение форматированного контекста диалога из базы данных"""
    messages = db.query(Message).filter(
        Message.session_id == user_id
    ).order_by(
        Message.timestamp.desc()
    ).limit(max_messages).all()
    
    if not messages:
        return ""
    
    # Переворачиваем порядок для хронологического вывода
    messages.reverse()
    return "\n".join(
        f"{msg.role}: {msg.content}" 
        for msg in messages
    )

def get_last_assistant_message(db: Session, user_id: str) -> Optional[str]:
    """Получение последнего сообщения ассистента"""
    last_message = db.query(Message).filter(
        Message.session_id == user_id,
        Message.role == "assistant"
    ).order_by(Message.timestamp.desc()).first()
    
    return last_message.content if last_message else None

def is_clarification_request(text: str) -> bool:
    """Проверяет, является ли текст уточняющим вопросом"""
    if not text:
        return False
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in Config.CLARIFICATION_PHRASES)

def levenshtein_distance(s1: str, s2: str) -> int:
    """Вычисляет расстояние Левенштейна между двумя строками"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def is_gibberish(text: str) -> bool:
    """Проверяет, является ли текст бессмысленным набором символов"""
    # Удаляем все не-буквенные символы
    words = re.findall(r'\b[а-яa-z]+\b', text.lower())
    
    if not words:
        return True
    
    # Проверяем каждое слово
    for word in words:
        if len(word) >= Config.MIN_WORD_LENGTH:
            # Проверяем, есть ли это слово в известных ключевых словах
            found_similar = False
            all_keywords = (Config.FOOD_KEYWORDS + Config.MUSEUM_KEYWORDS + 
                          Config.ATTRACTION_KEYWORDS + Config.ACCOMMODATION_KEYWORDS + 
                          Config.TRANSPORT_KEYWORDS)
            
            for keyword in all_keywords:
                if len(word) >= Config.MIN_WORD_LENGTH and levenshtein_distance(word, keyword) <= Config.MAX_TYPOS_PER_WORD:
                    found_similar = True
                    break
            
            if not found_similar:
                # Если слово длинное и не похоже ни на одно ключевое - вероятно опечатка
                if len(word) > 5:
                    return True
    
    return False

def contains_meaningful_words(text: str) -> bool:
    """Проверяет, содержит ли текст осмысленные слова"""
    words = re.findall(r'\b[а-яa-z]+\b', text.lower())
    
    meaningful_words = 0
    all_keywords = (Config.FOOD_KEYWORDS + Config.MUSEUM_KEYWORDS + 
                  Config.ATTRACTION_KEYWORDS + Config.ACCOMMODATION_KEYWORDS + 
                  Config.TRANSPORT_KEYWORDS + 
                  ["суздаль", "город", "посмотреть", "посетить", "где", "как", "что", "когда"])
    
    for word in words:
        if len(word) >= Config.MIN_WORD_LENGTH:
            for keyword in all_keywords:
                if levenshtein_distance(word, keyword) <= Config.MAX_TYPOS_PER_WORD:
                    meaningful_words += 1
                    break
    
    return meaningful_words >= 1  # Хотя бы одно осмысленное слово

def is_unclear_message(text: str) -> bool:
    """Проверяет, является ли сообщение непонятным или содержащим опечатки"""
    if not text or len(text.strip()) < Config.MIN_QUESTION_LENGTH:
        return True
    
    # Проверяем на бессмысленный текст
    if is_gibberish(text):
        return True
    
    # Проверяем, содержит ли текст осмысленные слова
    if not contains_meaningful_words(text):
        return True
    
    return False

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

def needs_clarification(question: str) -> Tuple[bool, Optional[str]]:
    """Проверяет, нуждается ли вопрос в уточнении"""
    question = question.strip()
    question_lower = question.lower()
    
    # Очень короткие вопросы
    if len(question) < 3:
        return True, "Пожалуйста, уточните ваш вопрос. Например:\n- Какие музеи стоит посетить?\n- Где можно попробовать медовуху?"
    
    # Проверяем, какие параметры уже указаны
    specified_params = {
        'cuisine': any(word in question_lower for word in ['русск', 'итальянск', 'европейск', 'азиатск', 'китайск', 'японск']),
        'budget': any(word in question_lower for word in ['эконом', 'дешев', 'средн', 'премиум', 'дорог']),
        'location': any(word in question_lower for word in ['центр', 'кремл', 'окраин', 'район']),
        'type': any(word in question_lower for word in ['кафе', 'ресторан', 'столов', 'паб', 'бар'])
    }
    
    words = question.split()
    
    # Если запрос содержит только тип кухни или только тип заведения
    if (('итальянск' in question_lower or 'европейск' in question_lower or 
         'русск' in question_lower or 'азиатск' in question_lower) and
        not any(word in question_lower for word in ['бюджет', 'цена', 'стоимость', 'центр', 'район'])):
        
        return True, (
            "Отлично! Теперь, пожалуйста, уточните:\n\n"
            "• Бюджет: какой диапазон цен предпочитаете?\n"
            "• Расположение: в центре города или не важно?\n"
            "• Атмосфера: уютное кафе или ресторан для ужина?\n\n"
            "Что для вас важнее?"
        )
    
    # Если указан только тип заведения
    if (any(word in question_lower for word in ['ресторан', 'кафе', 'столовая']) and
        not specified_params['cuisine'] and not specified_params['budget']):
        
        return True, (
            "Хорошо! Теперь уточните:\n\n"
            "• Кухня: какую кухню предпочитаете?\n"
            "• Бюджет: примерный диапазон цен?\n"
            "• Местоположение: где бы хотели?\n\n"
            "Что для вас в приоритете?"
        )
    
    # Однословные запросы
    if len(words) == 1:
        word = words[0].lower()
        if word in ["еда", "кафе", "ресторан"]:
            return True, get_food_clarification()
        elif word in ["музей", "музеи"]:
            return True, get_museum_clarification()
        elif word in ["достопримечательность", "посмотреть"]:
            return True, get_attraction_clarification()
    
    # Короткие вопросы без деталей (2-3 слова)
    if len(words) <= 3:
        question_type = classify_question(question)
        
        # Если уже есть некоторые параметры, уточняем недостающие
        if question_type == "food":
            missing_params = []
            if not specified_params['budget']:
                missing_params.append("бюджет")
            if not specified_params['location']:
                missing_params.append("расположение")
            if not specified_params['type']:
                missing_params.append("тип заведения")
            
            if missing_params:
                clarification = f"Отлично! Уточните {', '.join(missing_params)}:\n\n"
                if 'бюджет' in missing_params:
                    clarification += "• Бюджет: экономный, средний, премиум?\n"
                if 'расположение' in missing_params:
                    clarification += "• Расположение: центр, рядом с достопримечательностями?\n"
                if 'тип заведения' in missing_params:
                    clarification += "• Тип: кафе, ресторан, паб?\n"
                clarification += "\nЧто для вас важнее?"
                return True, clarification
            
            # Если все параметры указаны, не нужно уточнение
            return False, None
        
        # Для других категорий
        elif question_type == "museum":
            return True, get_museum_clarification()
        elif question_type == "attraction":
            return True, get_attraction_clarification()
    
    return False, None

def get_food_clarification() -> str:
    """Уточняющий вопрос для еды"""
    return (
        "Чтобы порекомендовать подходящие места, уточните:\n\n"
        "• Тип кухни: русская, итальянская, европейская, азиатская?\n"
        "• Бюджет: экономный, средний, премиум?\n"
        "• Расположение: в центре, рядом с кремлем, на окраине?\n"
        "• Тип заведения: кафе, ресторан, столовая, паб?\n\n"
        "Что для вас важнее в первую очередь?"
    )

def get_museum_clarification() -> str:
    """Уточняющий вопрос для музеев"""
    return (
        "Уточните, какие музеи вас интересуют:\n\n"
        "• Исторические - Суздальский кремль, музей деревянного зодчества\n"
        "• Художественные - галереи, иконопись\n"
        "• Тематические - медовухи, огурца, купеческого быта\n"
        "• Архитектурные - монастыри, церкви\n\n"
        "Что вас больше привлекает?"
    )

def get_attraction_clarification() -> str:
    """Уточняющий вопрос для достопримечательностей"""
    return (
        "Уточните, что хотите посмотреть:\n\n"
        "• Архитектура - кремль, монастыри, церкви\n"
        "• История - древние памятники, музеи\n"
        "• Природа - парки, река Каменка\n"
        "• Развлечения - festivals, мастер-классы\n\n"
        "Что вас интересует больше всего?"
    )

def get_accommodation_clarification() -> str:
    """Уточняющий вопрос для размещения"""
    return (
        "Уточните параметры размещения:\n\n"
        "• Бюджет: эконом, средний, luxury?\n"
        "• Тип: отель, гостиница, хостел, квартира?\n"
        "• Расположение: центр, тихий район, рядом с достопримечательностями?\n"
        "• Удобства: WiFi, парковка, завтрак?\n\n"
        "Что для вас наиболее важно?"
    )

def get_transport_clarification() -> str:
    """Уточняющий вопрос для транспорта"""
    return (
        "Уточните, о каком транспорте:\n\n"
        "• До Суздаля - из Москвы, из Владимира\n"
        "• Внутри города - такси, автобусы, пешие маршруты\n"
        "• Аренда - автомобили, велосипеды\n"
        "• Экскурсии - организованные туры\n\n"
        "Что именно вас интересует?"
    )

def is_user_response_to_clarification(db: Session, user_id: str, current_question: str) -> bool:
    """Проверяет, является ли текущий вопрос ответом на уточнение"""
    last_assistant_msg = get_last_assistant_message(db, user_id)
    if not last_assistant_msg or not is_clarification_request(last_assistant_msg):
        return False
    
    current_lower = current_question.lower()
    
    # Проверяем, содержит ли ответ конкретные параметры
    has_specific_answer = any([
        any(word in current_lower for word in ['эконом', 'средн', 'премиум', 'дешёв', 'дорог']),  # бюджет
        any(word in current_lower for word in ['центр', 'кремл', 'окраин', 'район']),  # расположение
        any(word in current_lower for word in ['кафе', 'ресторан', 'столов', 'паб']),  # тип
        any(word in current_lower for word in ['не важно', 'любой', 'всё равно']),  # безразличие
        len(current_question.split()) <= 3  # короткий ответ
    ])
    
    return has_specific_answer

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
    
    response = ["Рекомендую следующие места:\n"]
    for i, doc in enumerate(docs[:5], 1):
        name = doc.metadata.get("name", "Заведение")
        cuisine = doc.metadata.get("type", "кухня не указана")
        address = doc.metadata.get("address", "адрес не указан")
        price = doc.metadata.get("price", "")
        
        response.append(f"{i}. {name}")
        response.append(f"   Кухня: {cuisine}")
        if price:
            response.append(f"   Цены: {price}")
        if address:
            response.append(f"   Адрес: {address}")
        response.append("")
    
    response.append("Рекомендую уточнить часы работы по телефону перед посещением!")
    return "\n".join(response)

def format_museum_response(docs: List[Document]) -> str:
    """Форматирование ответа о музеях"""
    if not docs:
        return "К сожалению, не нашел информации о музеях по вашему запросу."
    
    response = ["Музеи Суздаля:\n"]
    for i, doc in enumerate(docs[:5], 1):
        name = doc.metadata.get("name", "Музей")
        description = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
        address = doc.metadata.get("address", "")
        hours = doc.metadata.get("hours", "")
        
        response.append(f"{i}. {name}")
        response.append(f"   {description}")
        if address:
            response.append(f"   {address}")
        if hours:
            response.append(f"   {hours}")
        response.append("")
    
    return "\n".join(response)

def format_attraction_response(docs: List[Document]) -> str:
    """Форматирование ответа о достопримечательностях"""
    if not docs:
        return "К сожалению, не нашел достопримечательностей по вашему запросу."
    
    response = ["Достопримечательности:\n"]
    for i, doc in enumerate(docs[:5], 1):
        name = doc.metadata.get("name", "Достопримечательность")
        description = doc.page_content[:120] + "..." if len(doc.page_content) > 120 else doc.page_content
        address = doc.metadata.get("address", "")
        
        response.append(f"{i}. {name}")
        response.append(f"   {description}")
        if address:
            response.append(f"   {address}")
        response.append("")
    
    return "\n".join(response)

def format_general_response(docs: List[Document], query: str) -> str:
    """Форматирование общего ответа"""
    if not docs:
        return f"К сожалению, не нашел информации по запросу '{query}'."
    
    response = [f"Результаты по запросу '{query}':\n"]
    for i, doc in enumerate(docs[:5], 1):
        name = doc.metadata.get("name", "Место")
        description = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
        address = doc.metadata.get("address", "")
        
        response.append(f"{i}. {name}")
        response.append(f"   {description}")
        if address:
            response.append(f"   {address}")
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
        
        # Проверяем, является ли сообщение непонятным или содержащим опечатки
        if is_unclear_message(question):
            return "Простите, не очень понял ваше сообщение. Пожалуйста, попробуйте сформулировать вопрос иначе. Например:\n- Какие музеи стоит посетить в Суздале?\n- Где можно поесть традиционную русскую кухню?\n- Как добраться до Суздальского кремля?"
        
        # Проверяем, является ли это ответом на уточнение
        if is_user_response_to_clarification(db, user_id, question):
            response = generate_clarified_response(db, user_id, question)
            update_dialog_context(db, user_id, "assistant", response)
            return response
        
        # Проверяем, нуждается ли вопрос в уточнении
        needs_clarify, clarification_text = needs_clarification(question)
        if needs_clarify:
            update_dialog_context(db, user_id, "user", question)
            update_dialog_context(db, user_id, "assistant", clarification_text)
            return clarification_text
        
        # Получаем предыдущий контекст
        dialog_context = get_dialog_context(db, user_id)
        
        # Поиск в базе знаний
        context_docs = document_retriever.invoke(question)
        
        # Формирование ответа
        if context_docs:
            web_results = perform_web_search(question) if len(context_docs) < 3 else ""
            response = generate_ai_response(question, context_docs, web_results, dialog_context)
        else:
            web_results = perform_web_search(question)
            response = generate_ai_response(question, [], web_results, dialog_context)
        
        # Сохраняем контекст
        update_dialog_context(db, user_id, "user", question)
        update_dialog_context(db, user_id, "assistant", response)
        
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
