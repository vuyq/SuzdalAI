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
    
    # Ключевые слова для классификации
    FOOD_KEYWORDS = ["еда", "поесть", "кафе", "ресторан", "перекусить", "кухня", "столовая", "меню", "завтрак", "обед", "ужин", "блюдо", "кухн"]
    MUSEUM_KEYWORDS = ["музей", "музеи", "экспозиция", "выставка", "галерея", "коллекция"]
    ATTRACTION_KEYWORDS = ["достопримечательность", "посмотреть", "посетить", "интересное", "место", "архитектура", "памятник", "кремль", "монастырь", "церковь", "собор"]
    ACCOMMODATION_KEYWORDS = ["отель", "гостиница", "хостел", "номер", "жилье", "размещение", "ночлег", "апартаменты"]
    TRANSPORT_KEYWORDS = ["транспорт", "добраться", "автобус", "поезд", "такси", "маршрут", "дорога", "проезд"]
    
    # Расширенные ключевые слова для детального анализа
    CUISINE_KEYWORDS = ["русск", "итальянск", "европейск", "азиатск", "китайск", "японск", "грузинск", "узбекск", "восточн", "европейск", "французск", "мексиканск"]
    BUDGET_KEYWORDS = ["эконом", "дешев", "недорог", "средн", "премиум", "дорог", "люкс", "бюджет", "цена", "стоимость"]
    LOCATION_KEYWORDS = ["центр", "кремл", "окраин", "район", "рядом", "недалек", "близко", "центр", "исторический центр"]
    TYPE_KEYWORDS = ["кафе", "ресторан", "столов", "паб", "бар", "бистро", "трактир", "закусочн", "кофейн", "пиццери", "суши"]
    ATMOSPHERE_KEYWORDS = ["уютн", "романтич", "семейн", "делов", "традицион", "современ", "аутентичн", "классическ", "вип"]
    
    # Фразы-триггеры для уточнений
    CLARIFICATION_TRIGGERS = [
        "где", "как", "что", "какой", "какая", "какие", "какое", 
        "посоветуй", "порекомендуй", "хочу", "ищу", "найти", "найди"
    ]
    
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
    
    # Минимальные требования для разных категорий
    MIN_REQUIREMENTS = {
        "food": ["cuisine", "budget"],  # Для еды обязательно знать кухню и бюджет
        "museum": ["type"],             # Для музеев - тип
        "attraction": ["category"],     # Для достопримечательностей - категорию
        "accommodation": ["budget", "type"],  # Для жилья - бюджет и тип
        "transport": ["direction"]      # Для транспорта - направление
    }

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
        return "Ошибка при выполнения поиска"

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

def is_detailed_question(question: str, question_type: str) -> Tuple[bool, Dict[str, bool]]:
    """Анализирует, содержит ли вопрос достаточную детализацию"""
    question_lower = question.lower()
    words = set(question_lower.split())
    
    # Анализ заполненности параметров
    specified_params = {
        'cuisine': any(keyword in question_lower for keyword in Config.CUISINE_KEYWORDS),
        'budget': any(keyword in question_lower for keyword in Config.BUDGET_KEYWORDS),
        'location': any(keyword in question_lower for keyword in Config.LOCATION_KEYWORDS),
        'type': any(keyword in question_lower for keyword in Config.TYPE_KEYWORDS),
        'atmosphere': any(keyword in question_lower for keyword in Config.ATMOSPHERE_KEYWORDS),
        'time': any(word in question_lower for word in ['утро', 'день', 'вечер', 'ночь', 'завтрак', 'обед', 'ужин']),
        'rating': any(word in question_lower for word in ['лучш', 'популярн', 'известн', 'рекоменду']),
        'specific_name': any(word in question_lower for word in ['назван', 'имя', 'как называется']),
        'price_range': re.search(r'\d+[\s-]+\d+', question_lower) is not None  # Диапазон цен типа "1000-2000"
    }
    
    # Проверяем длину и сложность вопроса
    word_count = len(question.split())
    has_adjectives = len([word for word in words if len(word) > 4]) >= 2  # Есть описательные слова
    has_specifics = any(specified_params.values())
    
    # Вопрос считается подробным если:
    # 1. Длиннее 5 слов И имеет специфичные параметры
    # 2. Имеет более 2 специфичных параметров
    # 3. Содержит конкретные названия или цифры
    is_detailed = (
        (word_count > 5 and has_specifics) or
        sum(specified_params.values()) >= 2 or
        specified_params['specific_name'] or
        specified_params['price_range']
    )
    
    return is_detailed, specified_params

def analyze_missing_requirements(question_type: str, specified_params: Dict[str, bool]) -> List[str]:
    """Анализирует недостающие требования для конкретного типа вопроса"""
    required_params = Config.MIN_REQUIREMENTS.get(question_type, [])
    missing = []
    
    for param in required_params:
        if not specified_params.get(param, False):
            missing.append(param)
    
    return missing

def generate_targeted_clarification(question_type: str, missing_params: List[str], specified_params: Dict[str, bool]) -> str:
    """Генерирует целенаправленное уточнение на основе недостающих параметров"""
    
    clarification_templates = {
        "food": {
            "cuisine": "🍝 **Какую кухню предпочитаете в ресторанах?**\n- Русская 🥘\n- Европейская 🍝\n- Азиатская 🍜\n- Итальянская 🍕\n- Другая",
            "budget": "💰 **Какой бюджет на рестораны?**\n- Экономный (до 1000 руб)\n- Средний (1000-2500 руб)\n- Премиум (от 2500 руб)",
            "location": "📍 **Где ищете рестораны?**\n- В центре/у Кремля\n- В конкретном районе\n- Не важно",
            "type": "🏢 **Какой тип заведения?**\n- Ресторан 🍽️\n- Кафе ☕\n- Столовая 🥄\n- Паб 🍺",
            "atmosphere": "🎭 **Какая атмосфера в ресторане?**\n- Уютная/семейная 👨‍👩‍👧‍👦\n- Романтическая 💑\n- Деловая 💼\n- Традиционная 🏮"
        },
        "museum": {
            "type": "🏛️ **Какие музеи вас интересуют?**\n- Исторические 📜 (Суздальский кремль)\n- Художественные 🎨 (иконопись)\n- Тематические 🔍 (медовуха, огурец)\n- Архитектурные 🏰 (монастыри)",
            "theme": "🎨 **Какая тематика музеев?**\n- История города\n- Искусство и иконопись\n- Народные промыслы\n- Архитектура",
            "location": "📍 **Расположение музеев?**\n- В центре города\n- В Кремле\n- В монастырях\n- Не важно"
        },
        "attraction": {
            "category": "🏰 **Какие достопримечательности хотите посмотреть?**\n- Архитектура 🏛️ (кремль, монастыри)\n- История 📚 (древние памятники)\n- Природа 🌳 (парки, река Каменка)\n- Религиозные ⛪ (церкви, соборы)",
            "type": "🔍 **Тип достопримечательностей?**\n- Кремль и укрепления\n- Монастыри и церкви\n- Музеи под открытым небом\n- Природные объекты",
            "location": "📍 **Где ищете достопримечательности?**\n- В центре города\n- Вокруг Кремля\n- За городом\n- Везде"
        },
        "accommodation": {
            "budget": "💰 **Бюджет на проживание?**\n- Эконом (до 2000 руб/ночь)\n- Средний (2000-5000 руб)\n- Премиум (от 5000 руб)",
            "type": "🏨 **Тип размещения?**\n- Отель ****\n- Гостиница\n- Хостел\n- Апартаменты\n- Гостевой дом",
            "location": "📍 **Расположение отеля?**\n- В центре города\n- Рядом с Кремлем\n- Тихий район\n- Не важно",
            "amenities": "🛏️ **Важные удобства?**\n- WiFi\n- Парковка\n- Завтрак\n- Кондиционер"
        },
        "transport": {
            "direction": "🚗 **Направление транспорта?**\n- До Суздаля (из Москвы/Владимира)\n- По городу\n- Аренда транспорта\n- Экскурсионные маршруты",
            "type": "🚌 **Тип транспорта?**\n- Автобус\n- Поезд\n- Такси\n- Аренда авто\n- Велосипед",
            "budget": "💰 **Бюджет на транспорт?**\n- Экономный\n- Средний\n- Комфортный"
        }
    }
    
    # Если недостающих параметров много, даем общее уточнение для категории
    if len(missing_params) >= 2:
        if question_type == "food":
            return (
                "🍽️ **Уточните параметры поиска ресторанов:**\n\n"
                "• **Кухня**: русская, европейская, азиатская?\n"
                "• **Бюджет**: экономный, средний, премиум?\n" 
                "• **Местоположение**: центр, район Кремля?\n"
                "• **Тип**: ресторан, кафе, паб?\n\n"
                "Что для вас важнее при выборе ресторана?"
            )
        elif question_type == "museum":
            return (
                "🏛️ **Уточните параметры поиска музеев:**\n\n"
                "• **Тип**: исторические, художественные, тематические?\n"
                "• **Тематика**: история, искусство, архитектура?\n"
                "• **Расположение**: центр, Кремль, монастыри?\n\n"
                "Какие музеи вас больше интересуют?"
            )
        elif question_type == "attraction":
            return (
                "🏰 **Уточните параметры достопримечательностей:**\n\n"
                "• **Категория**: архитектура, история, природа?\n"
                "• **Тип**: кремль, монастыри, церкви, музеи?\n"
                "• **Расположение**: центр, окраины, за городом?\n\n"
                "Что хотите посмотреть в первую очередь?"
            )
    
    # Целевые уточнения для недостающих параметров
    clarifications = []
    template = clarification_templates.get(question_type, {})
    
    for param in missing_params[:2]:  # Максимум 2 уточнения за раз
        if param in template:
            clarifications.append(template[param])
    
    if clarifications:
        return "\n\n".join(clarifications)
    
    # Универсальное уточнение
    return "Пожалуйста, уточните ваш запрос для более точных рекомендаций."

def improved_needs_clarification(question: str) -> Tuple[bool, Optional[str]]:
    """Улучшенная проверка необходимости уточнения"""
    question = question.strip()
    if len(question) < 2:
        return True, "Пожалуйста, задайте более конкретный вопрос."
    
    question_lower = question.lower()
    words = question_lower.split()
    
    # Определяем тип вопроса
    question_type = classify_question(question)
    
    # Проверяем, является ли вопрос триггерным (требует уточнения по умолчанию)
    is_trigger_question = (
        any(trigger in question_lower for trigger in Config.CLARIFICATION_TRIGGERS) and
        len(words) <= 4
    )
    
    # Анализируем детализацию вопроса
    is_detailed, specified_params = is_detailed_question(question, question_type)
    
    # Если вопрос уже подробный - не нужно уточнение
    if is_detailed:
        return False, None
    
    # Если это очень короткий вопрос (1-2 слова)
    if len(words) <= 2:
        single_word = words[0] if words else ""
        
        # Специфичные ответы для каждой категории
        category_responses = {
            # Еда и рестораны
            "еда": "🍽️ " + get_food_clarification(),
            "кафе": "☕ " + get_food_clarification(),
            "ресторан": "🍽️ " + get_food_clarification(),
            "рестораны": "🍽️ " + get_food_clarification(),
            "столовая": "🥄 " + get_food_clarification(),
            "паб": "🍺 " + get_food_clarification(),
            "бар": "🍸 " + get_food_clarification(),
            
            # Музеи
            "музей": "🏛️ " + get_museum_clarification(),
            "музеи": "🏛️ " + get_museum_clarification(),
            "галерея": "🎨 " + get_museum_clarification(),
            "выставка": "🖼️ " + get_museum_clarification(),
            
            # Достопримечательности
            "достопримечательности": "🏰 " + get_attraction_clarification(),
            "кремль": "🏰 " + get_attraction_clarification(),
            "монастырь": "⛪ " + get_attraction_clarification(),
            "церковь": "⛪ " + get_attraction_clarification(),
            
            # Жилье
            "отель": "🏨 " + get_accommodation_clarification(),
            "гостиница": "🏨 " + get_accommodation_clarification(),
            "хостел": "🛏️ " + get_accommodation_clarification(),
            "жилье": "🏠 " + get_accommodation_clarification(),
            
            # Транспорт
            "транспорт": "🚗 " + get_transport_clarification(),
            "автобус": "🚌 " + get_transport_clarification(),
            "такси": "🚕 " + get_transport_clarification()
        }
        
        if single_word in category_responses:
            return True, category_responses[single_word]
        
        # Для любых других однословных запросов
        return True, "Уточните, что именно вас интересует? Например:\n- 'Рестораны русской кухни'\n- 'Недорогие отели в центре'\n- 'Как добраться до Кремля'"
    
    # Для триггерных вопросов средней длины
    if is_trigger_question:
        missing_params = analyze_missing_requirements(question_type, specified_params)
        if missing_params:
            clarification = generate_targeted_clarification(question_type, missing_params, specified_params)
            return True, clarification
    
    # Для вопросов, где указана только кухня (для ресторанов)
    if (question_type == "food" and 
        specified_params['cuisine'] and 
        not any([specified_params['budget'], specified_params['location'], specified_params['type']])):
        
        return True, generate_targeted_clarification("food", ["budget", "location"], specified_params)
    
    # Для вопросов, где указан только тип (для музеев)
    if (question_type == "museum" and 
        specified_params['type'] and 
        not any([specified_params['location']])):
        
        return True, generate_targeted_clarification("museum", ["theme"], specified_params)
    
    # Если дошли досюда и вопрос не подробный, но и не явно требует уточнения
    if not is_detailed and len(words) <= 5:
        return True, "Уточните, пожалуйста, ваш запрос для более точного ответа."
    
    return False, None

def is_user_response_to_clarification(db: Session, user_id: str, current_question: str) -> bool:
    """Улучшенная проверка ответа на уточнение"""
    last_assistant_msg = get_last_assistant_message(db, user_id)
    if not last_assistant_msg or not is_clarification_request(last_assistant_msg):
        return False
    
    current_lower = current_question.lower()
    
    # Расширенные признаки ответа на уточнение
    clarification_indicators = [
        # Бюджетные указания
        any(word in current_lower for word in ['эконом', 'деш', 'недорог', 'средн', 'премиум', 'дорог', 'люкс']),
        # Локационные указания  
        any(word in current_lower for word in ['центр', 'кремл', 'окраин', 'район', 'не важно', 'любо']),
        # Кухонные предпочтения
        any(word in current_lower for word in ['русск', 'итальянск', 'европейск', 'азиатск', 'китайск', 'японск']),
        # Типы заведений
        any(word in current_lower for word in ['кафе', 'ресторан', 'столов', 'паб', 'бар']),
        # Короткие ответы (1-3 слова)
        len(current_question.split()) <= 3,
        # Ответы с цифрами (бюджет)
        bool(re.search(r'\d+', current_question)),
        # Выбор из предложенных вариантов
        any(word in current_lower for word in ['перв', 'втор', 'трет', 'последн', 'люб']),
        # Специфичные ответы для музеев
        any(word in current_lower for word in ['историч', 'художеств', 'тематич', 'архитектур']),
        # Специфичные ответы для достопримечательностей
        any(word in current_lower for word in ['архитектур', 'истори', 'природ', 'религиоз']),
    ]
    
    return any(clarification_indicators)

def get_food_clarification() -> str:
    """Уточняющий вопрос для еды"""
    return (
        "🍽️ **Расскажите о предпочтениях в ресторанах:**\n\n"
        "• **Кухня**: какую кухню предпочитаете?\n"
        "• **Бюджет**: какой диапазон цен?\n"
        "• **Расположение**: где ищете заведение?\n"
        "• **Тип**: ресторан, кафе, столовая или паб?\n\n"
        "Что для вас важнее при выборе места?"
    )

def get_museum_clarification() -> str:
    """Уточняющий вопрос для музеев"""
    return (
        "🏛️ **Уточните про музеи:**\n\n"
        "• **Тип музеев**: исторические, художественные, тематические?\n"
        "• **Тематика**: что именно интересует?\n"
        "• **Расположение**: в центре, в Кремле, в монастырях?\n\n"
        "Какие музеи хотите посетить?"
    )

def get_attraction_clarification() -> str:
    """Уточняющий вопрос для достопримечательностей"""
    return (
        "🏰 **Что интересует из достопримечательностей?**\n\n"
        "• **Категория**: архитектура, история, природа, религия?\n"
        "• **Конкретные места**: Кремль, монастыри, церкви?\n"
        "• **Расположение**: в центре, на окраинах?\n\n"
        "Что хотите увидеть в первую очередь?"
    )

def get_accommodation_clarification() -> str:
    """Уточняющий вопрос для размещения"""
    return (
        "🏨 **Уточните параметры проживания:**\n\n"
        "• **Бюджет**: эконом, средний, премиум?\n"
        "• **Тип**: отель, гостиница, хостел, апартаменты?\n"
        "• **Расположение**: центр, тихий район?\n"
        "• **Удобства**: что важно иметь?\n\n"
        "Что для вас наиболее важно в жилье?"
    )

def get_transport_clarification() -> str:
    """Уточняющий вопрос для транспорта"""
    return (
        "🚗 **Уточните про транспорт:**\n\n"
        "• **Направление**: до Суздаля, по городу, аренда?\n"
        "• **Тип транспорта**: автобус, поезд, такси, авто?\n"
        "• **Бюджет**: какой уровень комфорта?\n\n"
        "Что именно вас интересует?"
    )

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
            elif question_type == "transport":
                return format_transport_response(docs)
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
        return "К сожалению, не нашел подходящих ресторанов по вашему запросу."
    
    response = ["🍽️ **Рекомендую следующие рестораны и кафе:**\n"]
    for i, doc in enumerate(docs[:5], 1):
        name = doc.metadata.get("name", "Заведение")
        cuisine = doc.metadata.get("type", "кухня не указана")
        address = doc.metadata.get("address", "адрес не указан")
        price = doc.metadata.get("price", "")
        
        response.append(f"{i}. **{name}**")
        response.append(f"   🍳 Кухня: {cuisine}")
        if price:
            response.append(f"   💰 Цены: {price}")
        if address:
            response.append(f"   📍 Адрес: {address}")
        response.append("")
    
    response.append("Рекомендую уточнить часы работы по телефону перед посещением!")
    return "\n".join(response)

def format_museum_response(docs: List[Document]) -> str:
    """Форматирование ответа о музеях"""
    if not docs:
        return "К сожалению, не нашел информации о музеях по вашему запросу."
    
    response = ["🏛️ **Музеи Суздаля:**\n"]
    for i, doc in enumerate(docs[:5], 1):
        name = doc.metadata.get("name", "Музей")
        description = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
        address = doc.metadata.get("address", "")
        hours = doc.metadata.get("hours", "")
        
        response.append(f"{i}. **{name}**")
        response.append(f"   📖 {description}")
        if address:
            response.append(f"   📍 {address}")
        if hours:
            response.append(f"   🕒 {hours}")
        response.append("")
    
    return "\n".join(response)

def format_attraction_response(docs: List[Document]) -> str:
    """Форматирование ответа о достопримечательностях"""
    if not docs:
        return "К сожалению, не нашел достопримечательностей по вашему запросу."
    
    response = ["🏰 **Достопримечательности Суздаля:**\n"]
    for i, doc in enumerate(docs[:5], 1):
        name = doc.metadata.get("name", "Достопримечательность")
        description = doc.page_content[:120] + "..." if len(doc.page_content) > 120 else doc.page_content
        address = doc.metadata.get("address", "")
        
        response.append(f"{i}. **{name}**")
        response.append(f"   📖 {description}")
        if address:
            response.append(f"   📍 {address}")
        response.append("")
    
    return "\n".join(response)

def format_accommodation_response(docs: List[Document]) -> str:
    """Форматирование ответа о жилье"""
    if not docs:
        return "К сожалению, не нашел вариантов размещения по вашему запросу."
    
    response = ["🏨 **Варианты размещения в Суздале:**\n"]
    for i, doc in enumerate(docs[:5], 1):
        name = doc.metadata.get("name", "Отель")
        type_info = doc.metadata.get("type", "")
        address = doc.metadata.get("address", "")
        price = doc.metadata.get("price", "")
        
        response.append(f"{i}. **{name}**")
        if type_info:
            response.append(f"   🏢 Тип: {type_info}")
        if price:
            response.append(f"   💰 Цены: {price}")
        if address:
            response.append(f"   📍 Адрес: {address}")
        response.append("")
    
    return "\n".join(response)

def format_transport_response(docs: List[Document]) -> str:
    """Форматирование ответа о транспорте"""
    if not docs:
        return "К сожалению, не нашел информации о транспорте по вашему запросу."
    
    response = ["🚗 **Транспорт в Суздале:**\n"]
    for i, doc in enumerate(docs[:5], 1):
        name = doc.metadata.get("name", "Транспорт")
        description = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
        
        response.append(f"{i}. **{name}**")
        response.append(f"   {description}")
        response.append("")
    
    return "\n".join(response)

def format_general_response(docs: List[Document], query: str) -> str:
    """Форматирование общего ответа"""
    if not docs:
        return f"К сожалению, не нашел информации по запросу '{query}'."
    
    response = [f"**Результаты по запросу '{query}':**\n"]
    for i, doc in enumerate(docs[:5], 1):
        name = doc.metadata.get("name", "Место")
        description = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
        address = doc.metadata.get("address", "")
        
        response.append(f"{i}. **{name}**")
        response.append(f"   {description}")
        if address:
            response.append(f"   📍 {address}")
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
    """Улучшенная обработка вопроса с интеллектуальными уточнениями"""
    try:
        question = question.strip()
        if not question:
            return "Пожалуйста, задайте ваш вопрос о Суздале."
        
        # Проверяем, является ли это ответом на уточнение
        if is_user_response_to_clarification(db, user_id, question):
            response = generate_clarified_response(db, user_id, question)
            update_dialog_context(db, user_id, "assistant", response)
            return response
        
        # Интеллектуальная проверка необходимости уточнения
        needs_clarify, clarification_text = improved_needs_clarification(question)
        if needs_clarify:
            update_dialog_context(db, user_id, "user", question)
            update_dialog_context(db, user_id, "assistant", clarification_text)
            return clarification_text
        
        # Если вопрос достаточно детализирован - обрабатываем сразу
        dialog_context = get_dialog_context(db, user_id)
        context_docs = document_retriever.invoke(question)
        
        # Формирование ответа для детализированных вопросов
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
