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
    """Вычисляет косинусную схожесть между двумя векторами"""
    if not vec1 or not vec2:
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0

def semantic_search_messages(query: str, messages: List[Message], k: int = 3) -> List[Message]:
    """Семантический поиск по истории сообщений"""
    if not messages or not embedding_model:
        return messages[-k:] if k < len(messages) else messages
    
    try:
        query_embedding = embedding_model.embed_query(query)
        
        scored_messages = []
        for msg in messages:
            if msg.embeddings:
                similarity = cosine_similarity(query_embedding, msg.embeddings)
                scored_messages.append((msg, similarity))
            else:
                scored_messages.append((msg, 0.1))
        
        scored_messages.sort(key=lambda x: x[1], reverse=True)
        return [msg for msg, score in scored_messages[:k]]
    
    except Exception as e:
        logger.error(f"Ошибка семантического поиска: {e}")
        return messages[-k:] if k < len(messages) else messages

def extract_user_preferences(messages: List[Message]) -> Dict[str, Any]:
    """Извлекает предпочтения пользователя из истории диалога"""
    preferences = {
        "communication_style": "standard",
        "topics_of_interest": [],
        "dislikes": [],
        "specific_requests": []
    }
    
    style_keywords = {
        "formal": ["формально", "официально", "без смайликов", "без эмоций", "серьезно"],
        "friendly": ["дружелюбно", "неформально", "с юмором", "смайлики", "весело"],
        "detailed": ["подробно", "детально", "развернуто", "подробнее"],
        "brief": ["кратко", "сжато", "по делу", "покороче"]
    }
    
    for msg in messages:
        if msg.role == "user":
            content = msg.content.lower()
            
            for style, keywords in style_keywords.items():
                if any(keyword in content for keyword in keywords):
                    preferences["communication_style"] = style
            
            interest_patterns = [
                r"(интересуюсь|нравятся|люблю|хочу узнать про|интересует) ([^.,!?]+)",
                r"(мне интересны|мне нравятся|хотел бы) ([^.,!?]+)"
            ]
            for pattern in interest_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    topics = [topic.strip() for topic in re.split(r'[,и]', match[1]) if len(topic.strip()) > 2]
                    preferences["topics_of_interest"].extend(topics)
            
            dislike_patterns = [
                r"(не нравятся|не люблю|не интересно|не хочу|не надо) ([^.,!?]+)",
                r"(избегайте|не упоминайте|не говорите про|пропустите) ([^.,!?]+)"
            ]
            for pattern in dislike_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    topics = [topic.strip() for topic in re.split(r'[,и]', match[1]) if len(topic.strip()) > 2]
                    preferences["dislikes"].extend(topics)
    
    preferences["topics_of_interest"] = list(set([t for t in preferences["topics_of_interest"] if t]))
    preferences["dislikes"] = list(set([t for t in preferences["dislikes"] if t]))
    
    return preferences

def generate_conversation_summary(messages: List[Message]) -> str:
    """Генерирует семантическое резюме диалога"""
    if not messages:
        return "Нет истории диалога"
    
    recent_messages = messages[-min(Config.MEMORY_CONTEXT_SIZE, len(messages)):]
    
    summary_parts = []
    for msg in recent_messages:
        role = "User" if msg.role == "user" else "Assistant"
        summary_parts.append(f"{role}: {msg.content}")
    
    return "\n".join(summary_parts)

def apply_preferences_to_prompt(response: str, preferences: Dict[str, Any]) -> str:
    """Применяет предпочтения пользователя к стилю ответа"""
    if preferences["communication_style"] == "formal":
        response = re.sub(r"[😀-🙏️⚡️❤️🔥]", "", response)
        response = re.sub(r"\!+", ".", response)
    elif preferences["communication_style"] == "friendly":
        if "!" in response and "😊" not in response:
            response = response.replace("!", "! 😊")
        if "?" in response and "🤔" not in response:
            response = response.replace("?", "? 🤔")
    
    for dislike in preferences["dislikes"]:
        if dislike.lower() in response.lower():
            response = re.sub(fr"\b{re.escape(dislike)}\b", "[скрыто]", response, flags=re.IGNORECASE)
    
    return response

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

Учти историю диалога и предпочтения пользователя. Отвечай соответственно его стилю общения.
Отвечай на русском языке.
"""
tourism_prompt = PromptTemplate.from_template(TOURISM_PROMPT_TEMPLATE)

def generate_ai_response(question: str, context_docs: List[Document], web_results: str, 
                        conversation_summary: str, user_preferences: Dict) -> str:
    try:
        if not token_manager.is_token_valid():
            refresh_models()
        
        prompt_input = {
            "question": question,
            "context": "\n\n".join(d.page_content for d in context_docs) if context_docs else "Нет данных в базе",
            "web_search": web_results,
            "conversation_summary": conversation_summary,
            "user_preferences": json.dumps(user_preferences, ensure_ascii=False, indent=2)
        }
        
        response = ai_assistant.invoke(tourism_prompt.format(**prompt_input))
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        return apply_preferences_to_prompt(response_text, user_preferences)
        
    except Exception as e:
        logger.error(f"Ошибка генерации ответа: {e}")
        return "Извините, произошла ошибка при генерации ответа. Попробуйте позже."

def update_dialog_context(db: Session, user_id: str, role: str, message: str):
    try:
        session = db.query(ChatSession).filter(ChatSession.id == user_id).first()
        if not session:
            session = ChatSession(id=user_id, user_preferences={})
            db.add(session)
        
        message_embedding = None
        if embedding_model and len(message) > 10:
            try:
                message_embedding = embedding_model.embed_query(message)
            except Exception as e:
                logger.error(f"Ошибка создания эмбеддинга: {e}")
        
        db_message = Message(
            session_id=user_id, 
            role=role, 
            content=message, 
            timestamp=datetime.utcnow(),
            embeddings=message_embedding
        )
        db.add(db_message)
        
        messages = db.query(Message).filter(Message.session_id == user_id).all()
        if len(messages) % Config.PREFERENCES_UPDATE_INTERVAL == 0:
            session.user_preferences = extract_user_preferences(messages)
            session.conversation_summary = generate_conversation_summary(messages)
        
        db.commit()
    except Exception as e:
        logger.error(f"Ошибка обновления контекста диалога: {e}")
        db.rollback()
        raise

def get_relevant_memory(db: Session, user_id: str, current_question: str) -> str:
    """Получает релевантные сообщения из истории с помощью семантического поиска"""
    try:
        messages = db.query(Message).filter(Message.session_id == user_id).all()
        if not messages:
            return "Нет истории диалога"
        
        relevant_messages = semantic_search_messages(current_question, messages, Config.SEMANTIC_SEARCH_K)
        
        memory_context = []
        for msg in relevant_messages:
            role = "User" if msg.role == "user" else "Assistant"
            memory_context.append(f"{role}: {msg.content}")
        
        return "\n".join(memory_context) if memory_context else "Нет релевантной истории"
    except Exception as e:
        logger.error(f"Ошибка получения памяти: {e}")
        return "Ошибка доступа к памяти"

def get_dialog_context(db: Session, user_id: str, max_messages: int = 20) -> str:
    try:
        messages = db.query(Message).filter(Message.session_id == user_id)\
                     .order_by(Message.timestamp.asc()).limit(max_messages).all()
        
        dialog_lines = []
        for msg in messages:
            role = "User" if msg.role == "user" else "Assistant"
            dialog_lines.append(f"{role}: {msg.content}")
        
        return "\n".join(dialog_lines)
    except Exception as e:
        logger.error(f"Ошибка получения контекста диалога: {e}")
        return ""

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

def clear_chat_history(db: Session, user_id: str):
    try:
        db.query(Message).filter(Message.session_id == user_id).delete()
        session = db.query(ChatSession).filter(ChatSession.id == user_id).first()
        if session:
            session.last_question = None
            session.user_preferences = {}
            session.conversation_summary = None
            session.clarification_context = None
            session.updated_at = datetime.utcnow()
        db.commit()
        logger.info(f"История чата очищена для пользователя {user_id}")
    except Exception as e:
        logger.error(f"Ошибка очистки истории: {e}")
        db.rollback()
        raise

def is_message_unclear(message: str) -> bool:
    msg = message.strip()
    if len(msg) < 3:
        return True
    if all(c in ".,!? " for c in msg):
        return True
    return False

def needs_clarification(question: str) -> Tuple[bool, str, Optional[Dict]]:
    q = question.lower()
    
    if "рестора" in q or "поесть" in q or "кухн" in q or "кафе" in q or "еда" in q:
        return True, (
            "Я вижу, что Вы ищите ресторан или кафе. Можете уточнить:\n"
            "- Какой тип кухни предпочитаете (русская, японская, китайская, европейская, любая)?\n"
            "- Важно ли расположение (центр, окраина, рядом с достопримечательностями)?\n"
            "- Нужен ли бюджетный вариант или премиум?"
        ), {"type": "restaurant", "original_question": question}
    
    if "музе" in q:
        return True, (
            "В Суздале много интересных музеев. Какой вас интересует больше?\n\n"
            "- Музей деревянного зодчества\n"
            "- Спасо-Евфимиев монастырь (музейный комплекс)\n"
            "- Кремль с его экспозициями\n"
            "- Музей восковых фигур\n"
            "- Или что-то другое?"
        ), {"type": "museum", "original_question": question}
    
    if any(word in q for word in ["достопримечательности", "куда сходить", "что посетить", "что посмотреть"]):
        return True, (
            "Вы ищете достопримечательности. Хотите больше про:\n"
            "- Исторические объекты\n"
            "- Музеи\n"
            "- Храмы и монастыри\n"
            "- Природные места"
        ), {"type": "attractions", "original_question": question}
    
    return False, "", None

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
            entry.append(f"Тип: {meta['type']}")
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

def handle_clarification_response(db: Session, user_id: str, response: str, context: Dict) -> str:
    try:
        session = db.query(ChatSession).filter(ChatSession.id == user_id).first()
        session.clarification_context = None
        db.commit()
        
        original_question = context.get("original_question", "")
        clarification_type = context.get("type", "")
        
        if clarification_type == "restaurant":
            search_query = f"{original_question} {response}"
        elif clarification_type == "museum":
            if "деревян" in response.lower():
                search_query = "Музей деревянного зодчества"
            elif "евфими" in response.lower() or "монастырь" in response.lower():
                search_query = "Спасо-Евфимиев монастырь"
            elif "кремль" in response.lower():
                search_query = "Суздальский кремль"
            elif "восков" in response.lower():
                search_query = "Музей восковых фигур"
            else:
                search_query = f"музей {response}"
        else:
            search_query = f"{original_question} {response}"
        
        context_docs = search_in_vector_store(search_query)
        
        if context_docs:
            formatted_context = format_context_docs(context_docs)
            response_text = f"📚 Вот что я нашёл по вашему запросу:\n\n{formatted_context}\n\n"
            response_text += "Хотите, чтобы я сделал расширенный рассказ об этих местах? Напишите 'да' или 'расскажи подробнее'."
        else:
            web_results = perform_web_search(search_query)
            response_text = f"🌐 Вот что удалось найти в интернете:\n\n{web_results}"
        
        return response_text
        
    except Exception as e:
        logger.error(f"Ошибка обработки уточнения: {e}")
        return "Произошла ошибка при обработке вашего ответа."

def handle_question(db: Session, question: str, user_id: str) -> str:
    if not app_initialized:
        return "Приложение еще не инициализировано. Попробуйте позже."
    
    question = question.strip()
    if not question:
        return "Пожалуйста, задайте ваш вопрос о Суздале."

    session = db.query(ChatSession).filter(ChatSession.id == user_id).first()
    if session and session.clarification_context:
        return handle_clarification_response(db, user_id, question, session.clarification_context)

    if is_message_unclear(question):
        response = "Не очень понял ваше сообщение, пожалуйста, напишите ещё раз."
        update_dialog_context(db, user_id, "assistant", response)
        return response

    update_dialog_context(db, user_id, "user", question)

    user_preferences = session.user_preferences if session and session.user_preferences else {}
    conversation_summary = session.conversation_summary if session else ""
    relevant_memory = get_relevant_memory(db, user_id, question)
    full_context = f"{conversation_summary}\n{relevant_memory}" if conversation_summary else relevant_memory

    dialog_context = get_dialog_context(db, user_id)

    if question.lower() in ["да", "расскажи", "расскажи подробнее", "подробнее"]:
        last_q = get_last_question(db, user_id)
        if last_q:
            context_docs = search_in_vector_store(last_q)
            if context_docs:
                ai_answer = generate_ai_response(last_q, context_docs, "", full_context, user_preferences)
                response = f"🤖 Расширенный рассказ:\n\n{ai_answer}"
            else:
                response = "К сожалению, не могу найти информацию для подробного рассказа."
        else:
            response = "Не могу найти предыдущий запрос для подробного рассказа."
        update_dialog_context(db, user_id, "assistant", response)
        return response

    needs_clarify, clarification_text, clarification_context = needs_clarification(question)
    if needs_clarify:
        if session:
            session.clarification_context = clarification_context
            db.commit()
        update_dialog_context(db, user_id, "assistant", clarification_text)
        return clarification_text

    context_docs = search_in_vector_store(question)
    
    if not context_docs and documents:
        context_docs = fuzzy_retrieval(question, documents, limit=Config.RETRIEVER_K)

    if context_docs:
        formatted_context = format_context_docs(context_docs)
        ai_answer = generate_ai_response(question, context_docs, "", full_context, user_preferences)
        set_last_question(db, user_id, question)
        response = (
            f"📚 Вот что я нашёл в базе:\n\n{formatted_context}\n\n"
            f"🤖 {ai_answer}\n\n"
            "Хотите более подробную информацию или поиск в интернете?"
        )
    else:
        web_results = perform_web_search(question)
        response = f"🌐 В базе ничего не найдено. Вот что удалось найти в интернете:\n\n{web_results}"

    update_dialog_context(db, user_id, "assistant", response)
    return response

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

@app.get("/history/{user_id}")
async def get_history(user_id: str, db: Session = Depends(get_db)):
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
    try:
        clear_chat_history(db, user_id)
        return {"status": "history cleared", "user_id": user_id}
    except Exception as e:
        logger.error(f"Ошибка очистки истории: {e}")
        raise HTTPException(status_code=500, detail="Ошибка очистки истории")

@app.post("/reinitialize")
async def reinitialize():
    try:
        initialize_app()
        return {"status": "reinitialized", "success": app_initialized}
    except Exception as e:
        logger.error(f"Ошибка переинициализации: {e}")
        raise HTTPException(status_code=500, detail="Ошибка переинициализации")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
