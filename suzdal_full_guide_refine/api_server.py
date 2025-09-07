import os
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Depends
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
from typing import List, Tuple
import uuid
import uvicorn

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

# Индексы
Index('ix_messages_session_id', Message.session_id)
Index('ix_messages_timestamp', Message.timestamp)
Index('ix_messages_session_role', Message.session_id, Message.role)

# Создание таблиц
Base.metadata.create_all(bind=engine)

# Dependency для БД
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
    access_token = get_gigachat_token()
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

def load_data() -> List[Document]:
    try:
        df = pd.read_csv(Config.CSV_DATA_URL, sep=';', on_bad_lines='skip')
    except Exception:
        df = pd.read_csv(Config.CSV_DATA_URL, on_bad_lines='skip')

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
"""

tourism_prompt = PromptTemplate.from_template(TOURISM_PROMPT_TEMPLATE)

def generate_ai_response(question: str, context_docs: List[Document], web_results: str, dialog_context: str) -> str:
    prompt_input = {
        "question": question,
        "context": "\n\n".join(d.page_content for d in context_docs) if context_docs else "Нет данных в базе",
        "web_search": web_results,
        "dialog_context": dialog_context
    }
    response = ai_assistant.invoke(tourism_prompt.format(**prompt_input))
    return response.content if hasattr(response, 'content') else str(response)

def update_dialog_context(db: Session, user_id: str, role: str, message: str):
    session = db.query(ChatSession).filter(ChatSession.id == user_id).first()
    if not session:
        session = ChatSession(id=user_id)
        db.add(session)
        db.commit()
    db_message = Message(session_id=user_id, role=role, content=message, timestamp=datetime.utcnow())
    db.add(db_message)
    db.commit()

def get_dialog_context(db: Session, user_id: str, max_messages: int = 10) -> str:
    messages = db.query(Message).filter(Message.session_id == user_id).order_by(Message.timestamp.asc()).limit(max_messages).all()
    return "\n".join(f"{msg.role}: {msg.content}" for msg in messages)

# 🔹 Проверка на необходимость уточнения
def needs_clarification(question: str) -> Tuple[bool, str]:
    q = question.lower()
    if any(word in q for word in ["где поесть", "что посетить", "куда сходить", "достопримечательности", "музеи", "рестораны"]):
        return True, (
            "Можете уточнить, что для вас важнее?\n"
            "- бюджет\n"
            "- тип кухни или место\n"
            "- расположение\n"
            "- время работы\n\n"
            "Это поможет мне подобрать лучший вариант."
        )
    return False, ""

# 🔹 Форматирование найденных данных
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

# 🔹 Основная логика
def handle_question(db: Session, question: str, user_id: str) -> str:
    question = question.strip()
    if not question:
        return "Пожалуйста, задайте ваш вопрос о Суздале."

    update_dialog_context(db, user_id, "user", question)
    dialog_context = get_dialog_context(db, user_id)

    # 1. Проверка на необходимость уточнения
    needs_clarify, clarification_text = needs_clarification(question)
    if needs_clarify:
        update_dialog_context(db, user_id, "assistant", clarification_text)
        return clarification_text

    # 2. Поиск в базе
    context_docs = document_retriever.invoke(question)

    if context_docs:
        formatted_context = format_context_docs(context_docs)
        ai_answer = generate_ai_response(question, context_docs, "", dialog_context)
        response = f"📚 Вот что я нашёл в базе:\n\n{formatted_context}\n\n🤖 {ai_answer}"
        if len(context_docs) < 3:
            response += "\n\n🤔 В базе мало информации. Хотите, я попробую поискать ещё и в интернете?"
    else:
        if any(word in question.lower() for word in ["да", "ищи", "интернет"]):
            web_results = perform_web_search(question)
            response = f"🌐 Вот что удалось найти в интернете:\n\n{web_results}"
        else:
            response = "К сожалению, в базе нет информации. Хотите, я попробую поискать в интернете?"

    update_dialog_context(db, user_id, "assistant", response)
    return response

# Инициализация
try:
    download_certificate()
    embedding_model, ai_assistant = initialize_models()
    documents = load_data()
    vector_store = FAISS.from_documents(documents, embedding_model)
    document_retriever = vector_store.as_retriever(search_kwargs={"k": Config.RETRIEVER_K})
except Exception as e:
    logger.critical(f"Ошибка инициализации: {e}")
    documents = []

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
    if not item.question.strip():
        return {"answer": "Пожалуйста, задайте ваш вопрос."}
    response = handle_question(db, item.question, item.user_id)
    return {"answer": response}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

