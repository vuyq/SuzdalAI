import os
import ssl
import json
import logging
import requests
import uvicorn
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_gigachat import GigaChat, GigaChatEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from tenacity import retry, stop_after_attempt, wait_exponential
from ddgs import DDGS

# Constants
CONFIG = {
    "CERT_PATH": os.getenv("CERT_PATH"),
    "CERT_URL": os.getenv("CERT_URL"),
    "GIGACHAT_AUTH": os.getenv("GIGACHAT_AUTH"),
    "CSV_URL": "https://raw.githubusercontent.com/vuyq/SuzdalAI/refs/heads/main/suzdal_full_guide_refine/attractions.csv",
    "FEEDBACK_DB": "feedback.json",
    "PORT": int(os.getenv("PORT", 8001))
}

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Models
class Question(BaseModel):
    question: str

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    feedback: str

class Feedback(BaseModel):
    question: str
    answer: str
    is_helpful: bool
    timestamp: str

# SSL Context Manager
class SSLContextManager:
    def __init__(self, cert_path):
        self.cert_path = cert_path
        self.context = None

    def __enter__(self):
        self.context = ssl.create_default_context(cafile=self.cert_path)
        self.context.verify_mode = ssl.CERT_REQUIRED
        return self.context

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

# Core Functions
def setup_certificate():
    if not Path(CONFIG["CERT_PATH"]).exists():
        try:
            response = requests.get(CONFIG["CERT_URL"], timeout=10)
            response.raise_for_status()
            with open(CONFIG["CERT_PATH"], "wb") as f:
                f.write(response.content)
            logger.info("Certificate downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to download certificate: {e}")
            raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_gigachat_token():
    with SSLContextManager(CONFIG["CERT_PATH"]) as ssl_context:
        response = requests.post(
            "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
            headers={
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept': 'application/json',
                'RqUID': 'a2231e67-570e-47ca-bae8-82ca565850eb',
                'Authorization': f'Basic {CONFIG["GIGACHAT_AUTH"]}'
            },
            data={'scope': 'GIGACHAT_API_PERS'},
            verify=CONFIG["CERT_PATH"],
            timeout=10
        )
        response.raise_for_status()
        return response.json().get("access_token")

def initialize_models():
    try:
        access_token = get_gigachat_token()
        with SSLContextManager(CONFIG["CERT_PATH"]) as ssl_context:
            return (
                GigaChatEmbeddings(
                    access_token=access_token,
                    model="Embeddings",
                    scope="GIGACHAT_API_PERS",
                    verify_ssl_certs=False,
                    ca_bundle_file=CONFIG["CERT_PATH"],
                    ssl_context=ssl_context
                ),
                GigaChat(
                    access_token=access_token,
                    model="GigaChat-2",
                    temperature=0.2,
                    verify_ssl_certs=False,
                    ca_bundle_file=CONFIG["CERT_PATH"],
                    ssl_context=ssl_context
                )
            )
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        raise

def load_data():
    try:
        df = pd.read_csv(CONFIG["CSV_URL"], sep=';', on_bad_lines='skip')
        return [
            Document(
                page_content="\n".join(
                    f"{col}: {val if pd.notna(val) else 'не указано'}" 
                    for col, val in row.items()
                ),
                metadata={
                    "title": row.get("Name", ""),
                    "type": row.get("Type", ""),
                    "tags": row.get("Tags", ""),
                    "address": row.get("Address", "не указано")
                }
            )
            for _, row in df.iterrows()
        ]
    except Exception as e:
        logger.error(f"Data loading error: {e}")
        raise

# Response Handlers
class ResponseHandler:
    @staticmethod
    def format_from_context(context):
        if not context:
            return None
            
        main_doc = context[0]
        response = [
            "Вот что я нашел:",
            f"🏛 {main_doc.metadata.get('title', 'Название не указано')}",
            f"📍 Тип: {main_doc.metadata.get('type', 'не указан')}",
            f"📌 Адрес: {main_doc.metadata.get('address', 'не указан')}",
            f"\n{main_doc.page_content[:200]}{'...' if len(main_doc.page_content) > 200 else ''}"
        ]
        
        if len(context) > 1:
            response.append("\nТакже возможно вас заинтересует:")
            response.extend(
                f"- {doc.metadata.get('title', '')} ({doc.metadata.get('type', '')})"
                for doc in context[1:3]
            )
        
        return "\n".join(response)

    @staticmethod
    def format_from_web(results):
        return "В моей базе нет точной информации, но вот что удалось найти:\n\n" + \
               "\n".join(f"{r['title']}: {r['body']} (Источник: {r['href']})" for r in results[:3])

    @staticmethod
    def add_standard_questions(response):
        return f"{response}\n\nХотите узнать что-то еще?\nБыл ли полезен этот ответ? (Да/Нет)"

# Initialize application components
setup_certificate()
embedding_model, ai_assistant = initialize_models()
search = DDGS()
text_documents = load_data()
vector_store = FAISS.from_documents(text_documents, embedding_model)
document_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# FastAPI App
app = FastAPI(title="Suздаль Tourism Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"{request.method} {request.url}")
    return await call_next(request)

@app.get("/")
async def root():
    return {
        "message": "Добро пожаловать в API туристического помощника по Суздалю!",
        "endpoints": {
            "ask": "POST /ask для вопросов о городе",
            "feedback": "GET /feedback для просмотра отзывов",
            "health": "GET /health для проверки работы сервиса"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "OK"}

@app.post("/ask")
async def ask_question(item: Question):
    try:
        # First try local database
        context = document_retriever.invoke(item.question)
        if context and "не указано" not in context[0].page_content:
            response = ResponseHandler.format_from_context(context)
        else:
            # Fallback to web search
            results = search.text(f"{item.question} Суздаль", region='ru-ru', max_results=3)
            response = ResponseHandler.format_from_web(results) if results else "Информация не найдена"
        
        return {"answer": ResponseHandler.add_standard_questions(response)}
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(500, "Internal Server Error")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=CONFIG["PORT"])
