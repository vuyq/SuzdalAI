import os
import ssl
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_gigachat import GigaChat, GigaChatEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper


# Загрузка конфигурации
load_dotenv()

class Config:
    GIGACHAT_AUTH = os.getenv("GIGACHAT_AUTH")
    CERT_URL = os.getenv("CERT_URL")
    CERT_PATH = os.getenv("CERT_PATH")

# Скачивание сертификата
if not Path(Config.CERT_PATH).exists():
    try:
        response = requests.get(Config.CERT_URL)
        response.raise_for_status()
        with open(Config.CERT_PATH, "wb") as f:
            f.write(response.content)
    except Exception as e:
        raise Exception(f"Ошибка загрузки сертификата: {str(e)}")

# Инициализация GigaChat
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def init_gigachat():
    try:
        access_token = get_gigachat_token()
        return GigaChat(
            access_token=access_token,
            model="GigaChat-2",
            temperature=0,
            verify_ssl_certs=True,
            ca_bundle_file=Config.CERT_PATH
        )
    except Exception as e:
        raise Exception(f"Ошибка инициализации GigaChat: {str(e)}")

# Инициализация компонентов
try:
    ai_assistant = init_gigachat()
    search = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper(max_results=3))
except Exception as e:
    print(f"Ошибка инициализации: {str(e)}")
    raise

# Модели данных
class Question(BaseModel):
    question: str

class Feedback(BaseModel):
    is_helpful: bool
    comment: Optional[str] = None

# Шаблоны ответов
PROMPT_TEMPLATE = """
Ты - дружелюбный гид по Суздалю. Отвечай кратко и информативно.

Контекст:
{context}

Вопрос: {question}

Формат ответа:
1. Основная информация
2. Почему стоит посетить
3. Практические детали (адрес, время работы)
4. Лайфхаки (если есть)
"""

prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

# Функции обработки
def analyze_feedback(feedback: Feedback) -> str:
    """Анализирует обратную связь и возвращает сообщение"""
    if feedback.is_helpful:
        return "Спасибо! Рад, что помог."
    
    if feedback.comment:
        return f"Спасибо за комментарий! Учту это в будущем: '{feedback.comment}'"
    return "Спасибо за отзыв! Постараюсь улучшить свои ответы."

def format_response(answer: str, request_analysis: bool = True) -> dict:
    """Форматирует ответ с возможностью обратной связи"""
    response = {"answer": answer}
    if request_analysis:
        response["feedback_request"] = {
            "question": "Понравился ли вам ответ?",
            "options": ["Да", "Нет"]
        }
    return response

def get_context_answer(question: str) -> str:
    """Получает ответ из контекста"""
    # Здесь должна быть реализация поиска в вашей базе знаний
    return "Пример ответа из базы знаний"

def get_web_answer(question: str) -> str:
    """Ищет ответ в интернете"""
    try:
        return search.run(f"{question} Суздаль site:ru") or "Не удалось найти информацию"
    except Exception:
        return "Ошибка поиска"

# FastAPI приложение
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask")
async def ask_question(question: Question):
    try:
        # Сначала пробуем найти ответ в локальной базе
        answer = get_context_answer(question.question)
        
        # Если не нашли - ищем в интернете
        if "не удалось" in answer.lower():
            answer = get_web_answer(question.question)
        
        return format_response(answer)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def handle_feedback(feedback: Feedback):
    try:
        analysis_result = analyze_feedback(feedback)
        return {"message": analysis_result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
