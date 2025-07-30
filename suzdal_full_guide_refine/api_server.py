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

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

# Конфигурация
GIGACHAT_AUTH = os.getenv("GIGACHAT_AUTH") 
CERT_URL = os.getenv("CERT_URL")
CERT_PATH = os.getenv("CERT_PATH")
CSV_URL = "https://raw.githubusercontent.com/vuyq/SuzdalAI/refs/heads/main/suzdal_full_guide_refine/attractions.csv"
FEEDBACK_DB = "feedback.json"

# Модели данных
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

# Проверка и загрузка сертификата
def setup_certificate():
    if not Path(CERT_PATH).exists():
        try:
            response = requests.get(CERT_URL)
            response.raise_for_status()
            with open(CERT_PATH, "wb") as f:
                f.write(response.content)
            logger.info("Сертификат успешно скачан")
        except Exception as e:
            logger.error(f"Не удалось скачать сертификат: {str(e)}")
            raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_gigachat_token():
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': 'a2231e67-570e-47ca-bae8-82ca565850eb',
        'Authorization': f'Basic {GIGACHAT_AUTH}'
    }
    payload = {'scope': 'GIGACHAT_API_PERS'}
    
    # Создаем SSL контекст
    ssl_context = ssl.create_default_context(cafile=CERT_PATH)
    ssl_context.verify_mode = ssl.CERT_REQUIRED
    
    try:
        response = requests.post(
            url, 
            headers=headers, 
            data=payload, 
            verify=CERT_PATH,
            timeout=10
        )
        response.raise_for_status()
        return response.json().get("access_token")
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка получения токена: {str(e)}")
        raise

def initialize_models():
    try:
        access_token = get_gigachat_token()
        logger.info("Токен успешно получен")
        
        # Создаем SSL контекст с нашим сертификатом
        ssl_context = ssl.create_default_context(cafile=CERT_PATH)
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        
        embedding_model = GigaChatEmbeddings(
            access_token=access_token,
            model="Embeddings",
            scope="GIGACHAT_API_PERS",
            verify_ssl_certs=False,  # Отключаем встроенную проверку
            ca_bundle_file=CERT_PATH,
            ssl_context=ssl_context  # Добавляем наш контекст
        )
        
        ai_assistant = GigaChat(
            access_token=access_token,
            model="GigaChat-2",
            temperature=0.2,
            verify_ssl_certs=False,  # Отключаем встроенную проверку
            ca_bundle_file=CERT_PATH,
            ssl_context=ssl_context  # Добавляем наш контекст
        )
        
        return embedding_model, ai_assistant
    except Exception as e:
        logger.error(f"Ошибка инициализации: {str(e)}")
        raiserror(f"Ошибка инициализации: {str(e)}")
        raise

# Инициализация компонентов
setup_certificate()
embedding_model, ai_assistant = initialize_models()
search = DDGS()  # Инициализация нового поисковика

# Загрузка данных
def load_data():
    try:
        df = pd.read_csv(CSV_URL, sep=';')
    except Exception:
        try:
            df = pd.read_csv(CSV_URL, on_bad_lines='skip')
        except Exception as e:
            logger.error(f"Не удалось загрузить данные: {str(e)}")
            raise
    
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
def setup_certificate():
    if not Path(CERT_PATH).exists():
        try:
            response = requests.get(CERT_URL)
            response.raise_for_status()
            with open(CERT_PATH, "wb") as f:
                f.write(response.content)
            logger.info("Сертификат успешно скачан")
            
            # Проверяем, что сертификат валиден
            context = ssl.create_default_context()
            context.load_verify_locations(cafile=CERT_PATH)
            logger.info("Сертификат успешно верифицирован")
        except Exception as e:
            logger.error(f"Не удалось скачать или верифицировать сертификат: {str(e)}")
            raise

text_documents = load_data()
vector_store = FAISS.from_documents(text_documents, embedding_model)
document_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Улучшенный промт
tourism_prompt = PromptTemplate.from_template("""
Ты - дружелюбный и профессиональный гид по городу Суздаль. Отвечай вежливо, информативно и структурированно.

Контекст из базы знаний:
{context}

Информация из интернета:
{web_search}

Пользовательский вопрос: {question}

Правила ответа:
1. Если вопрос слишком общий (менее 4 слов), вежливо попроси уточнить
2. Если информация есть в базе:
- Начни с "Вот что я нашел:"
- Название: [название]
- Тип: [тип]
- Описание: кратко и интересно
- Адрес: {address_section} (только если есть)
- Совет: полезный лайфхак или рекомендация
- В конце спроси: "Хотите узнать что-то еще?"

3. Если информации нет в базе, но есть в интернете:
- "В моей базе нет точной информации, но вот что удалось найти:"
- Краткая выжимка
- Источник: [ссылка]
- В конце спроси: "Это ответило на ваш вопрос?"

4. Если информации нет нигде:
- Извинись и предложи альтернативы
- Спроси о других интересах

5. В конце любого ответа добавь:
"Был ли полезен этот ответ? (Да/Нет)"

Твой ответ:
""")

def format_address(context_docs: list) -> str:
    if not context_docs:
        return ""
    
    main_doc = context_docs[0]
    address = main_doc.metadata.get("address", "не указано")
    
    return address if address.lower() not in ["не указано", "нет информации", ""] else ""

def perform_web_search(question: str) -> str:
    try:
        results = search.text(f"{question} Суздаль", region='ru-ru', max_results=3)
        if results:
            return "\n".join([f"{r['title']}: {r['body']} (Источник: {r['href']})" for r in results])
        return "Не найдено информации в интернете"
    except Exception as e:
        logger.error(f"Ошибка при поиске в интернете: {e}")
        return "Не удалось выполнить поиск в интернете"

def is_answer_in_context(context: list) -> bool:
    if not context:
        return False
    content = "\n".join(doc.page_content for doc in context)
    return "не указано" not in content and len(content.strip()) > 50

def prepare_prompt_input(question: str, context: list, web_search: str = "") -> dict:
    address_section = format_address(context)
    context_str = "\n\n".join([doc.page_content for doc in context]) if context else "Нет данных в базе"
    
    return {
        "context": context_str,
        "web_search": web_search,
        "question": question,
        "address_section": f"Адрес: {address_section}" if address_section else ""
    }

def refine_question(question: str) -> str:
    question = question.strip()
    if len(question.split()) < 3:
        suggestions = [
            "интересные музеи", 
            "места для детей",
            "исторические здания",
            "рекомендации по питанию",
            "церкви и монастыри"
        ]
        return (
            f"Ваш запрос '{question}' слишком общий. Пожалуйста, уточните, что вас интересует?\n"
            f"Например:\n- " + "\n- ".join(suggestions) + 
            "\n\nИли задайте более конкретный вопрос."
        )
    return None

def save_feedback(feedback: Feedback) -> None:
    try:
        data = []
        if Path(FEEDBACK_DB).exists():
            with open(FEEDBACK_DB, "r") as f:
                data = json.load(f)
        
        data.append(feedback.dict())
        
        with open(FEEDBACK_DB, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка сохранения фидбека: {e}")

def add_feedback_question(answer: str) -> str:
    if "Был ли полезен этот ответ?" not in answer:
        return answer + "\n\nБыл ли полезен этот ответ? (Да/Нет)"
    return answer

def process_feedback(question: str, answer: str, user_response: str) -> str:
    user_response = user_response.lower().strip()
    if user_response in ['да', 'yes', 'д', 'y']:
        is_helpful = True
        response = "Спасибо за ваш отзыв! Рад, что информация была полезной."
    elif user_response in ['нет', 'no', 'н', 'n']:
        is_helpful = False
        response = "Спасибо за отзыв! Постараюсь улучшить свои ответы."
    else:
        return "Не понял ваш ответ. Пожалуйста, ответьте 'Да' или 'Нет'."

    feedback = Feedback(
        question=question,
        answer=answer,
        is_helpful=is_helpful,
        timestamp=datetime.now().isoformat()
    )
    save_feedback(feedback)
    
    return response

rag_pipeline = (
    RunnableParallel(
        {
            "context": lambda x: document_retriever.invoke(x["question"]),
            "web_search": lambda x: perform_web_search(x["question"]),
            "question": lambda x: x["question"]
        }
    )
    | (lambda x: prepare_prompt_input(x["question"], x["context"], x["web_search"]))
    | tourism_prompt
    | ai_assistant
    | StrOutputParser()
)

def ask_question(question: str, is_feedback: bool = False, prev_answer: str = "") -> str:
    if is_feedback:
        return process_feedback(question, prev_answer, question)
    
    if not question.strip():
        return add_feedback_question("Пожалуйста, задайте ваш вопрос о Суздале.")
    
    refinement = refine_question(question)
    if refinement:
        return add_feedback_question(refinement)
    
    try:
        context = document_retriever.invoke(question)
        if is_answer_in_context(context):
            response = rag_pipeline.invoke({"question": question})
        else:
            web_results = perform_web_search(question)
            if "Не найдено" not in web_results:
                response = rag_pipeline.invoke({
                    "question": question,
                    "context": [],
                    "web_search": web_results
                })
            else:
                response = (
                    "К сожалению, я не нашел информации по вашему запросу.\n"
                    "Можете переформулировать вопрос или уточнить, что именно вас интересует?\n"
                    "Например, вы можете спросить о:\n- музеях\n- ресторанах\n- исторических местах\n"
                )
        
        return add_feedback_question(response)
    except Exception as e:
        logger.error(f"Ошибка обработки запроса: {e}")
        return add_feedback_question(
            "Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже."
        )

# FastAPI приложение
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
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response

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

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse('favicon.ico')

@app.get("/health")
async def health_check():
    return {"status": "OK", "service": "Suздаль Tourism Assistant"}

@app.post("/ask")
async def ask(item: Question):
    try:
        response = ask_question(item.question)
        return {"answer": response}
    except Exception as e:
        logger.error(f"Error in /ask: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/feedback")
async def submit_feedback(item: FeedbackRequest):
    try:
        response = ask_question(
            question=item.feedback,
            is_feedback=True,
            prev_answer=item.answer
        )
        return {"answer": response}
    except Exception as e:
        logger.error(f"Error in /feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/feedback")
async def get_feedback():
    try:
        if Path(FEEDBACK_DB).exists():
            with open(FEEDBACK_DB, "r") as f:
                return json.load(f)
        return []
    except Exception as e:
        logger.error(f"Error getting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
