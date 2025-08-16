import os
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_gigachat import GigaChat, GigaChatEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential
from ddgs import DDGS
import logging
from typing import Dict, List, Optional

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

class Config:
    # Основные настройки
    MAX_RETRIES = 3
    REQUEST_TIMEOUT = 15
    MIN_QUESTION_LENGTH = 2
    SEARCH_RESULTS = 3
    RETRIEVER_K = 5
    MAX_CONTEXT_LENGTH = 3
    
    # Пути и URL
    CERT_PATH = os.getenv("CERT_PATH", "./cert.pem")
    CERT_URL = os.getenv("CERT_URL")
    GIGACHAT_AUTH = os.getenv("GIGACHAT_AUTH")
    CSV_DATA_URL = os.getenv("CSV_DATA_URL", "https://raw.githubusercontent.com/vuyq/SuzdalAI/main/suzdal_full_guide_refine/attractions.csv")
    
    # Ключевые слова
    FOOD_KEYWORDS = ["еда", "поесть", "кафе", "ресторан", "перекусить", "кухня", "столовая", "меню"]
    MUSEUM_KEYWORDS = ["музей", "музеи", "экспозиция", "выставка", "галерея"]
    ATTRACTION_KEYWORDS = ["достопримечательность", "что посмотреть", "что посетить", "интересное место"]

# Глобальное хранилище контекста
DIALOG_CONTEXTS: Dict[str, List[Dict[str, str]]] = {}

def download_certificate():
    """Загрузка SSL-сертификата при необходимости"""
    if not Path(Config.CERT_PATH).exists():
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
        'RqUID': 'a2231e67-570e-47ca-bae8-82ca565850eb',
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

def initialize_models() -> tuple:
    """Инициализация моделей GigaChat"""
    try:
        access_token = get_gigachat_token()
        logger.info("Токен GigaChat получен успешно")
        
        embedding_model = GigaChatEmbeddings(
            access_token=access_token,
            model="Embeddings",
            scope="GIGACHAT_API_PERS",
            verify_ssl_certs=True,
            ca_bundle_file=Config.CERT_PATH,
            timeout=Config.REQUEST_TIMEOUT
        )
        
        ai_assistant = GigaChat(
            access_token=access_token,
            model="GigaChat-2",
            temperature=0.2,
            verify_ssl_certs=True,
            ca_bundle_file=Config.CERT_PATH,
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
                    
                if col.lower() in ['name', 'type', 'tags', 'address']:
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

def update_dialog_context(user_id: str, role: str, message: str):
    """Обновление контекста диалога"""
    if user_id not in DIALOG_CONTEXTS:
        DIALOG_CONTEXTS[user_id] = []
    
    DIALOG_CONTEXTS[user_id].append({"role": role, "message": message})
    
    if len(DIALOG_CONTEXTS[user_id]) > Config.MAX_CONTEXT_LENGTH:
        DIALOG_CONTEXTS[user_id] = DIALOG_CONTEXTS[user_id][-Config.MAX_CONTEXT_LENGTH:]

def get_dialog_context(user_id: str) -> str:
    """Получение форматированного контекста диалога"""
    if user_id not in DIALOG_CONTEXTS or not DIALOG_CONTEXTS[user_id]:
        return ""
    
    return "\n".join(
        f"{item['role']}: {item['message']}" 
        for item in DIALOG_CONTEXTS[user_id]
    )

def classify_question(question: str) -> str:
    """Классификация вопроса по категориям"""
    question_lower = question.lower()
    
    if any(keyword in question_lower for keyword in Config.FOOD_KEYWORDS):
        return "food"
    elif any(keyword in question_lower for keyword in Config.MUSEUM_KEYWORDS):
        return "museum"
    elif any(keyword in question_lower for keyword in Config.ATTRACTION_KEYWORDS):
        return "attraction"
    return "general"

def refine_question(question: str, user_id: str) -> Optional[str]:
    """Проверка необходимости уточнения вопроса"""
    question = question.strip()
    if len(question) < Config.MIN_QUESTION_LENGTH:
        return "Пожалуйста, уточните ваш вопрос. Например:\n- Какие музеи стоит посетить?\n- Где можно попробовать медовуху?"
    
    question_type = classify_question(question)
    
    # Проверка предыдущего контекста
    context = get_dialog_context(user_id)
    if "assistant: Уточните" in context:
        update_dialog_context(user_id, "clarification", question)
        return None
    
    if question_type == "food" and len(question.split()) < 5:
        return (
            "Я могу порекомендовать места по критериям:\n"
            "- Тип кухни (русская, итальянская...)\n"
            "- Расположение (центр, рядом с кремлем...)\n"
            "- Бюджет (эконом, средний, премиум)\n"
            "Что для вас важнее?"
        )
    
    return None

def format_response(docs: List[Document], question_type: str) -> str:
    """Форматирование ответа на основе документов"""
    if not docs:
        return "Информация не найдена."
    
    response = []
    
    if question_type == "food":
        response.append("🍽 Вот варианты где можно поесть:")
        for doc in docs[:5]:
            name = doc.metadata.get("name", "Заведение")
            cuisine = doc.metadata.get("type", "кухня не указана")
            address = doc.metadata.get("address", "адрес не указан")
            response.append(f"\n• {name} ({cuisine})\n  📍 {address}")
    
    elif question_type == "museum":
        response.append("🏛️ Музеи и выставки:")
        for doc in docs[:5]:
            name = doc.metadata.get("name", "Экспозиция")
            desc = doc.page_content.split("\n")[0][:100] + "..."
            address = doc.metadata.get("address", "адрес не указан")
            response.append(f"\n• {name}\n  {desc}\n  📍 {address}")
    
    else:
        response.append("🏰 Достопримечательности:")
        for doc in docs[:5]:
            name = doc.metadata.get("name", "Место")
            desc = doc.page_content.split("\n")[0][:100] + "..."
            address = doc.metadata.get("address", "адрес не указан")
            response.append(f"\n• {name}\n  {desc}\n  📍 {address}")
    
    response.append("\nМогу уточнить детали по любому из мест.")
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

Сформируй ответ по правилам:
1. Для заведений питания укажи:
   - Название и тип кухни
   - Средний чек (если есть)
   - Адрес и особенности

2. Для музеев укажи:
   - Название и описание
   - Часы работы и стоимость
   - Интересные факты

3. Для других мест:
   - Краткое описание
   - Почему стоит посетить
   - Как добраться

Будь вежлив и предлагай уточнения если нужно.
"""

tourism_prompt = PromptTemplate.from_template(TOURISM_PROMPT_TEMPLATE)

def generate_response(question: str, context_docs: List[Document], 
                    web_results: str, dialog_context: str) -> str:
    """Генерация ответа с помощью GigaChat"""
    try:
        prompt_input = {
            "question": question,
            "context": "\n\n".join(d.page_content for d in context_docs) if context_docs else "Нет данных",
            "web_search": web_results,
            "dialog_context": dialog_context
        }
        
        response = ai_assistant.invoke(tourism_prompt.format(**prompt_input))
        return response.content if hasattr(response, 'content') else str(response)
    
    except Exception as e:
        logger.error(f"Ошибка генерации ответа: {e}")
        question_type = classify_question(question)
        return format_response(context_docs, question_type)

def handle_question(question: str, user_id: str) -> str:
    """Основная обработка вопроса"""
    try:
        # Проверка и уточнение вопроса
        refinement = refine_question(question, user_id)
        if refinement:
            update_dialog_context(user_id, "assistant", refinement)
            return refinement
        
        # Поиск в базе знаний
        question_type = classify_question(question)
        context_docs = document_retriever.invoke(question)
        
        # Если есть результаты в базе
        if context_docs:
            # Веб-поиск только если мало результатов
            web_results = ""
            if len(context_docs) < 2:
                web_results = perform_web_search(question)
            
            dialog_context = get_dialog_context(user_id)
            response = generate_response(question, context_docs, web_results, dialog_context)
        else:
            # Если в базе нет результатов - только веб-поиск
            web_results = perform_web_search(question)
            dialog_context = get_dialog_context(user_id)
            response = generate_response(question, [], web_results, dialog_context)
        
        # Сохраняем контекст
        update_dialog_context(user_id, "assistant", response)
        return response + "\n\nБыл ли ответ полезен? Задайте уточняющие вопросы!"
    
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
except Exception as e:
    logger.critical(f"Ошибка инициализации: {e}")
    raise

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
async def ask(item: Question):
    try:
        if not item.question.strip():
            return {"answer": "Пожалуйста, задайте ваш вопрос."}
        
        response = handle_question(item.question, item.user_id)
        return {"answer": response}
    
    except Exception as e:
        logger.error(f"API error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Внутренняя ошибка сервера. Пожалуйста, попробуйте позже."
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
