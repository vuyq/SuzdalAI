import os
import logging
import asyncio
import time
import requests
import uvicorn
import pandas as pd
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.documents import Document
from langchain_gigachat import GigaChat, GigaChatEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Загрузка конфигурации
load_dotenv()

# Конфигурационные переменные
GIGACHAT_AUTH = os.getenv("GIGACHAT_AUTH")
CERT_URL = os.getenv("CERT_URL")
CERT_PATH = os.getenv("CERT_PATH")

# Глобальные переменные для кэширования
GIGA_CHAT_INSTANCE = None
VECTOR_STORE = None
SEARCH_TOOL = None

# Скачивание сертификата
if not Path(CERT_PATH).exists():
    try:
        response = requests.get(CERT_URL)
        response.raise_for_status()
        with open(CERT_PATH, "wb") as f:
            f.write(response.content)
        logger.info("Сертификат успешно загружен")
    except Exception as e:
        logger.error(f"Ошибка загрузки сертификата: {str(e)}")
        raise

# Получение токена GigaChat
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
    response = requests.post(
        url, 
        headers=headers, 
        data=payload, 
        verify=CERT_PATH,
        timeout=10
    )
    response.raise_for_status()
    return response.json().get("access_token")

# Инициализация GigaChat
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def init_gigachat():
    try:
        access_token = get_gigachat_token()
        logger.info("Токен GigaChat успешно получен")
        return GigaChat(
            access_token=access_token,
            model="GigaChat-2",
            temperature=0,
            verify_ssl_certs=True,
            ca_bundle_file=CERT_PATH
        )
    except Exception as e:
        logger.error(f"Ошибка инициализации GigaChat: {str(e)}")
        raise

# Загрузка данных
def load_attractions_data():
    csv_url = "https://raw.githubusercontent.com/vuyq/SuzdalAI/main/suzdal_full_guide_refine/attractions.csv"
    try:
        df = pd.read_csv(csv_url, sep=';')
        logger.info(f"Загружено {len(df)} достопримечательностей")
        return df
    except Exception as e:
        logger.error(f"Ошибка загрузки данных: {str(e)}")
        raise

# Создание векторного хранилища
def create_vector_store(df):
    documents = [
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
    logger.info(f"Создано {len(documents)} документов для векторного поиска")
    
    embeddings = GigaChatEmbeddings(
        access_token=get_gigachat_token(),
        model="Embeddings",
        scope="GIGACHAT_API_PERS",
        verify_ssl_certs=True,
        ca_bundle_file=CERT_PATH
    )
    
    return FAISS.from_documents(documents, embeddings)

# Шаблон ответа на русском языке
prompt_template = PromptTemplate.from_template("""
Ты - гид по Суздалю. Отвечай только на русском языке. На вопрос: {question}

Найдены следующие достопримечательности:
{context}

Сформируй развернутый ответ, включив ВСЕ подходящие варианты. Для каждого места укажи:

📍 {название из metadata}
{type из metadata}
[краткое описание уникальных особенностей]
[интересные детали и исторические факты]
{address из metadata}
💡 Важно: [практическая информация или советы]

Если вариантов несколько - разделяй их пустой строкой.
""")

# Оптимизированная функция форматирования контекста
def format_context(docs):
    return "\n\n".join(
        f"Название: {doc.metadata['title']}\n"
        f"Тип: {doc.metadata['type']}\n"
        f"Адрес: {doc.metadata.get('address', 'не указан')}\n"
        f"Описание: {doc.page_content[:200]}{'...' if len(doc.page_content) > 200 else ''}"
        for doc in docs
    )

# Инициализация сервисов при старте приложения
async def initialize_services():
    global GIGA_CHAT_INSTANCE, VECTOR_STORE, SEARCH_TOOL
    
    try:
        # Инициализация GigaChat
        if GIGA_CHAT_INSTANCE is None:
            GIGA_CHAT_INSTANCE = init_gigachat()
        
        # Загрузка данных и создание векторного хранилища
        if VECTOR_STORE is None:
            df = load_attractions_data()
            VECTOR_STORE = create_vector_store(df)
        
        # Инициализация поискового инструмента
        if SEARCH_TOOL is None:
            SEARCH_TOOL = DuckDuckGoSearchRun(
                api_wrapper=DuckDuckGoSearchAPIWrapper(max_results=2)
            )
        
        logger.info("Все сервисы успешно инициализированы")
    except Exception as e:
        logger.error(f"Ошибка инициализации сервисов: {str(e)}")
        raise

# FastAPI приложение
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация при старте
@app.on_event("startup")
async def startup_event():
    await initialize_services()

@app.post("/ask")
async def ask_question(question: str = Body(..., embed=True)):
    try:
        start_time = time.time()
        
        # Получаем релевантные документы
        retriever = VECTOR_STORE.as_retriever(search_kwargs={"k": 3})
        docs = await asyncio.to_thread(retriever.invoke, question)
        
        # Формируем цепочку обработки
        rag_chain = (
            RunnableParallel({
                "context": lambda x: format_context(docs),
                "question": RunnablePassthrough()
            })
            | prompt_template
            | GIGA_CHAT_INSTANCE
            | StrOutputParser()
        )
        
        # Выполняем асинхронно
        answer = await asyncio.to_thread(rag_chain.invoke, question)
        
        logger.info(f"Запрос обработан за {time.time() - start_time:.2f} сек")
        return {
            "answer": answer,
            "processing_time": f"{time.time() - start_time:.2f} сек",
            "feedback_request": {
                "text": "Был ли этот ответ полезен?",
                "options": ["Да", "Нет"]
            }
        }
    except Exception as e:
        logger.error(f"Ошибка обработки запроса: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/feedback")
async def handle_feedback(
    is_helpful: bool = Body(...),
    comment: Optional[str] = Body(None)
):
    try:
        if is_helpful:
            return {"message": "Спасибо за ваш отзыв! Рады, что помогли."}
        else:
            msg = "Спасибо за обратную связь."
            if comment:
                msg += f" Ваш комментарий: '{comment}'"
            return {"message": msg}
    except Exception as e:
        logger.error(f"Ошибка обработки feedback: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
