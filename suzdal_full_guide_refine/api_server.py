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
import os
import logging
import asyncio
import time
import requests
import pandas as pd
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.documents import Document
from langchain_gigachat import GigaChat, GigaChatEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from tenacity import retry, stop_after_attempt, wait_exponential

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
RETRIEVER = None

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
                "address": row.get("Address", "не указано"),
                "description": row.get("Description", "")
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

# Инициализация сервисов
async def initialize_services():
    global GIGA_CHAT_INSTANCE, VECTOR_STORE, RETRIEVER
    
    try:
        # Инициализация GigaChat
        if GIGA_CHAT_INSTANCE is None:
            GIGA_CHAT_INSTANCE = init_gigachat()
        
        # Загрузка данных и создание векторного хранилища
        if VECTOR_STORE is None:
            df = load_attractions_data()
            VECTOR_STORE = create_vector_store(df)
            RETRIEVER = VECTOR_STORE.as_retriever(search_kwargs={"k": 3})
        
        logger.info("Все сервисы успешно инициализированы")
    except Exception as e:
        logger.error(f"Ошибка инициализации сервисов: {str(e)}")
        raise

prompt_template = PromptTemplate.from_template("""
Ты - профессиональный гид-экскурсовод по городу Суздаль. Отвечай ТОЛЬКО на русском языке в дружелюбном и информативном стиле.

Текущий запрос пользователя: 
{question}

Найдены подходящие достопримечательности:
{context}

Сформируй ИСЧЕРПЫВАЮЩИЙ ответ по следующей структуре:

1. 📍 Название: {title}
2. 🏛 Тип: {type} 
3. 🗺 Адрес: {address}
4. ❤️ Почему стоит посетить: {description}
5. 🔍 Интересные факты: [приведи 2-3 уникальных факта]
6. ⏰ Часы работы: [укажи если есть в данных]
7. 💰 Стоимость: [укажи билеты если есть информация]
8. 💡 Советы посетителям: [практические рекомендации]
9. 🚶 Как добраться: [кратко опиши маршрут от центра]

Если вариантов несколько - разделяй их 2 пустыми строками для лучшей читаемости.

После ответа ЗАДАЙ 2 УТОЧНЯЮЩИХ ВОПРОСА для уточнения потребностей пользователя (например: 
"Вас интересуют больше музеи или храмы?" 
"Планируете посещение с детьми?")

В конце добавь призыв к обратной связи:
"Был ли полезен мой ответ? Нам важно ваше мнение для улучшения сервиса!"

Сохраняй ДОБРОЖЕЛАТЕЛЬНЫЙ ТОН и ПРОФЕССИОНАЛИЗМ. Избегай сухого перечисления фактов - делай ответ живым и увлекательным!
""")

# Форматирование контекста
def format_context(docs):
    if not docs:
        return "Не найдено подходящих достопримечательностей"
    
    formatted = []
    for doc in docs:
        formatted.append(
            f"Название: {doc.metadata.get('title', 'Неизвестно')}\n"
            f"Тип: {doc.metadata.get('type', 'Неизвестно')}\n"
            f"Адрес: {doc.metadata.get('address', 'Не указан')}\n"
            f"Описание: {doc.metadata.get('description', doc.page_content[:200])}"
        )
    return "\n\n".join(formatted)

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
        
        # Валидация входящего запроса
        question = question.strip()
        if not question:
            raise HTTPException(
                status_code=400,
                detail="Пожалуйста, задайте вопрос о достопримечательностях Суздаля"
            )
        
        logger.info(f"Обработка вопроса: '{question}'")
        
        # Асинхронное получение релевантных документов
        docs = await asyncio.to_thread(
            RETRIEVER.invoke,
            question,
            config={"max_concurrency": 5}
        )
        
        # Форматирование контекста с обработкой пустого результата
        context = format_context(docs) if docs else "Не найдено подходящих достопримечательностей"
        
        # Подготовка данных для промпта
        input_data = {
            "context": context,
            "question": question,
            "title": docs[0].metadata["title"] if docs else "Неизвестно",
            "type": docs[0].metadata.get("type", "Неизвестно") if docs else "Неизвестно",
            "address": docs[0].metadata.get("address", "Не указан") if docs else "Не указан",
            "description": docs[0].metadata.get("description", "") if docs else ""
        }
        
        # Создание цепочки обработки с обработкой ошибок
        try:
            answer = await asyncio.wait_for(
                asyncio.to_thread(
                    RunnableParallel({
                        "context": lambda x: x["context"],
                        "question": RunnablePassthrough(),
                        **{k: lambda x, key=k: x[key] for k in ["title", "type", "address", "description"]}
                    })
                    | create_prompt_template()
                    | GIGA_CHAT_INSTANCE
                    | StrOutputParser()
                    .invoke,
                    input_data
                ),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="Превышено время ожидания ответа. Пожалуйста, попробуйте позже."
            )
        
        # Логирование и возврат результата
        processing_time = time.time() - start_time
        logger.info(
            f"Успешно обработан вопрос за {processing_time:.2f} сек: "
            f"'{question[:50]}...'"
        )
        
        return {
            "answer": answer,
            "processing_time": f"{processing_time:.2f} сек",
            "suggested_questions": [
                "Какие ещё достопримечательности вас интересуют?",
                "Нужна ли информация о времени работы или стоимости билетов?"
            ],
            "request_feedback": True
        }
        
    except HTTPException:
        raise  # Пробрасываем уже обработанные HTTP исключения
        
    except Exception as e:
        logger.error(
            f"Ошибка при обработке вопроса '{question}': {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте ещё раз."
        )

@app.post("/feedback")
async def collect_feedback(
    is_helpful: bool = Body(...),
    comment: str = Body(None),
    question: str = Body(...)
):
    # Логируем feedback для анализа
    logger.info(f"Feedback: helpful={is_helpful}, comment={comment}, question={question}")
    return {"message": "Спасибо за ваш отзыв!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
