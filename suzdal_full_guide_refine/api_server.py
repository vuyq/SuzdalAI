import os
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
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# Загрузка конфигурации
load_dotenv()

# Конфигурационные переменные
GIGACHAT_AUTH = os.getenv("GIGACHAT_AUTH")
CERT_URL = os.getenv("CERT_URL")
CERT_PATH = os.getenv("CERT_PATH")

# Скачивание сертификата
if not Path(CERT_PATH).exists():
    try:
        response = requests.get(CERT_URL)
        response.raise_for_status()
        with open(CERT_PATH, "wb") as f:
            f.write(response.content)
        print("Сертификат успешно загружен")
    except Exception as e:
        raise Exception(f"Ошибка загрузки сертификата: {str(e)}")

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
        print("Токен GigaChat успешно получен")
        return GigaChat(
            access_token=access_token,
            model="GigaChat-2",
            temperature=0,
            verify_ssl_certs=True,
            ca_bundle_file=CERT_PATH
        )
    except Exception as e:
        raise Exception(f"Ошибка инициализации GigaChat: {str(e)}")

# Инициализация компонентов
try:
    ai_assistant = init_gigachat()
    search = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper(max_results=3))
    print("Компоненты успешно инициализированы")
except Exception as e:
    print(f"Ошибка инициализации: {str(e)}")
    raise

# Загрузка данных
def load_attractions_data():
    csv_url = "https://raw.githubusercontent.com/vuyq/SuzdalAI/main/suzdal_full_guide_refine/attractions.csv"
    try:
        df = pd.read_csv(csv_url, sep=';')
        print(f"Загружено {len(df)} достопримечательностей")
        return df
    except Exception as e:
        print(f"Ошибка загрузки данных: {str(e)}")
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
    print(f"Создано {len(documents)} документов для векторного поиска")
    return FAISS.from_documents(documents, GigaChatEmbeddings(
        access_token=get_gigachat_token(),
        model="Embeddings",
        scope="GIGACHAT_API_PERS",
        verify_ssl_certs=True,
        ca_bundle_file=CERT_PATH
    ))

# Инициализация данных
try:
    df = load_attractions_data()
    vector_store = create_vector_store(df)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    print("Векторное хранилище создано")
except Exception as e:
    print(f"Ошибка инициализации данных: {str(e)}")
    raise

# Шаблон ответа на русском языке
prompt_template = PromptTemplate.from_template("""
Ты - гид по Суздалю. Отвечай только на русском языке. На вопрос: {question}

Найдены следующие достопримечательности:
{context}

Сформируй развернутый ответ, включив ВСЕ подходящие варианты. Для каждого места укажи:

📍 {название из metadata}
🎯 {type из metadata}
❤️ Почему стоит посетить: [краткое описание уникальных особенностей]
🔍 [интересные детали и исторические факты]
📌 Адрес: {address из metadata}
💡 Важно: [практическая информация или советы]

Если вариантов несколько - разделяй их пустой строкой.
""")

# Форматирование контекста
def format_context(docs):
    formatted = []
    for doc in docs:
        content_lines = doc.page_content.split('\n')
        description = next((line.split(':')[1].strip() for line in content_lines if line.startswith('Description:')), "не указано")[0]
        
        formatted.append(
            f"Название: {doc.metadata['title']}\n"
            f"Тип: {doc.metadata['type']}\n"
            f"Адрес: {doc.metadata.get('address', 'не указан')}\n"
            f"Описание: {description[:200]}{'...' if len(description) > 200 else ''}"
        )
    return "\n\n".join(formatted)

# Цепочка обработки
rag_chain = (
    RunnableParallel({
        "context": retriever | format_context,
        "question": RunnablePassthrough()
    })
    | prompt_template
    | ai_assistant
    | StrOutputParser()
)

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
async def ask_question(question: str = Body(..., embed=True)):
    try:
        logger.info(f"Processing question: {question}")
        docs = retriever.invoke(question)
        logger.info(f"Retrieved {len(docs)} documents")
        context = format_context(docs)
        logger.info(f"Formatted context: {context[:200]}...")
        
        answer = rag_chain.invoke(question)
        return {
            "answer": answer,
            "feedback_request": {
                "text": "Был ли этот ответ полезен?",
                "options": ["Да", "Нет"]
            }
        }
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

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
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
