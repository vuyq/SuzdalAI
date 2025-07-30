import os
import ssl
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
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# Загрузка переменных окружения
load_dotenv()

# Конфигурация
GIGACHAT_AUTH = os.getenv("GIGACHAT_AUTH") 
CERT_URL = os.getenv("CERT_URL")
CERT_PATH = os.getenv("CERT_PATH")
CSV_DATA_URL = "https://raw.githubusercontent.com/vuyq/SuzdalAI/refs/heads/main/suzdal_full_guide_refine/attractions.csv"

class Config:
    MAX_RETRIES = 3
    MIN_QUESTION_LENGTH = 4  # Минимальное количество слов для вопроса
    SEARCH_RESULTS = 3  # Количество результатов поиска
    RETRIEVER_K = 5  # Количество извлекаемых документов

# Загрузка сертификата
def download_certificate():
    if not Path(CERT_PATH).exists():
        try:
            response = requests.get(CERT_URL)
            response.raise_for_status()
            with open(CERT_PATH, "wb") as f:
                f.write(response.content)
            print("Сертификат успешно скачан")
        except Exception as e:
            raise Exception(f"Не удалось скачать сертификат: {str(e)}")

@retry(stop=stop_after_attempt(Config.MAX_RETRIES), 
       wait=wait_exponential(multiplier=1, min=2, max=10))
def get_gigachat_token():
    """Получение токена для GigaChat с обработкой ошибок"""
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': 'a2231e67-570e-47ca-bae8-82ca565850eb',
        'Authorization': f'Basic {GIGACHAT_AUTH}'
    }
    payload = {'scope': 'GIGACHAT_API_PERS'}
    
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
        raise Exception(f"Ошибка при получении токена: {str(e)}")

def initialize_models():
    """Инициализация моделей GigaChat"""
    try:
        access_token = get_gigachat_token()
        print("Токен успешно получен")
        
        embedding_model = GigaChatEmbeddings(
            access_token=access_token,
            model="Embeddings",
            scope="GIGACHAT_API_PERS",
            verify_ssl_certs=True,
            ca_bundle_file=CERT_PATH
        )
        
        ai_assistant = GigaChat(
            access_token=access_token,
            model="GigaChat-2",
            temperature=0,
            verify_ssl_certs=True,
            ca_bundle_file=CERT_PATH
        )
        
        return embedding_model, ai_assistant
    except Exception as e:
        print(f"Ошибка инициализации моделей: {str(e)}")
        raise

def load_data():
    """Загрузка и подготовка данных"""
    try:
        df = pd.read_csv(CSV_DATA_URL, sep=';')
    except Exception:
        df = pd.read_csv(CSV_DATA_URL, on_bad_lines='skip')
    
    text_documents = [
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
    return text_documents

# Инициализация компонентов
download_certificate()
embedding_model, ai_assistant = initialize_models()
text_documents = load_data()

# Создание векторного хранилища
vector_store = FAISS.from_documents(text_documents, embedding_model)
document_retriever = vector_store.as_retriever(search_kwargs={"k": Config.RETRIEVER_K})

# Инициализация поисковика
search = DuckDuckGoSearchRun(
    api_wrapper=DuckDuckGoSearchAPIWrapper(max_results=Config.SEARCH_RESULTS)
)

# Улучшенный промт с более четкой структурой
TOURISM_PROMPT_TEMPLATE = """
Привет! Я ваш виртуальный гид по Суздалю. Я постараюсь дать максимально полезный и точный ответ.

[Контекст из базы знаний]:
{context}

[Информация из интернета]:
{web_search}

[Ваш вопрос]:
{question}

Правила формирования ответа:
1. Если информация есть в базе:
- Начните с "Вот что я нашел:"
- Укажите название и тип места
- Кратко опишите, почему стоит посетить
- Добавьте адрес, если известен: {address_section}
- Дайте полезный совет или лайфхак
- В конце спросите, был ли ответ полезен

2. Если информации нет в базе, но есть в интернете:
- Начните с "В моей базе нет точной информации, но вот что я нашел в интернете:"
- Предоставьте краткую выжимку
- Укажите источник (если доступен)
- Предложите уточнить вопрос, если информация недостаточно точная
- В конце спросите, был ли ответ полезен

3. Если вопрос слишком общий:
- Вежливо попросите уточнить
- Приведите 2-3 примера, как можно уточнить вопрос
- Предложите помощь в формулировке запроса

4. Если информации нет вообще:
- Извинитесь
- Предложите альтернативные варианты (другие достопримечательности, общую информацию)
- Предложите поискать другую информацию по Суздалю

Ваш ответ (обязательно соблюдайте структуру и будьте дружелюбны):
"""

tourism_prompt = PromptTemplate.from_template(TOURISM_PROMPT_TEMPLATE)

def format_address(context_docs: list) -> str:
    """Форматирование адреса с проверкой наличия"""
    if not context_docs:
        return ""
    
    main_doc = context_docs[0]
    address = main_doc.metadata.get("address", "не указано")
    
    if address.lower() in ["не указано", "нет информации", ""]:
        return ""
    return address

def perform_web_search(question: str) -> str:
    """Выполнение поиска в интернете с обработкой ошибок"""
    try:
        search_query = f"site:ru {question} Суздаль"
        search_results = search.run(search_query)
        return search_results if search_results else "Не найдено информации в интернете"
    except Exception as e:
        print(f"Ошибка при поиске в интернете: {e}")
        return "Не удалось выполнить поиск в интернете"

def is_answer_in_context(context: list) -> bool:
    """Проверка наличия полезной информации в контексте"""
    if not context:
        return False
    content = "\n".join(doc.page_content for doc in context)
    return "не указано" not in content and len(content.strip()) > 50

def prepare_prompt_input(question: str, context: list, web_search: str = "") -> dict:
    """Подготовка входных данных для промта"""
    address_section = format_address(context)
    context_str = "\n\n".join([doc.page_content for doc in context]) if context else "Нет данных в базе"
    
    return {
        "context": context_str,
        "web_search": web_search,
        "question": question,
        "address_section": address_section if address_section else "адрес не указан"
    }

def add_feedback_request(response: str) -> str:
    """Добавление запроса обратной связи к ответу"""
    feedback_prompt = "\n\nБыл ли этот ответ полезен для вас? Если у вас есть дополнительные вопросы, не стесняйтесь задавать!"
    return response + feedback_prompt

def refine_question(question: str) -> str:
    """Уточнение слишком общего вопроса"""
    words = question.strip().split()
    if len(words) < Config.MIN_QUESTION_LENGTH:
        clarification_examples = [
            "- Конкретное место (например, 'Суздальский кремль')",
            "- Тип достопримечательности (музеи, храмы, рестораны)",
            "- Интересы (история, архитектура, активный отдых)"
        ]
        examples = "\n".join(clarification_examples)
        return (
            f"Ваш вопрос довольно общий: '{question}'. Чтобы я мог дать более точный ответ, "
            f"уточните, пожалуйста, что именно вас интересует. Например:\n{examples}\n"
            "Можете задать вопрос более подробно?"
        )
    return None

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
    | add_feedback_request
)

def ask_question(question: str) -> str:
    """Основная функция обработки вопроса"""
    # Проверка на пустой вопрос
    if not question.strip():
        return "Пожалуйста, задайте ваш вопрос о Суздале. Я постараюсь помочь!"
    
    # Уточнение слишком общего вопроса
    refinement = refine_question(question)
    if refinement:
        return refinement
    
    # Поиск в локальной базе
    context = document_retriever.invoke(question)
    
    if is_answer_in_context(context):
        try:
            response = rag_pipeline.invoke({"question": question})
            return response
        except Exception as e:
            print(f"Ошибка при обработке вопроса: {e}")
            return "Произошла ошибка при обработке вашего вопроса. Пожалуйста, попробуйте позже."
    else:
        # Поиск в интернете, если в базе нет информации
        web_results = perform_web_search(question)
        if "Не найдено" not in web_results:
            try:
                response = rag_pipeline.invoke({
                    "question": question,
                    "context": [],
                    "web_search": web_results
                })
                return response
            except Exception as e:
                print(f"Ошибка при обработке веб-результатов: {e}")
        
        # Если информация не найдена нигде
        return (
            "К сожалению, мне не удалось найти информацию по вашему запросу. "
            "Можете попробовать переформулировать вопрос или уточнить детали. "
            "Также я могу предложить общую информацию о Суздале или популярные достопримечательности."
        )

# FastAPI приложение
app = FastAPI(title="Suздаль Tourism Assistant", 
              description="AI помощник по туризму в Суздале")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask(item: Question):
    try:
        response = ask_question(item.question)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
