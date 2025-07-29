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

load_dotenv()

GIGACHAT_AUTH = os.getenv("GIGACHAT_AUTH") 
CERT_URL = os.getenv("CERT_URL")
CERT_PATH = os.getenv("CERT_PATH")

if not Path(CERT_PATH).exists():
    try:
        response = requests.get(CERT_URL)
        response.raise_for_status()
        with open(CERT_PATH, "wb") as f:
            f.write(response.content)
        print("Сертификат успешно скачан")
    except Exception as e:
        raise Exception(f"Не удалось скачать сертификат: {str(e)}")

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
except Exception as e:
    print(f"Ошибка инициализации: {str(e)}")
    raise

# Инициализация поисковика DuckDuckGo
search = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper(max_results=3))

csv_file_path = "https://raw.githubusercontent.com/vuyq/SuzdalAI/refs/heads/main/suzdal_full_guide_refine/attractions.csv"  
try:
    df = pd.read_csv(csv_file_path, sep=';')
except Exception:
    df = pd.read_csv(csv_file_path, on_bad_lines='skip')

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
            "address": row.get("Address", "не указано")  # Сохраняем адрес в метаданных
        }
    )
    for _, row in df.iterrows()
]

vector_store = FAISS.from_documents(text_documents, embedding_model)
document_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

tourism_prompt = PromptTemplate.from_template("""
Привет! Я твой гид по Суздалю. Отвечаю просто и по делу:

1. Сначала проверю свою базу знаний
2. Если не найду - посмотрю в интернете
3. Если вопрос непонятен - уточню

Мои данные:
{context}

Информация из интернета:
{web_search}

Твой вопрос: {question}

Формат ответа:
Если знаю ответ:
"Вот что нашел:"
- [Название] | [Тип] 
- Почему стоит посетить: кратко
- Где найти: {address_section}
- Лайфхак: полезная фишка

Если данных нет в базе, но есть в интернете:
"В моей базе нет точной информации, но вот что удалось найти:"
[краткая выжимка из интернета]
[источник: ссылка]

Если вопрос расплывчатый:
"Уточни, пожалуйста, что тебя интересует? Например: [варианты уточнений]"

Твой ответ:
""")

def format_address(context_docs: list) -> str:
    """Форматирует раздел с адресом, оставляя пустым если адреса нет"""
    if not context_docs:
        return ""
    
    # Берем первый документ (наиболее релевантный)
    main_doc = context_docs[0]
    address = main_doc.metadata.get("address", "не указано")
    
    if address.lower() in ["не указано", "нет информации", ""]:
        return ""
    else:
        return address

def perform_web_search(question: str) -> str:
    try:
        search_results = search.run(f"site:ru {question} Суздаль")
        return search_results if search_results else "Не найдено информации в интернете"
    except Exception as e:
        print(f"Ошибка при поиске в интернете: {e}")
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
        "address_section": address_section if address_section else "адрес не указан"
    }

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

def refine_question(question: str) -> str:
    if len(question.strip().split()) < 4:
        return f"Ваш вопрос довольно общий: '{question}'. Уточните, что именно вас интересует (тип достопримечательности, возраст посетителей, архитектура, кухня и т.д.)"
    return None

def ask_question(question):
    refinement = refine_question(question)
    if refinement:
        return refinement
    
    # Сначала проверяем локальную базу
    context = document_retriever.invoke(question)
    if is_answer_in_context(context):
        return rag_pipeline.invoke({"question": question})
    else:
        # Если в базе нет информации, используем веб-поиск
        web_results = perform_web_search(question)
        if "Не найдено" not in web_results:
            return rag_pipeline.invoke({
                "question": question,
                "context": [],
                "web_search": web_results
            })
        return "К сожалению, не удалось найти информацию ни в моей базе, ни в интернете. Попробуйте переформулировать вопрос."

app = FastAPI()
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
    response = ask_question(item.question)
    return {"answer": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
