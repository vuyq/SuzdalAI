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
CSV_URL = "https://raw.githubusercontent.com/vuyq/SuzdalAI/refs/heads/main/suzdal_full_guide_refine/attractions.csv"

# Проверка и загрузка сертификата
def setup_certificate():
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
        raise Exception(f"Ошибка получения токена: {str(e)}")

def initialize_models():
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
            temperature=0.2,  # Немного повысим для более естественных ответов
            verify_ssl_certs=True,
            ca_bundle_file=CERT_PATH
        )
        
        return embedding_model, ai_assistant
    except Exception as e:
        print(f"Ошибка инициализации: {str(e)}")
        raise

# Инициализация компонентов
setup_certificate()
embedding_model, ai_assistant = initialize_models()
search = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper(max_results=3))

# Загрузка данных
def load_data():
    try:
        df = pd.read_csv(CSV_URL, sep=';')
    except Exception:
        try:
            df = pd.read_csv(CSV_URL, on_bad_lines='skip')
        except Exception as e:
            raise Exception(f"Не удалось загрузить данные: {str(e)}")
    
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
        "address_section": f"Адрес: {address_section}" if address_section else ""
    }

def refine_question(question: str) -> str:
    question = question.strip()
    if len(question.split()) < 3:  # Уменьшил порог для более частых уточнений
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

def ask_question(question: str) -> str:
    if not question.strip():
        return "Пожалуйста, задайте ваш вопрос о Суздале."
    
    refinement = refine_question(question)
    if refinement:
        return refinement
    
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
                    "Был ли полезен этот ответ? (Да/Нет)"
                )
        
        # Добавляем вопрос о качестве ответа, если его еще нет
        if "Был ли полезен этот ответ?" not in response:
            response += "\n\nБыл ли полезен этот ответ? (Да/Нет)"
            
        return response
    except Exception as e:
        print(f"Ошибка обработки запроса: {e}")
        return (
            "Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже.\n"
            "Был ли полезен этот ответ? (Да/Нет)"
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
