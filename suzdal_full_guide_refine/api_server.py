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
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from tenacity import retry, stop_after_attempt, wait_exponential
from ddgs import DDGS

# Загрузка переменных окружения
load_dotenv()

# Конфигурация
class Config:
    MAX_RETRIES = 3
    MIN_QUESTION_LENGTH = 2
    SEARCH_RESULTS = 3
    RETRIEVER_K = 5
    CERT_PATH = os.getenv("CERT_PATH")
    CERT_URL = os.getenv("CERT_URL")
    GIGACHAT_AUTH = os.getenv("GIGACHAT_AUTH")
    CSV_DATA_URL = "https://raw.githubusercontent.com/vuyq/SuzdalAI/refs/heads/main/suzdal_full_guide_refine/attractions.csv"
    MAX_CONTEXT_LENGTH = 3
    FOOD_KEYWORDS = ["еда", "поесть", "кафе", "ресторан", "перекусить", "кухня", "питание", "обед", "ужин"]
    MUSEUM_KEYWORDS = ["музей", "музеи", "экспозиция", "выставка", "галерея"]

# Глобальное хранилище контекста диалогов
DIALOG_CONTEXTS = {}

def download_certificate():
    if not Path(Config.CERT_PATH).exists():
        try:
            response = requests.get(Config.CERT_URL)
            response.raise_for_status()
            with open(Config.CERT_PATH, "wb") as f:
                f.write(response.content)
            print("Сертификат успешно скачан")
        except Exception as e:
            raise Exception(f"Не удалось скачать сертификат: {str(e)}")

@retry(stop=stop_after_attempt(Config.MAX_RETRIES), 
       wait=wait_exponential(multiplier=1, min=2, max=10))
def get_gigachat_token():
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
            timeout=10
        )
        response.raise_for_status()
        return response.json().get("access_token")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Ошибка при получении токена: {str(e)}")

def initialize_models():
    try:
        access_token = get_gigachat_token()
        print("Токен успешно получен")
        
        embedding_model = GigaChatEmbeddings(
            access_token=access_token,
            model="Embeddings",
            scope="GIGACHAT_API_PERS",
            verify_ssl_certs=True,
            ca_bundle_file=Config.CERT_PATH
        )
        
        ai_assistant = GigaChat(
            access_token=access_token,
            model="GigaChat-2",
            temperature=0.2,
            verify_ssl_certs=True,
            ca_bundle_file=Config.CERT_PATH
        )
        
        return embedding_model, ai_assistant
    except Exception as e:
        print(f"Ошибка инициализации моделей: {str(e)}")
        raise

def load_data():
    try:
        df = pd.read_csv(Config.CSV_DATA_URL, sep=';')
    except Exception:
        try:
            df = pd.read_csv(Config.CSV_DATA_URL)
        except Exception as e:
            print(f"Ошибка загрузки данных: {e}")
            return []
    
    text_documents = []
    for _, row in df.iterrows():
        try:
            doc_content = []
            for col, val in row.items():
                if pd.notna(val):
                    doc_content.append(f"{col}: {val}")
            
            doc = Document(
                page_content="\n".join(doc_content),
                metadata={
                    "title": row.get("Name", ""),
                    "type": row.get("Type", ""),
                    "tags": row.get("Tags", ""),
                    "address": row.get("Address", "не указано")
                }
            )
            text_documents.append(doc)
        except Exception as e:
            print(f"Ошибка обработки строки {_}: {e}")
    
    return text_documents

def perform_web_search(question: str) -> str:
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(f"{question} Суздаль", max_results=Config.SEARCH_RESULTS):
                results.append(f"{r['title']}\n{r['href']}\n{r['body']}")
        return "\n\n".join(results) if results else "Не найдено информации в интернете"
    except Exception as e:
        print(f"Ошибка поиска: {e}")
        return "Не удалось выполнить поиск"

def update_dialog_context(user_id: str, role: str, message: str):
    if user_id not in DIALOG_CONTEXTS:
        DIALOG_CONTEXTS[user_id] = []
    
    DIALOG_CONTEXTS[user_id].append({"role": role, "message": message})
    
    if len(DIALOG_CONTEXTS[user_id]) > Config.MAX_CONTEXT_LENGTH:
        DIALOG_CONTEXTS[user_id] = DIALOG_CONTEXTS[user_id][-Config.MAX_CONTEXT_LENGTH:]

def get_dialog_context(user_id: str) -> str:
    if user_id not in DIALOG_CONTEXTS or not DIALOG_CONTEXTS[user_id]:
        return "Нет предыдущего контекста"
    
    return "\n".join(
        f"{item['role']}: {item['message']}" 
        for item in DIALOG_CONTEXTS[user_id]
    )

def refine_question(question: str, user_id: str) -> str:
    question_lower = question.lower().strip()
    if not question_lower:
        return "Пожалуйста, задайте ваш вопрос о Суздале."
    
    words = question_lower.split()
    if len(words) < Config.MIN_QUESTION_LENGTH:
        return (
            "Уточните, пожалуйста, ваш запрос. Например:\n"
            "- Какие музеи стоит посетить?\n"
            "- Где можно попробовать медовуху?\n"
            "- Какие достопримечательности находятся в центре?"
        )
    
    # Проверяем, был ли предыдущий уточняющий вопрос
    context = get_dialog_context(user_id)
    if "assistant: Уточните" in context or "assistant: Я могу порекомендовать" in context:
        update_dialog_context(user_id, "clarification", question)
        return None
    
    # Уточняем вопросы о еде
    if any(keyword in question_lower for keyword in Config.FOOD_KEYWORDS) and len(words) < 6:
        refinement = (
            "Я могу порекомендовать места по разным критериям:\n"
            "- По типу кухни (русская, итальянская, азиатская...)\n"
            "- По расположению (центр, рядом с кремлем...)\n"
            "- По бюджету (эконом, средний, премиум)\n"
            "- По атмосфере (уютное, семейное, романтическое...)\n\n"
            "Что для вас важнее при выборе места?"
        )
        update_dialog_context(user_id, "assistant", refinement)
        return refinement
    
    return None

def format_address(context_docs: list) -> str:
    if not context_docs:
        return ""
    
    main_doc = context_docs[0]
    address = main_doc.metadata.get("address", "не указано")
    
    if address.lower() in ["не указано", "нет информации", ""]:
        return ""
    return address

def is_answer_in_context(context: list) -> bool:
    if not context:
        return False
    
    for doc in context:
        if "не указано" not in doc.page_content and len(doc.page_content.strip()) > 20:
            return True
    return False

def prepare_prompt_input(question: str, context: list, web_search: str = "", dialog_context: str = "") -> dict:
    address_section = format_address(context)
    context_str = "\n\n".join([doc.page_content for doc in context]) if context else "Нет данных в базе"
    
    return {
        "context": context_str,
        "web_search": web_search,
        "question": question,
        "address_section": address_section if address_section else "адрес не указан",
        "dialog_context": dialog_context
    }

def add_feedback_request(response: str) -> str:
    return response + "\n\nБыл ли этот ответ полезен? Задайте дополнительные вопросы, если нужно!"

TOURISM_PROMPT_TEMPLATE = """
Я ваш виртуальный гид по Суздалю. Отвечаю на вопросы о достопримечательностях, питании и услугах.

[Предыдущий диалог]:
{dialog_context}

[Контекст из базы знаний]:
{context}

[Информация из интернета]:
{web_search}

[Ваш вопрос]:
{question}

Правила ответа:
1. Для музеев и достопримечательностей:
- Название и краткое описание
- Часы работы и стоимость (если есть)
- Интересные факты
- Как добраться

2. Для заведений питания:
- Название и тип кухни
- Средний чек (если известен)
- Адрес и особенности
- Рекомендации

3. Для других вопросов:
- Дайте развернутый ответ
- Используйте контекст и интернет
- Будьте дружелюбны и информативны

Ответ:
"""

tourism_prompt = PromptTemplate.from_template(TOURISM_PROMPT_TEMPLATE)

def handle_question(question: str, user_id: str) -> str:
    dialog_context = get_dialog_context(user_id)
    context = document_retriever.invoke(question)
    
    # Проверяем наличие уточнений
    clarification = None
    for msg in DIALOG_CONTEXTS.get(user_id, []):
        if msg["role"] == "clarification":
            clarification = msg["message"]
            break
    
    # Формируем уточненный вопрос при наличии
    search_question = question
    if clarification:
        search_question = f"{question} (уточнение: {clarification})"
        context = document_retriever.invoke(search_question)
    
    if is_answer_in_context(context):
        prompt_input = prepare_prompt_input(
            question=search_question,
            context=context,
            dialog_context=dialog_context
        )
        try:
            response = ai_assistant.invoke(tourism_prompt.format(**prompt_input))
        except Exception as e:
            print(f"Ошибка GigaChat: {e}")
            response = "Нашел информацию, но возникла ошибка при формировании ответа."
    else:
        web_results = perform_web_search(search_question)
        if "Не найдено" not in web_results:
            prompt_input = prepare_prompt_input(
                question=search_question,
                context=[],
                web_search=web_results,
                dialog_context=dialog_context
            )
            try:
                response = ai_assistant.invoke(tourism_prompt.format(**prompt_input))
            except Exception as e:
                print(f"Ошибка GigaChat: {e}")
                response = f"Вот что я нашел в интернете:\n{web_results}"
        else:
            response = "К сожалению, не удалось найти информацию. Попробуйте переформулировать вопрос."
    
    update_dialog_context(user_id, "assistant", response)
    return add_feedback_request(response)

def ask_question(question: str, user_id: str = "default") -> str:
    if not question.strip():
        return "Пожалуйста, задайте ваш вопрос о Суздале."
    
    update_dialog_context(user_id, "user", question)
    
    refinement = refine_question(question, user_id)
    if refinement:
        return refinement
    
    return handle_question(question, user_id)

# Инициализация компонентов
try:
    download_certificate()
    embedding_model, ai_assistant = initialize_models()
    text_documents = load_data()
    
    if not text_documents:
        raise Exception("Не удалось загрузить данные о достопримечательностях")
    
    vector_store = FAISS.from_documents(text_documents, embedding_model)
    document_retriever = vector_store.as_retriever(search_kwargs={"k": Config.RETRIEVER_K})
except Exception as e:
    print(f"Ошибка инициализации: {e}")
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
        response = ask_question(item.question, item.user_id)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки запроса: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
