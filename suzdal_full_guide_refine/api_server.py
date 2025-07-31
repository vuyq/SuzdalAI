import os
import requests
import pandas as pd
import uvicorn
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
from typing import Dict, List
from uuid import uuid4

# Загрузка переменных окружения
load_dotenv()

# Конфигурация
class Config:
    MAX_RETRIES = 3
    MIN_QUESTION_LENGTH = 3
    SEARCH_RESULTS = 3
    RETRIEVER_K = 5
    CERT_PATH = os.getenv("CERT_PATH")
    CERT_URL = os.getenv("CERT_URL")
    GIGACHAT_AUTH = os.getenv("GIGACHAT_AUTH")
    CSV_DATA_URL = "https://raw.githubusercontent.com/vuyq/SuzdalAI/refs/heads/main/suzdal_full_guide_refine/attractions.csv"
    MAX_HISTORY_LENGTH = 40  # Максимальное количество запоминаемых сообщений

# Система хранения истории диалогов
class DialogHistory:
    def __init__(self):
        self.sessions: Dict[str, List[Dict]] = {}
    
    def create_session(self) -> str:
        session_id = str(uuid4())
        self.sessions[session_id] = []
        return session_id
    
    def add_message(self, session_id: str, role: str, content: str):
        if session_id in self.sessions:
            self.sessions[session_id].append({"role": role, "content": content})
            # Ограничиваем длину истории
            if len(self.sessions[session_id]) > Config.MAX_HISTORY_LENGTH * 2:
                self.sessions[session_id] = self.sessions[session_id][-Config.MAX_HISTORY_LENGTH * 2:]
    
    def get_history(self, session_id: str) -> List[Dict]:
        return self.sessions.get(session_id, [])

dialog_history = DialogHistory()

# Загрузка сертификата
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
            temperature=0,
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
        df = pd.read_csv(Config.CSV_DATA_URL, on_bad_lines='skip')
    
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

def perform_web_search(question: str) -> str:
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(f"{question} Суздаль site:ru", max_results=Config.SEARCH_RESULTS):
                results.append(f"{r['title']}\n{r['href']}\n{r['body']}")
        return "\n\n".join(results) if results else "Не найдено информации в интернете"
    except Exception as e:
        print(f"Ошибка поиска: {e}")
        return "Не удалось выполнить поиск"

def refine_question(question: str) -> str:
    question_lower = question.lower()
    words = question.strip().split()
    
    food_keywords = ["еда", "поесть", "кафе", "ресторан", "перекусить", "кухня"]
    if any(keyword in question_lower for keyword in food_keywords) and len(words) < 6:
        return (
            "Я могу порекомендовать места по разным критериям:\n"
            "- По типу кухни (русская, итальянская, азиатская...)\n"
            "- По расположению (центр, рядом с кремлем...)\n"
            "- По бюджету (эконом, средний, премиум)\n"
            "- По атмосфере (уютное, семейное, романтическое...)\n\n"
            "Что для вас важнее при выборе места?"
        )
    
    if len(words) < Config.MIN_QUESTION_LENGTH:
        return (
            "Уточните, пожалуйста, ваш запрос. Например:\n"
            "- Какие музеи стоит посетить с детьми?\n"
            "- Где можно попробовать традиционную суздальскую кухню?\n"
            "- Какие достопримечательности находятся в центре города?\n"
            "- Где недорого пообедать рядом с Торговыми рядами?"
        )
    
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
    content = "\n".join(doc.page_content for doc in context)
    return "не указано" not in content and len(content.strip()) > 50

def prepare_prompt_input(question: str, context: list, web_search: str = "", history: List[Dict] = None) -> dict:
    address_section = format_address(context)
    context_str = "\n\n".join([doc.page_content for doc in context]) if context else "Нет данных в базе"
    
    # Формируем историю диалога для контекста
    history_context = ""
    if history:
        history_context = "\n\nПредыдущие вопросы и ответы:\n"
        for msg in history[-Config.MAX_HISTORY_LENGTH:]:  # Берем только последние N сообщений
            prefix = "Вопрос: " if msg["role"] == "user" else "Ответ: "
            history_context += f"{prefix}{msg['content']}\n"
    
    return {
        "context": context_str,
        "web_search": web_search,
        "question": question,
        "address_section": address_section if address_section else "адрес не указан",
        "history": history_context
    }

def add_feedback_request(response: str) -> str:
    feedback_prompt = "\n\nБыл ли этот ответ полезен для вас? Если у вас есть дополнительные вопросы, не стесняйтесь задавать!"
    return response + feedback_prompt

TOURISM_PROMPT_TEMPLATE = """
Привет! Я ваш виртуальный гид по Суздалю. Я постараюсь дать максимально полезный и точный ответ.

[Контекст из базы знаний]:
{context}

[Информация из интернета]:
{web_search}

{history}

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
- Начните с "Вот что я нашел в интернете:"
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

def handle_food_question(question: str, session_id: str) -> str:
    context = document_retriever.invoke(question)
    
    recommendations = []
    for doc in context:
        if doc.metadata.get("type", "").lower() in ["кафе", "ресторан", "столовая"]:
            name = doc.metadata.get("title", "Заведение")
            desc = doc.page_content.split("\n")[0][:100] + "..."
            address = doc.metadata.get("address", "адрес не указан")
            recommendations.append(f"- {name}: {desc}\n  Адрес: {address}")
    
    if recommendations:
        response = (
            "Вот несколько вариантов где можно поесть в Суздале:\n\n"
            + "\n\n".join(recommendations[:5]) +
            "\n\nМогу уточнить рекомендации по конкретным критериям - просто скажите, что для вас важно!"
        )
    else:
        web_results = perform_web_search(question)
        if "Не найдено" not in web_results:
            response = f"Вот что я нашел в интернете:\n{web_results}"
        else:
            response = (
                "К сожалению, не нашел конкретных рекомендаций. "
                "Попробуйте уточнить:\n"
                "- Какую кухню предпочитаете?\n"
                "- В каком районе ищете заведение?\n"
                "- Какой уровень цен вас интересует?"
            )
    
    dialog_history.add_message(session_id, "assistant", response)
    return add_feedback_request(response)

def handle_general_question(question: str, session_id: str) -> str:
    context = document_retriever.invoke(question)
    history = dialog_history.get_history(session_id)
    
    if is_answer_in_context(context):
        prompt_input = prepare_prompt_input(question, context, history=history)
        response = ai_assistant.invoke(tourism_prompt.format(**prompt_input))
    else:
        web_results = perform_web_search(question)
        if "Не найдено" not in web_results:
            prompt_input = prepare_prompt_input(question, [], web_results, history=history)
            response = ai_assistant.invoke(tourism_prompt.format(**prompt_input))
        else:
            response = "К сожалению, не удалось найти информацию. Попробуйте переформулировать вопрос."
    
    dialog_history.add_message(session_id, "assistant", response)
    return add_feedback_request(response)

def ask_question(question: str, session_id: str) -> str:
    if not question.strip():
        return "Пожалуйста, задайте ваш вопрос о Суздале. Я постараюсь помочь!"

    dialog_history.add_message(session_id, "user", question)
    
    food_keywords = ["еда", "поесть", "кафе", "ресторан", "перекусить", "кухня"]
    is_food_question = any(keyword in question.lower() for keyword in food_keywords)
    
    refinement = refine_question(question)
    if refinement:
        dialog_history.add_message(session_id, "assistant", refinement)
        return refinement
    
    if is_food_question:
        return handle_food_question(question, session_id)
    
    return handle_general_question(question, session_id)

# Инициализация компонентов
download_certificate()
embedding_model, ai_assistant = initialize_models()
text_documents = load_data()
vector_store = FAISS.from_documents(text_documents, embedding_model)
document_retriever = vector_store.as_retriever(search_kwargs={"k": Config.RETRIEVER_K})

# FastAPI приложение
app = FastAPI(title="Суздаль Tourism Assistant", 
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
    session_id: str = None  # Идентификатор сессии для поддержания контекста

@app.post("/ask")
async def ask(item: Question):
    try:
        # Создаем новую сессию, если не передана
        if not item.session_id:
            item.session_id = dialog_history.create_session()
        
        response = ask_question(item.question, item.session_id)
        return {"answer": response, "session_id": item.session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
