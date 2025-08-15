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
from uuid import uuid4
from typing import Dict, List, Optional

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
    MAX_HISTORY_LENGTH = 40

class DialogManager:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
    
    def create_session(self) -> str:
        session_id = str(uuid4())
        self.sessions[session_id] = {
            "history": [],
            "awaiting_clarification": False,
            "clarification_type": None,
            "previous_question": None
        }
        return session_id
    
    def add_message(self, session_id: str, role: str, content: str):
        if session_id in self.sessions:
            self.sessions[session_id]["history"].append({"role": role, "content": content})
            # Ограничиваем длину истории
            self.sessions[session_id]["history"] = self.sessions[session_id]["history"][-Config.MAX_HISTORY_LENGTH * 2:]
    
    def get_history(self, session_id: str) -> List[Dict]:
        return self.sessions.get(session_id, {}).get("history", [])
    
    def set_clarification_state(self, session_id: str, state: bool, clarification_type: Optional[str] = None):
        if session_id in self.sessions:
            self.sessions[session_id]["awaiting_clarification"] = state
            self.sessions[session_id]["clarification_type"] = clarification_type if state else None
    
    def is_awaiting_clarification(self, session_id: str) -> bool:
        return self.sessions.get(session_id, {}).get("awaiting_clarification", False)
    
    def get_clarification_type(self, session_id: str) -> Optional[str]:
        return self.sessions.get(session_id, {}).get("clarification_type")
    
    def set_previous_question(self, session_id: str, question: str):
        if session_id in self.sessions:
            self.sessions[session_id]["previous_question"] = question
    
    def get_previous_question(self, session_id: str) -> Optional[str]:
        return self.sessions.get(session_id, {}).get("previous_question")

dialog_manager = DialogManager()

def download_certificate():
    if Config.CERT_PATH and Config.CERT_URL and not Path(Config.CERT_PATH).exists():
        try:
            response = requests.get(Config.CERT_URL)
            response.raise_for_status()
            os.makedirs(os.path.dirname(Config.CERT_PATH), exist_ok=True)
            with open(Config.CERT_PATH, "wb") as f:
                f.write(response.content)
        except Exception as e:
            raise Exception(f"Не удалось скачать сертификат: {str(e)}")

def get_gigachat_token():
    if not Config.GIGACHAT_AUTH:
        raise Exception("GIGACHAT_AUTH не установлен в переменных окружения")
    
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': str(uuid4()),
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
        raise Exception(f"Ошибка инициализации моделей: {str(e)}")

def load_data():
    try:
        df = pd.read_csv(Config.CSV_DATA_URL, sep=';')
    except Exception as e:
        try:
            df = pd.read_csv(Config.CSV_DATA_URL)
        except Exception as e:
            raise Exception("Не удалось загрузить данные")
    
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
                "address": row.get("Address", "не указан")
            }
        )
        for _, row in df.iterrows()
    ]

def perform_web_search(question: str) -> str:
    try:
        with DDGS() as ddgs:
            results = [
                f"{r['title']}\n{r['href']}\n{r['body']}"
                for r in ddgs.text(f"{question} Суздаль site:ru", max_results=Config.SEARCH_RESULTS)
            ]
        return "\n\n".join(results) if results else "Не найдено информации в интернете"
    except Exception:
        return "Не удалось выполнить поиск"

def refine_question(question: str, session_id: str) -> Optional[str]:
    question_lower = question.lower()
    
    if dialog_manager.is_awaiting_clarification(session_id):
        return None
    
    food_keywords = ["еда", "поесть", "кафе", "ресторан", "перекусить", "кухня"]
    if any(keyword in question_lower for keyword in food_keywords) and len(question.strip().split()) < 6:
        dialog_manager.set_clarification_state(session_id, True, "food_preferences")
        return (
            "Я могу порекомендовать места по разным критериям:\n"
            "1. По типу кухни (русская, итальянская, азиатская...)\n"
            "2. По расположению (центр, рядом с кремлем...)\n"
            "3. По бюджету (эконом, средний, премиум)\n"
            "4. По атмосфере (уютное, семейное, романтическое...)\n\n"
            "Что для вас важнее при выборе места? Можете указать несколько критериев."
        )
    
    if len(question.strip().split()) < Config.MIN_QUESTION_LENGTH:
        dialog_manager.set_clarification_state(session_id, True, "general_question")
        return (
            "Уточните, пожалуйста, ваш запрос. Например:\n"
            "1. Какие музеи стоит посетить с детьми?\n"
            "2. Где можно попробовать традиционную суздальскую кухню?\n"
            "3. Какие достопримечательности находятся в центре города?\n"
            "4. Где недорого пообедать рядом с Торговыми рядами?\n\n"
            "Какой вариант вам ближе или у вас другой вопрос?"
        )
    
    return None

def format_address(context_docs: list) -> str:
    if not context_docs:
        return ""
    
    address = context_docs[0].metadata.get("address", "не указан")
    return "" if address.lower() in ["не указан", "нет информации", ""] else address

def is_answer_in_context(context: list) -> bool:
    if not context:
        return False
    content = "\n".join(doc.page_content for doc in context)
    return "не указано" not in content and len(content.strip()) > 50

def prepare_prompt_input(question: str, context: list, web_search: str = "", history: List[Dict] = None) -> dict:
    address_section = format_address(context)
    context_str = "\n\n".join(doc.page_content for doc in context) if context else "Нет данных в базе"
    
    history_context = ""
    if history:
        history_context = "\n\nПредыдущие вопросы и ответы:\n" + "\n".join(
            f"{'Вопрос' if msg['role'] == 'user' else 'Ответ'}: {msg['content']}"
            for msg in history[-Config.MAX_HISTORY_LENGTH:]
        )
    
    return {
        "context": context_str,
        "web_search": web_search,
        "question": question,
        "address_section": address_section or "адрес не указан",
        "history": history_context
    }

def add_feedback_request(response: str) -> str:
    return response + "\n\nБыл ли этот ответ полезен для вас? Если у вас есть дополнительные вопросы, не стесняйтесь задавать!"

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

def handle_food_question(question: str, session_id: str, clarification: Optional[str] = None) -> str:
    final_question = f"{question} {clarification}" if clarification else question
    
    if "кухня" not in final_question.lower():
        final_question += " кухня"
    
    try:
        context = document_retriever.invoke(final_question)
    except Exception:
        return "Произошла ошибка при поиске информации о местах питания. Пожалуйста, попробуйте позже."
    
    recommendations = []
    for doc in context:
        if doc.metadata.get("type", "").lower() in ["кафе", "ресторан", "столовая"]:
            name = doc.metadata.get("title", "Заведение")
            desc = doc.page_content.split("\n")[0][:100] + "..."
            address = doc.metadata.get("address", "адрес не указан")
            tags = doc.metadata.get("tags", "").lower()
            
            if clarification:
                if "итальянск" in clarification.lower() and "итальянск" not in tags:
                    continue
                if "центр" in clarification.lower() and "центр" not in address.lower():
                    continue
            
            recommendations.append(f"- {name}: {desc}\n  Адрес: {address}")
    
    if recommendations:
        response = (
            f"С учетом ваших предпочтений ({clarification}), вот подходящие варианты:\n\n"
            if clarification else
            "Вот несколько мест где можно поесть в Суздале:\n\n"
        ) + "\n\n".join(recommendations[:5]) + (
            "\n\nЕсли хотите уточнить критерии или узнать больше - просто спросите!"
        )
    else:
        web_results = perform_web_search(final_question)
        response = (
            f"В базе нет подходящих вариантов, но вот что я нашел в интернете:\n{web_results}"
            if "Не найдено" not in web_results else
            "К сожалению, не нашел подходящих вариантов по вашему запросу.\n"
            "Можете уточнить:\n"
            "- Точное название заведения\n"
            "- Другие критерии поиска\n"
            "- Интересующую вас кухню или тип заведения"
        )
    
    dialog_manager.add_message(session_id, "assistant", response)
    dialog_manager.set_clarification_state(session_id, False)
    return add_feedback_request(response)

def handle_general_question(question: str, session_id: str, clarification: Optional[str] = None) -> str:
    final_question = f"{question} {clarification}" if clarification else question
    
    try:
        context = document_retriever.invoke(final_question)
    except Exception:
        return "Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже."
    
    history = dialog_manager.get_history(session_id)
    
    if is_answer_in_context(context):
        prompt_input = prepare_prompt_input(final_question, context, history=history)
        try:
            response = ai_assistant.invoke(tourism_prompt.format(**prompt_input))
        except Exception:
            response = "Произошла ошибка при генерации ответа. Пожалуйста, попробуйте позже."
    else:
        web_results = perform_web_search(final_question)
        if "Не найдено" not in web_results:
            prompt_input = prepare_prompt_input(final_question, [], web_results, history=history)
            try:
                response = ai_assistant.invoke(tourism_prompt.format(**prompt_input))
            except Exception:
                response = "Произошла ошибка при генерации ответа. Пожалуйста, попробуйте позже."
        else:
            response = "К сожалению, не удалось найти информацию. Попробуйте переформулировать вопрос."
    
    dialog_manager.add_message(session_id, "assistant", response)
    dialog_manager.set_clarification_state(session_id, False)
    return add_feedback_request(response)

def ask_question(question: str, session_id: str) -> str:
    if not question.strip():
        return "Пожалуйста, задайте ваш вопрос о Суздале. Я постараюсь помочь!"

    dialog_manager.add_message(session_id, "user", question)
    
    if dialog_manager.is_awaiting_clarification(session_id):
        prev_question = dialog_manager.get_previous_question(session_id)
        clarification_type = dialog_manager.get_clarification_type(session_id)
        
        if clarification_type == "food_preferences":
            dialog_manager.set_previous_question(session_id, prev_question)
            return handle_food_question(prev_question, session_id, clarification=question)
        elif clarification_type == "general_question":
            dialog_manager.set_previous_question(session_id, prev_question)
            return handle_general_question(prev_question, session_id, clarification=question)
    
    refinement = refine_question(question, session_id)
    if refinement:
        dialog_manager.add_message(session_id, "assistant", refinement)
        dialog_manager.set_previous_question(session_id, question)
        return refinement
    
    food_keywords = ["еда", "поесть", "кафе", "ресторан", "перекусить", "кухня"]
    if any(keyword in question.lower() for keyword in food_keywords):
        return handle_food_question(question, session_id)
    
    return handle_general_question(question, session_id)

# Инициализация компонентов
try:
    download_certificate()
    embedding_model, ai_assistant = initialize_models()
    text_documents = load_data()
    vector_store = FAISS.from_documents(text_documents, embedding_model)
    document_retriever = vector_store.as_retriever(search_kwargs={"k": Config.RETRIEVER_K})
except Exception as e:
    print(f"Ошибка инициализации: {str(e)}")
    raise

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
    session_id: Optional[str] = None

@app.post("/ask")
async def ask(item: Question):
    try:
        if not item.session_id:
            item.session_id = dialog_manager.create_session()
        
        response = ask_question(item.question, item.session_id)
        return {
            "answer": response,
            "session_id": item.session_id
        }
            
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
