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

load_dotenv()

GIGACHAT_AUTH = os.getenv("GIGACHAT_AUTH") 
CERT_PATH = os.getenv("CERT_PATH")
CERT_URL = os.getenv("CERT_URL")

if not Path(CERT_PATH).exists():
    try:
        print(f"Скачиваю сертификат с {CERT_URL}...")
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

csv_file_path = "https://raw.githubusercontent.com/vuyq/SuzdalAI/refs/heads/main/SuzdalAI/app/data/suzdal_attractions.csv"  
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
            "tags": row.get("Tags", "")
        }
    )
    for _, row in df.iterrows()
]

vector_store = FAISS.from_documents(text_documents, embedding_model)
document_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

tourism_prompt = PromptTemplate.from_template("""
Ты — профессиональный гид-аналитик. Отвечай на вопросы, используя ТОЛЬКО данные из предоставленной базы достопримечательностей.

Жёсткие правила:
1. Используй ТОЛЬКО информацию из контекста ниже
2. Если данных для ответа нет — честно говори "В моих данных нет этой информации"
3. Если вопрос слишком общий и не даёт точного направления поиска — задай уточняющий вопрос
4. Будь максимально конкретным и используй цифры/факты из данных
5. Формат ответа: маркированный список с ключевыми параметрами

Контекст (CSV): {context}
Структура ответа:
Если вопрос достаточно конкретный:
Вот места для посещения по вашему запросу:
- Название: 
- Тип: 
- Почему рекомендовано: 
- Основные особенности:
- Контакты/расположение: 
- Важная дополнительная информация: 

Если вопрос слишком общий:
- Уточни у пользователя, что именно его интересует (например, возраст детей, интерес к архитектуре, кухня и т.д.)

Вопрос: {question}
""")

rag_pipeline = (
    RunnablePassthrough.assign(context=lambda x: document_retriever.invoke(x["question"]))
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
    return rag_pipeline.invoke({"question": question})

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
