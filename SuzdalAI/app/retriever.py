import pandas as pd
from pathlib import Path
import requests
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_gigachat import GigaChatEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# Параметры
CSV_URL = "https://raw.githubusercontent.com/vuyq/SuzdalAI/main/suzdal_attractions_verified_17.csv"
LOCAL_CSV_PATH = "app/data/suzdal_attractions.csv"

def download_csv_if_needed(url=CSV_URL, local_path=LOCAL_CSV_PATH):
    """Скачиваем CSV с GitHub, если файла ещё нет"""
    local_file = Path(local_path)
    if not local_file.exists():
        local_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"Скачиваю CSV с {url} ...")
        response = requests.get(url)
        response.raise_for_status()
        local_file.write_bytes(response.content)
        print("Файл успешно скачан и сохранён в:", local_path)
    else:
        print("Локальный CSV уже существует, пропускаем загрузку.")

def load_docs(file_path=LOCAL_CSV_PATH):
    """Загрузка CSV в список документов"""
    df = pd.read_csv(file_path, sep=';')
    docs = [
        Document(
            page_content="\n".join(f"{col}: {val if pd.notna(val) else 'не указано'}" for col, val in row.items()),
            metadata={"title": row.get("Name", ""), "type": row.get("Type", "")}
        )
        for _, row in df.iterrows()
    ]
    print(f"Загружено документов: {len(docs)}")
    return docs

def create_vector_store(docs, token, cert_path):
    """Создание векторного хранилища"""
    embeddings = GigaChatEmbeddings(
        access_token=token,
        model="Embeddings",
        verify_ssl_certs=True,
        ca_bundle_file=cert_path
    )
    print("Инициализация векторного хранилища...")
    return FAISS.from_documents(docs, embeddings)

if __name__ == "__main__":
    # Пример использования
    download_csv_if_needed()
    token = os.getenv("GIGACHAT_TOKEN")        # Задай в .env или получи другим способом
    cert_path = os.getenv("GIGACHAT_CERT")     # Задай путь к сертификату
    docs = load_docs()
    if token and cert_path:
        vector_store = create_vector_store(docs, token, cert_path)
        print("Векторное хранилище готово!")
    else:
        print("⚠ Не указан токен или сертификат в переменных окружения!")
