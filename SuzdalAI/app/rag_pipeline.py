import os
import uuid
import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_core.prompts import PromptTemplate
from langchain_gigachat import GigaChat
from app.retriever import load_docs, create_vector_store
from app.utils import refine_question

# Загружаем переменные окружения
load_dotenv()

GIGACHAT_AUTH = os.getenv("GIGACHAT_AUTH")
CERT_PATH = os.getenv("CERT_PATH")

@retry(stop=stop_after_attempt(3), wait=wait_exponential())
def get_token():
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': str(uuid.uuid4()),
        'Authorization': f'Basic {GIGACHAT_AUTH}'
    }
    payload = {'scope': 'GIGACHAT_API_PERS'}
    r = requests.post(url, headers=headers, data=payload, verify=CERT_PATH)
    r.raise_for_status()
    return r.json().get("access_token")


# Получаем токен
token = get_token()

# Загружаем документы и создаём векторное хранилище
docs = load_docs()
print(f"Загружено документов: {len(docs)}")
vector_store = create_vector_store(docs, token, CERT_PATH)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Инициализация GigaChat
ai = GigaChat(
    access_token=token,
    model="GigaChat-2",
    temperature=0,
    verify_ssl_certs=True,
    ca_bundle_file=CERT_PATH
)

# Новый промт по твоему описанию
prompt = PromptTemplate.from_template("""
Ты — профессиональный гид-аналитик. Отвечай на вопросы, используя ТОЛЬКО данные из предоставленной базы достопримечательностей.

Жёсткие правила:
1. Используй ТОЛЬКО информацию из контекста ниже
2. Если данных для ответа нет — честно говори "В моих данных нет этой информации"
3. Если вопрос слишком общий и не даёт точного направления поиска — задай уточняющий вопрос
4. Будь максимально конкретным и используй цифры/факты из данных
5. Формат ответа: маркированный список с ключевыми параметрами

Контекст (CSV): {context}

Вопрос: {question}

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

Твой ответ (только факты из данных!):
""")

def rag_search(query):
    # Уточняем вопрос, если он слишком общий
    refined = refine_question(query)
    # Получаем релевантный контекст
    context = retriever.invoke(refined)
    # Формируем финальный промпт
    final_prompt = prompt.format(context=context, question=refined)
    # Вызываем модель
    response = ai.invoke(final_prompt)
    # response — это просто строка (или объект с content)
    if hasattr(response, 'content'):
        return response.content
    return response


if __name__ == "__main__":
    while True:
        user_query = input("Введите ваш вопрос (или 'exit' для выхода): ")
        if user_query.lower() == 'exit':
            break
        try:
            answer = rag_search(user_query)
            print("\nОтвет:")
            print(answer)
        except Exception as e:
            print("Произошла ошибка:", e)
