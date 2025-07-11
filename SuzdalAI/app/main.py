from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.utils import get_token
import requests

app = FastAPI()

class Question(BaseModel):
    query: str

@app.get("/")
def root():
    return {"message": "Добро пожаловать в GigaChat Proxy"}

@app.post("/ask")
def ask(question: Question):
    try:
        token = get_token()
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        data = {
            "model": "GigaChat-2",
            "messages": [
                {"role": "user", "content": question.query}
            ]
        }
        r = requests.post("https://gigachat.devices.sberbank.ru/api/v1/chat/completions",
                          json=data, headers=headers, verify=True)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
