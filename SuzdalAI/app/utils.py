import os
from dotenv import load_dotenv
import requests
import uuid

load_dotenv()

GIGACHAT_AUTH = os.getenv("GIGACHAT_AUTH")
CERT_PATH = os.getenv("CERT_PATH")

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
