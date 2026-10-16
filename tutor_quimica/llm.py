"""
Conexão com o Llama via Ollama (servidor local).

Para trocar de modelo:  TUTOR_LLM=llama3.1:70b streamlit run app.py
Para outro servidor:    OLLAMA_URL=http://maquina:11434 streamlit run app.py
"""
import json
import os
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODELO = os.environ.get("TUTOR_LLM", "llama3.1:8b")


def chat_ollama(mensagens: list, temperatura: float = 0.3) -> str:
    corpo = json.dumps({
        "model": MODELO,
        "messages": mensagens,
        "stream": False,
        "options": {"temperature": temperatura},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat", data=corpo,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["message"]["content"]
