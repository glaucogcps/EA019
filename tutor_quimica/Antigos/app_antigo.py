"""
Interface do tutor.   Rodar com:  streamlit run app.py
"""
import json
from pathlib import Path

import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

from llm import MODELO, chat_ollama
from tutor import Exercicio, iniciar_sessao, turno
from verificador import VerificadorTransformer

try:
    from streamlit_ketcher import st_ketcher   # editor de estruturas
except ImportError:
    st_ketcher = None

BASE = Path(__file__).parent
st.set_page_config(page_title="Tutor de Química Orgânica", layout="wide")


@st.cache_resource
def carregar():
    dados = json.loads((BASE / "exercicios.json").read_text(encoding="utf-8"))
    exercicios = [Exercicio.de_dict(d) for d in dados]
    verificador = VerificadorTransformer(BASE / "trad_best.pt", BASE / "vocab_tokens.json")
    return exercicios, verificador


def desenhar(smiles, tamanho=(520, 200)):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Draw.MolToImage(mol, size=tamanho) if mol else None


for arquivo in ("trad_best.pt", "vocab_tokens.json", "quimica_tradutor.py"):
    if not (BASE / arquivo).exists():
        st.error(f"Falta o arquivo {arquivo} na pasta do projeto. "
                 "Baixe da pasta tradutor/v4 best do GitHub.")
        st.stop()

with st.spinner("Carregando o modelo químico..."):
    exercicios, verificador = carregar()

# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.header("Exercício")
    idx = st.selectbox("Escolha", range(len(exercicios)),
                       format_func=lambda i: exercicios[i].titulo)
    reiniciar = st.button("Reiniciar exercício")
    modo_professor = st.toggle("Modo professor")
    st.caption(f"Modelo de linguagem: {MODELO}")

ex = exercicios[idx]
if reiniciar or st.session_state.get("ex_id") != ex.id:
    st.session_state.ex_id = ex.id
    st.session_state.sessao = iniciar_sessao(ex, verificador)
    st.session_state.chat = []            # lista de (papel, texto, smiles)
    st.session_state.veredito = None
sessao = st.session_state.sessao

# ---------------------------------------------------------------- cabeçalho
st.title(ex.titulo)
st.write(ex.enunciado)
img = desenhar(ex.reagentes)
if img:
    st.image(img, caption="Reagentes")

col_chat, col_editor = st.columns([3, 2], gap="large")

# ---------------------------------------------------------------- editor
with col_editor:
    st.subheader("Sua resposta")
    if st_ketcher:
        smiles = st_ketcher(key=f"ketcher_{ex.id}")
        st.caption("Desenhe o produto e clique em Apply no editor.")
    else:
        smiles = st.text_input("SMILES do produto", key=f"smi_{ex.id}")
    if smiles:
        previa = desenhar(smiles, (300, 160))
        if previa:
            st.image(previa, caption="Estrutura a enviar")
    enviar_estrutura = st.button("Enviar estrutura", disabled=not smiles,
                                 type="primary", use_container_width=True)

pergunta = st.chat_input("Escreva sua dúvida ou explique seu raciocínio")


def processar(texto, smi):
    st.session_state.chat.append(("user", texto or "Esta é minha resposta.", smi))
    try:
        with st.spinner("O tutor está pensando..."):
            resposta, veredito = turno(sessao, texto, smi, chat_ollama)
    except Exception as erro:
        resposta = ("Não consegui falar com o modelo de linguagem. "
                    f"Confira se o Ollama está rodando. Detalhe: {erro}")
        veredito = {"status": "erro"}
    st.session_state.chat.append(("assistant", resposta, None))
    st.session_state.veredito = veredito


if enviar_estrutura:
    processar("", smiles)
elif pergunta:
    processar(pergunta, None)

# ---------------------------------------------------------------- conversa
with col_chat:
    if not st.session_state.chat:
        st.info("Desenhe o produto que você espera e envie, ou comece com uma pergunta.")
    for papel, texto, smi in st.session_state.chat:
        with st.chat_message(papel):
            st.write(texto)
            if smi:
                im = desenhar(smi, (260, 140))
                if im:
                    st.image(im)
    if sessao.resolvido:
        st.success("Exercício resolvido.")

# ---------------------------------------------------------------- depuração
if modo_professor:
    with st.sidebar:
        st.divider()
        st.subheader("Bastidores")
        pred = sessao.predicao_modelo
        st.write({
            "referência": sessao.referencia,
            "fonte": "gabarito" if ex.produto_esperado else "modelo",
            "confiança": sessao.confianca,
            "predição do modelo": pred.produto if pred else None,
            "confiança do modelo": pred.confianca if pred else None,
            "modelo discorda do gabarito": sessao.modelo_discorda,
            "tentativas": sessao.tentativas,
            "último veredito": st.session_state.veredito,
        })
