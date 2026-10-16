"""
Interface do tutor.   Rodar com:  streamlit run app.py

Dois modos (escolha na barra lateral):
  * Exercícios:   reações curadas do exercicios.json, com gabarito.
  * Reação livre: o aluno digita os reagentes, o modelo do GitHub calcula o produto.
"""
import json
from pathlib import Path

import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

from llm import MODELO, chat_ollama
from tutor import Exercicio, canonicalizar, exercicio_livre, iniciar_sessao, turno
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


def nova_sessao(ex):
    st.session_state.ex_id = ex.id
    with st.spinner("Calculando o produto com o modelo..."):
        st.session_state.sessao = iniciar_sessao(ex, verificador)
    st.session_state.chat = []            # lista de (papel, texto, smiles)
    st.session_state.veredito = None
    st.session_state.revelado = False
    st.session_state.ultimo_enviado = None


for arquivo in ("trad_best.pt", "vocab_tokens.json", "quimica_tradutor.py"):
    if not (BASE / arquivo).exists():
        st.error(f"Falta o arquivo {arquivo} na pasta do projeto. "
                 "Baixe da pasta tradutor/v4 best do GitHub.")
        st.stop()

with st.spinner("Carregando o modelo químico..."):
    exercicios, verificador = carregar()

# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    modo = st.radio("Modo", ["Exercícios", "Reação livre"], horizontal=True)
    st.divider()

    if modo == "Exercícios":
        st.header("Exercício")
        idx = st.selectbox("Escolha", range(len(exercicios)),
                           format_func=lambda i: exercicios[i].titulo)
        ex = exercicios[idx]
        if st.button("Reiniciar exercício") or st.session_state.get("ex_id") != ex.id:
            nova_sessao(ex)

    else:
        st.header("Reação livre")
        st.caption("Digite os reagentes em SMILES, separados por ponto. "
                   "Inclua catalisadores e solventes: o modelo aprendeu com eles.")
        entrada = st.text_input("Reagentes", placeholder="ex: CC(=O)Cl.Oc1ccccc1")
        if entrada:
            previa = desenhar(entrada, (300, 160))
            if previa:
                st.image(previa)
        if st.button("Começar", type="primary", disabled=not entrada):
            reag = canonicalizar(entrada)
            if reag is None:
                st.error("SMILES inválido. Confira valências, parênteses e anéis.")
            else:
                estranhos = verificador.tokens_desconhecidos(reag)
                if estranhos:
                    st.warning("O modelo nunca viu estes pedaços no treino e vai "
                               f"ignorá-los: {', '.join(estranhos)}. A previsão pode sair ruim.")
                nova_sessao(exercicio_livre(reag))
        if not str(st.session_state.get("ex_id", "")).startswith("livre:"):
            st.info("Digite os reagentes e clique em Começar.")
            st.stop()
        ex = st.session_state.sessao.exercicio

    st.divider()
    modo_professor = st.toggle("Modo professor")
    st.caption(f"Modelo de linguagem: {MODELO}")

sessao = st.session_state.sessao
livre = ex.produto_esperado is None and ex.id.startswith("livre:")

# ---------------------------------------------------------------- cabeçalho
st.title(ex.titulo)
st.write("Qual é o produto principal desta reação?" if livre else ex.enunciado)
img = desenhar(ex.reagentes)
if img:
    st.image(img, caption="Reagentes")

col_chat, col_editor = st.columns([3, 2], gap="large")

# ---------------------------------------------------------------- editor
with col_editor:
    st.subheader("Sua resposta")
    smiles_desenho = ""
    if st_ketcher:
        smiles_desenho = st_ketcher(key=f"ketcher_{ex.id}") or ""
        st.caption("Desenhe o produto e clique em Apply no editor.")
    with st.expander("Ou digite o SMILES", expanded=not st_ketcher):
        smiles_digitado = st.text_input("SMILES do produto", key=f"smi_{ex.id}").strip()
    # O que foi digitado tem prioridade sobre o desenho
    smiles = smiles_digitado or smiles_desenho.strip()

    repetido = bool(smiles) and smiles == st.session_state.get("ultimo_enviado")
    if smiles:
        previa = desenhar(smiles, (300, 160))
        if previa:
            st.image(previa, caption="Estrutura a enviar")
        else:
            st.warning("Esta estrutura é quimicamente impossível. Você pode enviar "
                       "mesmo assim para ver como o tutor reage.")
        st.code(smiles, language=None)
    if repetido:
        st.info("Esta estrutura já foi enviada. Para tentar outra, altere o desenho e "
                "clique em Apply. Se você alterou e ela não mudou aqui, o editor não "
                "conseguiu ler o desenho: procure átomos marcados em vermelho (valência errada).")
    enviar_estrutura = st.button("Enviar estrutura", disabled=not smiles or repetido,
                                 type="primary", use_container_width=True)

    # Na reação livre não existe gabarito: o aluno pode pedir para ver o que o modelo calculou
    if livre:
        st.divider()
        pred = sessao.predicao_modelo
        if not st.session_state.revelado:
            if st.button("Ver produto calculado pelo modelo", use_container_width=True):
                st.session_state.revelado = True
                sessao.resolvido = True          # a partir daqui o tutor discute livremente
                st.rerun()
        elif pred is None or pred.produto is None:
            st.error("O modelo não conseguiu prever um produto para estes reagentes.")
        else:
            st.markdown("**Previsão do modelo**")
            st.image(desenhar(pred.produto, (300, 160)), caption=pred.produto)
            if pred.alternativas:
                with st.expander("Outras possibilidades"):
                    for alt in pred.alternativas:
                        st.image(desenhar(alt, (260, 130)), caption=alt)
            st.caption("O modelo acerta cerca de metade das reações e erra mais as "
                       "reações clássicas de livro. Confira pelo mecanismo.")

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
    st.session_state.ultimo_enviado = smiles
    processar("", smiles)
    st.rerun()          # redesenha a tela já sabendo que esta estrutura foi enviada
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
    if sessao.resolvido and not livre:
        st.success("Exercício resolvido.")

# ---------------------------------------------------------------- depuração
if modo_professor:
    with st.sidebar:
        st.divider()
        st.subheader("Bastidores")
        pred = sessao.predicao_modelo
        st.write({
            "reagentes": ex.reagentes,
            "referência": sessao.referencia,
            "fonte": "gabarito" if ex.produto_esperado else "modelo",
            "confiança": sessao.confianca,
            "predição do modelo": pred.produto if pred else None,
            "alternativas": pred.alternativas if pred else None,
            "modelo discorda do gabarito": sessao.modelo_discorda,
            "tentativas": sessao.tentativas,
            "último veredito": st.session_state.veredito,
        })
