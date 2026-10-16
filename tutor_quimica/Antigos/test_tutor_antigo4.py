"""Testes da lógica do tutor sem precisar do Llama nem do modelo.   Rodar com:  python test_tutor.py"""
import json
from pathlib import Path

from tutor import Exercicio, canonicalizar, iniciar_sessao, turno, viola_saida
from verificador import CONFIANCA_V4, Predicao

BASE = Path(__file__).parent
exercicios = {d["id"]: Exercicio.de_dict(d)
              for d in json.loads((BASE / "exercicios.json").read_text(encoding="utf-8"))}


class VerificadorFalso:
    """Imita o modelo do GitHub: sempre confiança 0,7, às vezes errado."""
    previsoes = {"C=CC.Br": "CCCBr", "C=CC.O.O=S(=O)(O)O": "CC(C)O"}

    def prever(self, reagentes):
        p = self.previsoes.get(reagentes)
        return Predicao(canonicalizar(p), CONFIANCA_V4 if p else 0.0, [])


verificador = VerificadorFalso()
llm_falso = lambda msgs: "Boa tentativa. O que acontece primeiro com o alceno?"


def status(ex_id, smiles, texto=""):
    s = iniciar_sessao(exercicios[ex_id], verificador)
    _, v = turno(s, texto, smiles, llm_falso)
    return v["status"], s


# todos os gabaritos são SMILES válidos
for ex in exercicios.values():
    assert canonicalizar(ex.reagentes) is not None, ex.id
    if ex.produto_esperado:
        assert canonicalizar(ex.produto_esperado) is not None, ex.id

# com gabarito: veredito firme, mesmo com o modelo errando
assert status("hbr_propeno", "BrC(C)C")[0] == "correto"           # outra grafia do mesmo produto
assert status("hbr_propeno", "CCCBr")[0] == "incorreto"           # anti-Markovnikov
assert status("hbr_propeno", "CCCBr")[1].modelo_discorda is True
assert status("hbr_propeno", "CC(C)(C)(C)CBr")[0] == "invalido"   # carbono com 5 ligações
assert status("hbr_propeno", "C=CC")[0] == "igual_reagente"
assert status("hbr_propeno", None, "não entendi")[0] == "sem_estrutura"

# sem gabarito: o modelo v4 só sugere, nunca afirma
assert status("hidratacao_sem_gabarito", "CC(C)O")[0] == "provavelmente_correto"
assert status("hidratacao_sem_gabarito", "CCCO")[0] == "provavelmente_incorreto"

# depois de acertar, o tutor pode falar livremente
_, s = status("fischer", "CCOC(C)=O")
assert s.resolvido
_, v = turno(s, "por que precisa do ácido?", None, llm_falso)
assert v["status"] == "resolvido"

# filtro de saída
assert viola_saida("O produto é CC(C)Br, certo?")
assert viola_saida("Você chegou no 2-bromopropano?", ["2-bromopropano"])
assert not viola_saida("Qual carbocátion é mais estável (primário ou secundário)?")

# SMILES entre parênteses também é bloqueado
assert viola_saida("um grupo funcional cloro (CC(=O)Cl). O ácido")

# na reação livre, o Llama recebe os reagentes em português, nunca em SMILES
from tutor import exercicio_livre, montar_mensagens
s = iniciar_sessao(exercicio_livre("CC(=O)Cl.Oc1ccccc1"), verificador)
prompt = montar_mensagens(s, {"status": "sem_estrutura"}, "me ajude")[0]["content"]
assert "cloreto de acetila" in prompt and "fenol" in prompt
assert "CC(=O)Cl" not in prompt

# o Llama fica sabendo qual estrutura o aluno enviou
capturado = {}
def llm_espiao(msgs):
    capturado["msgs"] = msgs
    return "Interessante. O que acontece com o oxigênio do fenol?"
s = iniciar_sessao(exercicio_livre("CC(=O)Cl.Oc1ccccc1"), verificador)
turno(s, "", "C1=CC=CC1", llm_espiao)
assert "ENVIOU uma estrutura" in capturado["msgs"][0]["content"]
assert "ciclopentadieno" in capturado["msgs"][-1]["content"]

# se o Llama insistir em vazar, cai na resposta segura
s = iniciar_sessao(exercicios["hbr_propeno"], verificador)
resposta, _ = turno(s, "", "CCCBr", lambda msgs: "Na verdade é CC(C)Br.")
assert "CC(C)Br" not in resposta

print("Todos os testes passaram.")
