"""
Ponte entre o chat e o modelo químico do GitHub (tradutor v4 best).

O chat só pergunta uma coisa: "dados estes reagentes, qual produto o modelo prevê?".
O v4 não mede incerteza, então a confiança é fixa em 0,7: o tutor trata a
previsão do modelo como "provável", nunca como certeza. Nos exercícios com
gabarito, o gabarito manda e o modelo nem é usado para julgar.
"""
from dataclasses import dataclass, field
from typing import Optional

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

CONFIANCA_V4 = 0.7


def _canon(smiles: Optional[str]) -> Optional[str]:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else None


@dataclass
class Predicao:
    produto: Optional[str]                         # SMILES canônico, ou None
    confianca: float                               # 0 a 1
    alternativas: list = field(default_factory=list)


class VerificadorTransformer:
    """Chama o modelo do GitHub ao vivo."""

    def __init__(self, pesos="trad_best.pt", vocab="vocab_tokens.json"):
        from quimica_tradutor import PreditorDeReacoes
        self.p = PreditorDeReacoes(str(pesos), str(vocab))

    def prever(self, reagentes: str) -> Predicao:
        try:
            brutos = self.p.prever(reagentes, n_best=3)
        except Exception:
            return Predicao(None, 0.0, [])           # SMILES que o tokenizador não aceita
        candidatos = [c for c in dict.fromkeys(_canon(b) for b in brutos) if c]  # tira repetidos
        if not candidatos:
            return Predicao(None, 0.0, [])
        return Predicao(candidatos[0], CONFIANCA_V4, candidatos[1:])
