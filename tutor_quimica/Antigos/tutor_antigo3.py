"""
Lógica do tutor socrático.

Quem decide se a química está certa é este código (RDKit + verificador).
O Llama só transforma o veredito em uma conversa. Ele nunca decide nada.
"""
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

from verificador import Predicao

RDLogger.DisableLog("rdApp.*")

# Com o v4 a confiança é sempre 0,7 (ver verificador.py), que fica entre os dois
# limiares: o tutor trata a previsão do modelo como "provável".
LIMIAR_ALTO = 0.9
LIMIAR_BAIXO = 0.5


# ---------------------------------------------------------------- química

def canonicalizar(smiles: Optional[str]) -> Optional[str]:
    if not smiles or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    return Chem.MolToSmiles(mol) if mol else None


def problemas_quimicos(smiles: str) -> list:
    mol = Chem.MolFromSmiles(smiles.strip(), sanitize=False)
    if mol is None:
        return ["o SMILES não pôde ser lido (erro de sintaxe)"]
    problemas = [p.Message() for p in Chem.DetectChemistryProblems(mol)]
    return problemas or ["estrutura rejeitada pelo RDKit"]


# ---------------------------------------------------------------- descrição dos reagentes
# O Llama não sabe ler SMILES. Então o RDKit descreve cada reagente em português
# e é essa descrição que vai para o prompt.

NOMES_COMUNS = {
    "CC(=O)Cl": "cloreto de acetila", "Oc1ccccc1": "fenol", "CCO": "etanol",
    "CO": "metanol", "CC(=O)O": "ácido acético", "CC(=O)OC(C)=O": "anidrido acético",
    "O=S(=O)(O)O": "ácido sulfúrico", "Br": "brometo de hidrogênio (HBr)",
    "Cl": "cloreto de hidrogênio (HCl)", "O": "água", "BrBr": "bromo (Br2)",
    "ClCl": "cloro (Cl2)", "C=C": "eteno", "C=CC": "propeno", "C=CC=C": "1,3-butadieno",
    "c1ccccc1": "benzeno", "C1=CCC=C1": "ciclopentadieno",
    "C1=CCCCC1": "cicloexeno", "CC(=O)Oc1ccccc1": "acetato de fenila", "Cc1ccccc1": "tolueno", "Nc1ccccc1": "anilina",
    "ClCCl": "diclorometano (solvente)", "CCN(CC)CC": "trietilamina (base)",
    "C1CCOC1": "tetra-hidrofurano, THF (solvente)", "C1COCCO1": "1,4-dioxano (solvente)",
    "CC(C)=O": "acetona", "OC1CCCCC1": "cicloexanol", "O=C1CCCCC1": "cicloexanona",
    "O=C1CCC(=O)N1Br": "N-bromossuccinimida (NBS)",
    "c1ccc(P(c2ccccc2)c2ccccc2)cc1": "trifenilfosfina",
    "c1cc[nH+]cc1": "íon piridínio (parte do PCC)",
    "O=[Cr](=O)([O-])Cl": "clorocromato (parte do PCC)",
    "[Na+]": "íon sódio", "[K+]": "íon potássio", "[OH-]": "hidróxido",
    "O=C([O-])[O-]": "carbonato", "OB(O)c1ccccc1": "ácido fenilborônico",
}
NOMES_COMUNS = {Chem.MolToSmiles(Chem.MolFromSmiles(k)): v for k, v in NOMES_COMUNS.items()}

GRUPOS = [  # (nome, SMARTS). A ordem importa: os mais específicos primeiro.
    ("cloreto de acila", "[CX3](=O)[Cl,Br]"),
    ("anidrido", "[CX3](=O)O[CX3](=O)"),
    ("carbamato (ex.: grupo protetor Boc)", "[NX3][CX3](=O)[OX2][#6]"),
    ("ácido carboxílico", "[CX3](=O)[OX2H1]"),
    ("éster", "[CX3;!$(C[NX3])](=O)[OX2][#6]"),
    ("amida", "[CX3;!$(C(=O)O)](=O)[NX3]"),
    ("aldeído", "[CX3H1](=O)[#6]"),
    ("cetona", "[#6][CX3](=O)[#6]"),
    ("fenol (OH ligado a anel aromático)", "[OX2H][c]"),
    ("álcool", "[OX2H][CX4]"),
    ("amina", "[NX3;H2,H1;!$(NC=O);!$(N-a)]"),
    ("amina aromática", "[NX3;H2,H1;!$(NC=O)]-a"),
    ("nitrila", "C#N"),
    ("alceno (ligação dupla C=C)", "C=C"),
    ("alquino (ligação tripla C≡C)", "C#C"),
    ("haleto de alquila", "[CX4][Cl,Br,I]"),
    ("haleto de arila", "c[Cl,Br,I]"),
    ("éter", "[OD2]([#6;!$(C=O)])[#6;!$(C=O)]"),
    ("anel aromático", "a1aaaaa1"),
]
GRUPOS = [(n, Chem.MolFromSmarts(s)) for n, s in GRUPOS]


def descrever_molecula(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "estrutura não reconhecida"
    canon = Chem.MolToSmiles(mol)
    grupos = [n for n, padrao in GRUPOS if mol.HasSubstructMatch(padrao)]
    nome = NOMES_COMUNS.get(canon)
    if nome and grupos:
        return f"{nome} (grupos funcionais: {', '.join(grupos)})"
    if nome:
        return nome
    formula = rdMolDescriptors.CalcMolFormula(mol)
    return f"composto {formula}" + (f" com {', '.join(grupos)}" if grupos else "")


def descrever_reagentes(reagentes: str) -> str:
    """Lista em português das espécies presentes, uma por linha."""
    return "\n".join(f"  {i}. {descrever_molecula(r)}"
                      for i, r in enumerate(reagentes.split("."), 1))


# ---------------------------------------------------------------- dados

@dataclass
class Exercicio:
    id: str
    titulo: str
    enunciado: str
    reagentes: str                      # SMILES completo, com catalisadores e solventes
    produto_esperado: Optional[str]     # gabarito curado; None se não houver
    produto_nome: str = ""              # nunca aparece para o aluno antes do acerto
    nota_mecanismo: str = ""            # orientação do professor para o Llama

    @classmethod
    def de_dict(cls, d: dict) -> "Exercicio":
        return cls(
            id=d["id"],
            titulo=d["titulo"],
            enunciado=d["enunciado"],
            reagentes=d["reagentes"],
            produto_esperado=d.get("produto_esperado"),
            produto_nome=d.get("produto_nome", ""),
            nota_mecanismo=d.get("nota_mecanismo", ""),
        )


def exercicio_livre(reagentes: str) -> Exercicio:
    """Reação digitada pelo aluno: sem gabarito, quem julga é o modelo."""
    return Exercicio(
        id=f"livre:{reagentes}",
        titulo="Reação livre",
        enunciado="O aluno propôs a própria reação e quer descobrir o produto principal.",
        reagentes=reagentes,
        produto_esperado=None,
    )


@dataclass
class Sessao:
    exercicio: Exercicio
    predicao_modelo: Optional[Predicao] = None
    referencia: Optional[str] = None
    confianca: float = 0.0
    confiavel: bool = False
    modelo_discorda: bool = False
    tentativas: int = 0
    resolvido: bool = False
    historico: list = field(default_factory=list)


def iniciar_sessao(ex: Exercicio, verificador) -> Sessao:
    s = Sessao(ex)
    pred = verificador.prever(ex.reagentes)
    s.predicao_modelo = pred
    if ex.produto_esperado:
        # Gabarito do professor manda. O modelo fica só registrado para avaliação.
        s.referencia = canonicalizar(ex.produto_esperado)
        s.confianca = 1.0
        s.confiavel = True
        s.modelo_discorda = pred.produto is not None and pred.produto != s.referencia
    else:
        s.referencia = pred.produto
        s.confianca = pred.confianca
        s.confiavel = pred.produto is not None and pred.confianca >= LIMIAR_ALTO
    return s


# ---------------------------------------------------------------- veredito

def avaliar(s: Sessao, smiles_aluno: Optional[str]) -> dict:
    if not smiles_aluno or not smiles_aluno.strip():
        return {"status": "resolvido" if s.resolvido else "sem_estrutura"}

    c = canonicalizar(smiles_aluno)
    if c is None:
        return {"status": "invalido", "problemas": problemas_quimicos(smiles_aluno),
                "descricao": "uma estrutura quimicamente impossível"}

    reagentes = {canonicalizar(r) for r in s.exercicio.reagentes.split(".")}
    if c in reagentes:
        return {"status": "igual_reagente", "smiles": c, "descricao": descrever_molecula(c)}

    if s.referencia is None or (not s.confiavel and s.confianca < LIMIAR_BAIXO):
        return {"status": "incerto", "smiles": c, "descricao": descrever_molecula(c)}

    if c == s.referencia:
        status = "correto" if s.confiavel else "provavelmente_correto"
    else:
        status = "incorreto" if s.confiavel else "provavelmente_incorreto"
    return {"status": status, "smiles": c, "descricao": descrever_molecula(c)}


# ---------------------------------------------------------------- prompt

SISTEMA_BASE = (
    "Você é um tutor socrático de Química Orgânica para alunos de graduação. "
    "Responda sempre em português do Brasil, com no máximo 4 frases, e termine "
    "com uma única pergunta para o aluno. Nunca escreva SMILES nem estruturas "
    "em texto; refira-se a moléculas por nome ou descrição. Os FATOS abaixo vêm "
    "de um verificador químico e são verdadeiros: nunca os contradiga e nunca "
    "diga se a resposta do aluno está certa ou errada além do que os FATOS dizem. "
    "Fale só das espécies listadas nos FATOS."
)

POLITICA = {
    "invalido": (
        "A estrutura enviada é quimicamente impossível. Explique em termos gerais "
        "o tipo de problema (por exemplo, valência de um átomo) e peça que o aluno "
        "revise o desenho. Não mostre nem descreva a estrutura correta."
    ),
    "igual_reagente": (
        "O aluno enviou uma molécula idêntica a um dos reagentes, ou seja, nada reagiu. "
        "Pergunte o que ele espera que aconteça com esse reagente nas condições dadas."
    ),
    "correto": (
        "A resposta está correta. Confirme e peça que o aluno justifique pelo mecanismo: "
        "quem é o nucleófilo, quem é o eletrófilo e qual etapa define o produto."
    ),
    "incorreto": (
        "A resposta está errada. NÃO revele nem descreva o produto correto. Faça uma "
        "única pergunta que leve o aluno a reexaminar a etapa do mecanismo onde "
        "provavelmente errou."
    ),
    "provavelmente_correto": (
        "O verificador considera provável que a resposta esteja correta, mas não tem "
        "certeza. Não confirme ainda: peça antes a justificativa pelo mecanismo."
    ),
    "provavelmente_incorreto": (
        "O verificador considera provável que a resposta esteja errada, mas não tem "
        "certeza. Não afirme que está errada; questione o mecanismo para o aluno "
        "testar a própria hipótese."
    ),
    "incerto": (
        "O verificador NÃO sabe com confiança qual é o produto desta reação. Não diga "
        "se a resposta está certa ou errada. Diga com honestidade que vão decidir pelo "
        "raciocínio mecanístico e conduza essa discussão passo a passo."
    ),
    "sem_estrutura": (
        "O aluno escreveu sem enviar estrutura. Responda à dúvida conceitual sem revelar "
        "o produto. Se ele citar um produto só em texto, não avalie: peça que desenhe a "
        "estrutura no editor e envie."
    ),
    "resolvido": (
        "O aluno já acertou este exercício. Pode discutir livremente o mecanismo e "
        "falar do produto pelo nome."
    ),
}


def montar_mensagens(s: Sessao, veredito: dict, texto_aluno: str) -> list:
    ex = s.exercicio
    fatos = [
        f"Exercício: {ex.enunciado}",
        "Espécies presentes na reação (lista COMPLETA; não existe nenhuma outra, "
        "não invente reagentes):\n" + descrever_reagentes(ex.reagentes),
        f"Veredito do verificador: {veredito['status']}",
        f"Tentativas de estrutura até agora: {s.tentativas}",
    ]
    if "descricao" in veredito:
        fatos.append("O aluno ENVIOU uma estrutura nesta mensagem: " + veredito["descricao"])
    if "problemas" in veredito:
        fatos.append("Problemas detectados na estrutura: " + "; ".join(veredito["problemas"]))
    if ex.nota_mecanismo:
        fatos.append(f"Nota do professor sobre o mecanismo (use para guiar, não copie): {ex.nota_mecanismo}")
    if s.resolvido and ex.produto_nome:
        fatos.append(f"Produto (o aluno já acertou): {ex.produto_nome}")

    sistema = (
        f"{SISTEMA_BASE}\n\nFATOS:\n- " + "\n- ".join(fatos)
        + f"\n\nINSTRUÇÃO PARA ESTA RESPOSTA: {POLITICA[veredito['status']]}"
    )
    return [{"role": "system", "content": sistema}, *s.historico,
            {"role": "user", "content": texto_aluno}]


# ---------------------------------------------------------------- filtro de saída

_PARECE_SMILES = re.compile(r"[A-Za-z0-9@+\-\[\]()=#/\\%.]{5,}")
RESPOSTA_SEGURA = (
    "Vamos com calma e olhar o mecanismo passo a passo. "
    "Nos reagentes, qual espécie você acha que ataca primeiro, e onde?"
)


def viola_saida(texto: str, termos_proibidos=()) -> bool:
    """True se o Llama escreveu uma estrutura ou entregou o nome do produto."""
    for tok in _PARECE_SMILES.findall(texto):
        tok = tok.strip(".")   # ponto final da frase não faz parte da estrutura
        while tok.startswith("(") and tok.endswith(")"):   # ex: "(CC(=O)Cl)"
            tok = tok[1:-1]
        if any(ch in tok for ch in "()=#[]") and Chem.MolFromSmiles(tok) is not None:
            return True
    baixo = texto.lower()
    return any(t and t.lower() in baixo for t in termos_proibidos)


# ---------------------------------------------------------------- turno

def turno(s: Sessao, texto_aluno: str, smiles_aluno: Optional[str],
          llm: Callable[[list], str], max_regeneracoes: int = 2):
    """Um turno completo. Devolve (resposta_do_tutor, veredito)."""
    texto_aluno = (texto_aluno or "").strip() or "Esta é minha resposta."
    if smiles_aluno and smiles_aluno.strip():
        s.tentativas += 1

    veredito = avaliar(s, smiles_aluno)
    if veredito["status"] == "correto":
        s.resolvido = True

    if "descricao" in veredito:
        texto_aluno = f"{texto_aluno} [Estrutura que desenhei: {veredito['descricao']}]"
    mensagens = montar_mensagens(s, veredito, texto_aluno)
    proibidos = [] if s.resolvido else [s.exercicio.produto_nome]

    resposta = RESPOSTA_SEGURA
    for _ in range(max_regeneracoes + 1):
        candidata = llm(mensagens)
        if not viola_saida(candidata, proibidos):
            resposta = candidata
            break

    s.historico += [{"role": "user", "content": texto_aluno},
                    {"role": "assistant", "content": resposta}]
    return resposta, veredito
