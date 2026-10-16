# -*- coding: utf-8 -*-
"""
quimica_tradutor.py
Inferência do Transformer químico (EA019/tradutor/v4 best).
Código copiado do transformer_v4.ipynb, só a parte de inferência.
Precisa de trad_best.pt e vocab_tokens.json na mesma pasta.
"""
import json
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------- tokenização
SMI_REGEX = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)
PAD, SOS, EOS, UNK = "<PAD>", "<SOS>", "<EOS>", "<UNK>"


def tokenizar(smiles: str):
    toks = SMI_REGEX.findall(smiles)
    assert "".join(toks) == smiles, f"Tokenização perdeu caracteres: {smiles}"
    return toks


def carregar_vocab(caminho: str):
    with open(caminho, encoding="utf-8") as f:
        tok2id = json.load(f)
    return tok2id, {v: k for k, v in tok2id.items()}


def codificar(smiles: str, tok2id: dict, max_len: int = 300):
    return [tok2id.get(t, tok2id[UNK]) for t in tokenizar(smiles)][:max_len]


# ---------------------------------------------------------------- modelo
class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, ff_dim=1024, num_heads=8,
                 num_layers=4, dropout=0.1, max_len=300):
        super().__init__()
        self.vocab_size, self.embed_dim = vocab_size, embed_dim
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos = PositionalEncoding(embed_dim, max_len + 2)
        self.drop = nn.Dropout(dropout)
        enc = nn.TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout,
                                         batch_first=True, norm_first=True)
        dec = nn.TransformerDecoderLayer(embed_dim, num_heads, ff_dim, dropout,
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers, norm=nn.LayerNorm(embed_dim),
                                             enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(dec, num_layers, norm=nn.LayerNorm(embed_dim))
        self.out = nn.Linear(embed_dim, vocab_size)
        self.out.weight = self.emb.weight

    def _embed(self, ids):
        return self.drop(self.pos(self.emb(ids) * math.sqrt(self.embed_dim)))

    def encode(self, src):
        src_pad = src == 0
        return self.encoder(self._embed(src), src_key_padding_mask=src_pad), src_pad

    def decode(self, dec_in, mem, src_pad):
        T = dec_in.size(1)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=dec_in.device), 1)
        h = self.decoder(self._embed(dec_in), mem, tgt_mask=causal,
                         tgt_key_padding_mask=(dec_in == 0),
                         memory_key_padding_mask=src_pad)
        return self.out(h)


# ---------------------------------------------------------------- gramática SMILES (SBS)
_BONDS = set("=#-/\\:~")


def _classe(tok):
    if tok in (PAD, SOS, UNK):
        return 7
    if tok == EOS:
        return 6
    if tok == "(":
        return 1
    if tok == ")":
        return 2
    if tok == ".":
        return 5
    if tok in _BONDS:
        return 4
    if tok.isdigit() or tok.startswith("%"):
        return 3
    if tok[0] == "[" or tok in ("B", "Br", "C", "Cl", "N", "O", "S", "P", "F", "I",
                                "b", "c", "n", "o", "s", "p", "*"):
        return 0
    return 7


class GramaticaSMILES:
    def __init__(self, tok2id):
        self.classe = torch.tensor([_classe(t) for t, _ in sorted(tok2id.items(), key=lambda x: x[1])])
        self.anel_num = torch.zeros(len(tok2id), dtype=torch.long)
        for t, i in tok2id.items():
            if _classe(t) == 3:
                self.anel_num[i] = int(t[1:]) if t.startswith("%") else int(t)
        self.tabela = torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 0],   # após átomo
            [1, 0, 0, 0, 1, 0, 0, 0],   # após '('
            [1, 1, 1, 1, 1, 1, 1, 0],   # após ')'
            [1, 1, 1, 1, 1, 1, 1, 0],   # após dígito de anel
            [1, 0, 0, 1, 0, 0, 0, 0],   # após ligação
            [1, 0, 0, 0, 0, 0, 0, 0],   # após '.'
            [0, 0, 0, 0, 0, 0, 0, 0],   # após EOS
            [1, 0, 0, 0, 0, 0, 0, 0],   # início
        ], dtype=torch.bool)

    def estado_inicial(self):
        return (0, 0, 7)

    def avancar(self, estado, tok_id):
        depth, rings, _ = estado
        c = int(self.classe[tok_id])
        if c == 1:
            depth += 1
        elif c == 2:
            depth -= 1
        elif c == 3:
            rings ^= 1 << int(self.anel_num[tok_id])
        return (depth, rings, c)

    def mascara(self, estados, device):
        ult = torch.tensor([e[2] for e in estados])
        m = self.tabela[ult][:, self.classe]
        depth = torch.tensor([e[0] for e in estados])
        rings = torch.tensor([e[1] for e in estados])
        fechado = (depth == 0) & (rings == 0)
        m[:, self.classe == 2] &= (depth > 0).unsqueeze(1)
        m[:, (self.classe == 5) | (self.classe == 6)] &= fechado.unsqueeze(1)
        return m.to(device)


# ---------------------------------------------------------------- preditor
class PreditorDeReacoes:
    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tok2id, self.id2tok = carregar_vocab(vocab_path)
        ck = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg = ck["config"]
        self.max_len = cfg.get("max_len", 300)
        self.model = Seq2SeqTransformer(
            len(self.tok2id), cfg.get("embed_dim", 256), cfg.get("ff_dim", 1024),
            cfg.get("num_heads", 8), cfg.get("num_layers", 4),
            cfg.get("dropout", 0.1), self.max_len,
        ).to(self.device)
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.eval()
        self.gramatica = GramaticaSMILES(self.tok2id)

    @torch.no_grad()
    def prever(self, smiles_reagentes: str, beam_width: int = 5, n_best: int = 3,
               usar_sbs: bool = True, alpha: float = 0.6):
        """Devolve os n_best produtos mais prováveis (strings SMILES, sem canonicalizar)."""
        sos, eos = self.tok2id[SOS], self.tok2id[EOS]
        src = torch.tensor([codificar(smiles_reagentes, self.tok2id, self.max_len)], device=self.device)
        mem, src_pad = self.model.encode(src)
        mem = mem.expand(beam_width, -1, -1)
        src_pad = src_pad.expand(beam_width, -1)

        seqs = torch.full((1, 1), sos, dtype=torch.long, device=self.device)
        scores = torch.zeros(1, device=self.device)
        gram = self.gramatica if usar_sbs else None
        estados = [gram.estado_inicial()] if gram else None
        terminadas = []

        for _ in range(self.max_len):
            k = seqs.size(0)
            logits = self.model.decode(seqs, mem[:k], src_pad[:k])[:, -1]
            logp = F.log_softmax(logits.float(), -1)
            logp[:, 0] = -1e9
            logp[:, sos] = -1e9
            if gram is not None:
                logp = logp.masked_fill(~gram.mascara(estados, logp.device), -1e9)
            cand = (scores.unsqueeze(1) + logp).view(-1)
            top = cand.topk(min(beam_width * 2, cand.numel()))
            novas, novos_scores, novos_estados = [], [], []
            for s, idx in zip(top.values.tolist(), top.indices.tolist()):
                if s < -1e8:
                    continue
                b, tok = divmod(idx, logp.size(1))
                if tok == eos:
                    L = seqs.size(1)
                    terminadas.append((s / (((5 + L) / 6) ** alpha), seqs[b, 1:].tolist()))
                else:
                    novas.append(torch.cat([seqs[b], torch.tensor([tok], device=self.device)]))
                    novos_scores.append(s)
                    if gram is not None:
                        novos_estados.append(gram.avancar(estados[b], tok))
                if len(novas) == beam_width:
                    break
            if not novas:
                break
            seqs = torch.stack(novas)
            scores = torch.tensor(novos_scores, device=self.device)
            estados = novos_estados if gram is not None else None
            if len(terminadas) >= beam_width:
                melhor_viva = scores.max().item() / (((5 + self.max_len) / 6) ** alpha)
                if melhor_viva < max(terminadas)[0]:
                    break

        if not terminadas:
            terminadas = [(s / (((5 + seqs.size(1)) / 6) ** alpha), seq[1:].tolist())
                          for s, seq in zip(scores.tolist(), seqs)]
        terminadas.sort(key=lambda x: -x[0])
        return ["".join(self.id2tok[i] for i in toks) for _, toks in terminadas[:n_best]]
