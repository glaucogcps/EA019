# -*- coding: utf-8 -*-
"""
TRADUTOR REAGENTES -> PRODUTO (SMILES)  —  versão corrigida (v2)
================================================================
Reescrita do transformer_v1.ipynb com as correções discutidas:

  1. Tokenização por REGEX (Schwaller et al.): 'Cl', 'Br', '[C@@H]', '[Na+]' viram
     UM token. O dicionário EMBEDDING_LOOKUP da v1 nunca era usado (o código lia
     vocab_oficial.json char-a-char) — aqui o vocabulário é construído dos tokens.
  2. Máscara de PAD em TODAS as atenções (encoder self-attn, decoder self-attn,
     cross-attn). Na v1 o encoder atendia a ~400 posições de PAD por sequência e a
     máscara da cross-attention (encoder_output.sum(-1)==0) era sempre falsa.
  3. Loss = cross-entropy padrão + label smoothing 0.1, ignore_index=PAD.
     Os pesos 20x/15x/5x da v1 distorciam o objetivo (o modelo era punido 4x mais
     por errar 'C' do que por errar um fechamento de anel '1' ou um parêntese —
     exatamente os tokens que tornam um SMILES válido).
  4. Padding dinâmico por lote (pad até o maior da batch, não até 512) -> ~10-20x
     menos custo de atenção -> batches maiores (64-128) -> treino mais estável.
  5. Otimização "Noam" (warmup + decaimento 1/sqrt(t)), lr pico ~5e-4, AdamW.
     Na v1: lr = 1e-5 fixo — a loss ainda caía linearmente na época 15 (sub-treino).
  6. Encoder NÃO congelado. Pesos do autoencoder podem ser usados como inicialização
     (opcional), mas o encoder inteiro é treinado na tarefa de tradução.
  7. Beam search correto: hipóteses terminadas guardadas à parte, normalização por
     comprimento (na v1 uma hipótese curta com <EOS> sempre vencia -> saídas
     truncadas / cópia do primeiro reagente) e decodificação incremental
     (encoder roda 1 vez; decoder recebe só os tokens já gerados).
  8. Avaliação correta: acurácia top-1 por SMILES CANÔNICO (RDKit) em um conjunto
     de validação que NÃO está no treino + taxa de SMILES válidos + acurácia por
     token. A "acurácia mascarada" com teacher forcing (os 70%) não mede tradução.
  9. Limpeza dos dados: canonicalização, remoção de mapeamento atômico, produto =
     maior fragmento (remove contra-íons/sais), dedup, remoção de linhas em que o
     produto já é um dos reagentes.

Uso (Kaggle/Colab):
    python transformer_v2_corrigido.py            # treina com CONFIG abaixo
ou importe as funções e chame treinar_traducao(...) / traduzir(...).
"""

import os, re, json, math, random, logging, time
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Device: {device}")

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    "dataset_path": "dados_reacoes.csv",     # colunas: input_reagentes, output_produtos
    "vocab_path": "vocab_tokens.json",
    "n_amostras": 400000,                    # None = usa tudo (~1,35 M). Mais dados = melhor.
    "max_len": 300,                          # em TOKENS (regex), não caracteres
    "embed_dim": 256,
    "ff_dim": 1024,
    "num_heads": 8,
    "num_layers": 4,
    "dropout": 0.1,
    "label_smoothing": 0.1,
    "batch_size": 64,
    "lr_peak": 5e-4,
    "warmup_steps": 4000,
    "epochs": 30,
    "patience": 5,
    "seed": 42,
    "autoencoder_init": None,                # ex.: "molgpt_ep03.pt" (só inicializa, NÃO congela)
    "produto_maior_fragmento": True,         # remove sais / contra-íons do alvo
    "beam_width": 5,
    "n_eval_traducao": 500,                  # nº de pares de validação para top-1 canônica
}

# ============================================================
# 1. TOKENIZAÇÃO (regex de Schwaller et al., Molecular Transformer)
# ============================================================
SMI_REGEX = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)

def tokenizar(smiles: str):
    toks = SMI_REGEX.findall(smiles)
    assert ''.join(toks) == smiles, f"Tokenização perdeu caracteres: {smiles}"
    return toks

PAD, SOS, EOS, UNK = '<PAD>', '<SOS>', '<EOS>', '<UNK>'

def construir_vocab(lista_smiles, path):
    cnt = Counter()
    for s in lista_smiles:
        cnt.update(tokenizar(s))
    tokens = [PAD, SOS, EOS, UNK] + sorted(cnt, key=lambda t: (-cnt[t], t))
    tok2id = {t: i for i, t in enumerate(tokens)}
    with open(path, 'w') as f:
        json.dump(tok2id, f, indent=1)
    logger.info(f"Vocabulário: {len(tok2id)} tokens (regex). Salvo em {path}")
    return tok2id

def carregar_vocab(path):
    with open(path) as f:
        tok2id = json.load(f)
    return tok2id, {v: k for k, v in tok2id.items()}

def codificar(smiles, tok2id, max_len):
    ids = [tok2id.get(t, tok2id[UNK]) for t in tokenizar(smiles)]
    return ids[:max_len]

# ============================================================
# 2. DADOS — limpeza com RDKit
# ============================================================
def _canon(smi, maior_fragmento=False):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)                      # remove :n do atom mapping
    if maior_fragmento:
        frags = Chem.GetMolFrags(m, asMols=True)
        m = max(frags, key=lambda x: x.GetNumHeavyAtoms())
    return Chem.MolToSmiles(m)

def carregar_pares(path, n_amostras=None, seed=42, maior_fragmento=True):
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    df = pd.read_csv(path, usecols=['input_reagentes', 'output_produtos'])
    df = df.dropna()
    if n_amostras and n_amostras < len(df):
        df = df.sample(n_amostras, random_state=seed)
    logger.info(f"Lidas {len(df)} linhas. Canonicalizando com RDKit...")
    reag, prod = [], []
    descartadas = 0
    for r, p in zip(df['input_reagentes'].astype(str), df['output_produtos'].astype(str)):
        rc = _canon(r)
        pc = _canon(p, maior_fragmento)
        if rc is None or pc is None:
            descartadas += 1
            continue
        if pc in rc.split('.'):                 # "reação" em que o produto já era reagente
            descartadas += 1
            continue
        reag.append(rc); prod.append(pc)
    pares = list(dict.fromkeys(zip(reag, prod)))   # dedup mantendo ordem
    random.Random(seed).shuffle(pares)
    logger.info(f"Pares válidos: {len(pares)} | descartadas: {descartadas} | duplicatas: {len(reag)-len(pares)}")
    return [p[0] for p in pares], [p[1] for p in pares]

class ReacaoDataset(Dataset):
    def __init__(self, reag, prod, tok2id, max_len):
        self.src = [codificar(r, tok2id, max_len) for r in reag]
        self.tgt = [codificar(p, tok2id, max_len - 1) for p in prod]
        self.sos, self.eos = tok2id[SOS], tok2id[EOS]
    def __len__(self):
        return len(self.src)
    def __getitem__(self, i):
        return self.src[i], [self.sos] + self.tgt[i], self.tgt[i] + [self.eos]

def collate(batch):
    """Padding dinâmico: só até o maior da batch."""
    src, dec_in, tgt = zip(*batch)
    def pad(seqs):
        L = max(len(s) for s in seqs)
        out = torch.zeros(len(seqs), L, dtype=torch.long)     # PAD = 0
        for i, s in enumerate(seqs):
            out[i, :len(s)] = torch.tensor(s)
        return out
    return pad(src), pad(dec_in), pad(tgt)

# ============================================================
# 3. MODELO
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, ff_dim=1024, num_heads=8,
                 num_layers=4, dropout=0.1, max_len=300):
        super().__init__()
        self.vocab_size, self.embed_dim = vocab_size, embed_dim
        # Embedding COMPARTILHADO entre encoder, decoder e camada de saída
        # (mesmo vocabulário dos dois lados; menos parâmetros, converge melhor)
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos = PositionalEncoding(embed_dim, max_len + 2)
        self.drop = nn.Dropout(dropout)
        enc_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout,
                                               batch_first=True, norm_first=True)
        dec_layer = nn.TransformerDecoderLayer(embed_dim, num_heads, ff_dim, dropout,
                                               batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers, norm=nn.LayerNorm(embed_dim))
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers, norm=nn.LayerNorm(embed_dim))
        self.out = nn.Linear(embed_dim, vocab_size)
        self.out.weight = self.emb.weight          # weight tying
        nn.init.normal_(self.emb.weight, mean=0.0, std=embed_dim ** -0.5)
        with torch.no_grad():
            self.emb.weight[0].zero_()

    def _embed(self, ids):
        return self.drop(self.pos(self.emb(ids) * math.sqrt(self.embed_dim)))

    def encode(self, src):
        src_pad = (src == 0)                       # (B, S) True onde é PAD
        mem = self.encoder(self._embed(src), src_key_padding_mask=src_pad)
        return mem, src_pad

    def decode(self, dec_in, mem, src_pad):
        T = dec_in.size(1)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=dec_in.device), 1)
        h = self.decoder(self._embed(dec_in), mem,
                         tgt_mask=causal,
                         tgt_key_padding_mask=(dec_in == 0),
                         memory_key_padding_mask=src_pad)
        return self.out(h)

    def forward(self, src, dec_in):
        mem, src_pad = self.encode(src)
        return self.decode(dec_in, mem, src_pad)

def build_model(vocab_size, cfg):
    m = Seq2SeqTransformer(vocab_size, cfg["embed_dim"], cfg["ff_dim"], cfg["num_heads"],
                           cfg["num_layers"], cfg["dropout"], cfg["max_len"]).to(device)
    logger.info(f"Modelo: {sum(p.numel() for p in m.parameters()):,} parâmetros")
    return m

# ============================================================
# 4. LOSS / MÉTRICAS / OTIMIZADOR
# ============================================================
def loss_fn(logits, tgt, label_smoothing):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                           ignore_index=0, label_smoothing=label_smoothing)

@torch.no_grad()
def acuracia_token(logits, tgt):
    mask = tgt != 0
    return ((logits.argmax(-1) == tgt) & mask).sum().item() / max(mask.sum().item(), 1)

def noam_lambda(d_model, warmup, lr_peak):
    # lr(t) = lr_peak * min(t/warmup, sqrt(warmup/t))
    def f(step):
        step = max(step, 1)
        return min(step / warmup, math.sqrt(warmup / step))
    return f

# ============================================================
# 5. INFERÊNCIA — beam search correto
# ============================================================
@torch.no_grad()
def traduzir(model, smiles_reagentes, tok2id, id2tok, max_len=300, beam_width=5,
             alpha=0.6, n_best=1):
    """
    Beam search com:
      - encoder executado UMA vez;
      - decoder recebe só os tokens já gerados (sem PAD até max_len);
      - hipóteses terminadas guardadas à parte e comparadas com
        normalização por comprimento ((5+L)/6)^alpha (Wu et al., GNMT).
    Retorna lista de n_best SMILES (strings), do melhor para o pior.
    """
    model.eval()
    sos, eos = tok2id[SOS], tok2id[EOS]
    src = torch.tensor([codificar(smiles_reagentes, tok2id, max_len)], device=device)
    mem, src_pad = model.encode(src)
    mem = mem.expand(beam_width, -1, -1)
    src_pad = src_pad.expand(beam_width, -1)

    seqs = torch.full((1, 1), sos, dtype=torch.long, device=device)
    scores = torch.zeros(1, device=device)
    terminadas = []                              # (score_normalizado, tokens)

    for t in range(max_len):
        k = seqs.size(0)
        logits = model.decode(seqs, mem[:k], src_pad[:k])[:, -1]
        logp = F.log_softmax(logits, -1)
        logp[:, 0] = -1e9                        # nunca gerar PAD
        logp[:, sos] = -1e9                      # nem SOS
        cand = (scores.unsqueeze(1) + logp).view(-1)
        top = cand.topk(min(beam_width * 2, cand.numel()))
        novas_seqs, novos_scores = [], []
        for s, idx in zip(top.values.tolist(), top.indices.tolist()):
            b, tok = divmod(idx, logp.size(1))
            if tok == eos:
                L = seqs.size(1)                 # nº de tokens gerados (sem SOS)
                terminadas.append((s / (((5 + L) / 6) ** alpha), seqs[b, 1:].tolist()))
            else:
                novas_seqs.append(torch.cat([seqs[b], torch.tensor([tok], device=device)]))
                novos_scores.append(s)
            if len(novas_seqs) == beam_width:
                break
        if not novas_seqs:
            break
        seqs = torch.stack(novas_seqs)
        scores = torch.tensor(novos_scores, device=device)
        # parada: já temos beam_width terminadas e a melhor viva não pode superar a melhor terminada
        if len(terminadas) >= beam_width:
            melhor_viva = scores.max().item() / (((5 + max_len) / 6) ** alpha)
            if melhor_viva < max(terminadas)[0]:
                break
    if not terminadas:                            # nenhuma terminou: usa as vivas
        terminadas = [(s / (((5 + seqs.size(1)) / 6) ** alpha), seq[1:].tolist())
                      for s, seq in zip(scores.tolist(), seqs)]
    terminadas.sort(key=lambda x: -x[0])
    return [''.join(id2tok[i] for i in toks) for _, toks in terminadas[:n_best]]

def avaliar_traducao(model, reag, prod, tok2id, id2tok, cfg, n=500, mostrar=5):
    """Top-1 por SMILES canônico + taxa de validade. reag/prod devem ser de VALIDAÇÃO."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    acertos = validos = 0
    n = min(n, len(reag))
    for i in range(n):
        pred = traduzir(model, reag[i], tok2id, id2tok, cfg["max_len"], cfg["beam_width"])[0]
        m = Chem.MolFromSmiles(pred)
        ok = False
        if m is not None:
            validos += 1
            ok = Chem.MolToSmiles(m) == prod[i]  # prod já está canônico
        acertos += ok
        if i < mostrar:
            logger.info(f"  [{'OK ' if ok else 'ERR'}] reag: {reag[i][:90]}")
            logger.info(f"        prod: {prod[i][:90]}")
            logger.info(f"        pred: {pred[:90]}")
    logger.info(f"Top-1 canônica: {acertos}/{n} = {100*acertos/n:.1f}% | SMILES válidos: {100*validos/n:.1f}%")
    return acertos / n, validos / n

# ============================================================
# 6. TREINO
# ============================================================
def treinar_traducao(cfg=CONFIG):
    torch.manual_seed(cfg["seed"]); random.seed(cfg["seed"]); np.random.seed(cfg["seed"])

    reag, prod = carregar_pares(cfg["dataset_path"], cfg["n_amostras"], cfg["seed"],
                                cfg["produto_maior_fragmento"])
    n_val = max(int(len(reag) * 0.05), 1000) if len(reag) > 2000 else max(len(reag) // 10, 1)
    reag_tr, reag_val = reag[n_val:], reag[:n_val]
    prod_tr, prod_val = prod[n_val:], prod[:n_val]

    tok2id = construir_vocab(reag_tr + prod_tr, cfg["vocab_path"])
    id2tok = {v: k for k, v in tok2id.items()}

    ds_tr = ReacaoDataset(reag_tr, prod_tr, tok2id, cfg["max_len"])
    ds_val = ReacaoDataset(reag_val, prod_val, tok2id, cfg["max_len"])
    dl_tr = DataLoader(ds_tr, cfg["batch_size"], shuffle=True, collate_fn=collate,
                       num_workers=2, pin_memory=True, drop_last=True)
    dl_val = DataLoader(ds_val, cfg["batch_size"], shuffle=False, collate_fn=collate,
                        num_workers=2, pin_memory=True)
    logger.info(f"Treino: {len(ds_tr)} | Val: {len(ds_val)} | batches/época: {len(dl_tr)}")

    model = build_model(len(tok2id), cfg)
    if cfg.get("autoencoder_init"):
        ck = torch.load(cfg["autoencoder_init"], map_location=device, weights_only=True)
        sd = ck.get('model_state_dict', ck)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        logger.info(f"Init do autoencoder: {len(missing)} chaves ausentes, {len(unexpected)} inesperadas "
                    f"(arquiteturas diferentes => a maioria não bate; tudo bem, NADA é congelado)")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr_peak"], betas=(0.9, 0.98),
                            eps=1e-9, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, noam_lambda(cfg["embed_dim"], cfg["warmup_steps"], cfg["lr_peak"]))
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    best_val, sem_melhora, hist = float('inf'), 0, []
    step = 0
    for ep in range(1, cfg["epochs"] + 1):
        model.train(); t0 = time.time()
        tl = ta = nb = 0
        for src, dec_in, tgt in dl_tr:
            src, dec_in, tgt = src.to(device), dec_in.to(device), tgt.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                logits = model(src, dec_in)
                loss = loss_fn(logits.float(), tgt, cfg["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            tl += loss.item(); ta += acuracia_token(logits, tgt); nb += 1
            if nb % 200 == 0:
                logger.info(f"  ep {ep} | batch {nb}/{len(dl_tr)} | loss {tl/nb:.4f} | acc {ta/nb:.4f} | lr {sched.get_last_lr()[0]:.2e}")

        model.eval(); vl = va = nv = 0
        with torch.no_grad():
            for src, dec_in, tgt in dl_val:
                src, dec_in, tgt = src.to(device), dec_in.to(device), tgt.to(device)
                logits = model(src, dec_in)
                vl += loss_fn(logits, tgt, 0.0).item(); va += acuracia_token(logits, tgt); nv += 1
        vl /= nv; va /= nv
        hist.append({'epoch': ep, 'loss': tl/nb, 'acc': ta/nb, 'val_loss': vl, 'val_acc': va})
        logger.info(f"Época {ep}/{cfg['epochs']} | loss {tl/nb:.4f} acc {ta/nb:.4f} | "
                    f"val_loss {vl:.4f} val_acc {va:.4f} | {time.time()-t0:.0f}s")
        torch.save({'model_state_dict': model.state_dict(), 'config': cfg, 'epoch': ep}, f"trad_ep{ep:02d}.pt")
        if vl < best_val:
            best_val, sem_melhora = vl, 0
            torch.save({'model_state_dict': model.state_dict(), 'config': cfg, 'epoch': ep}, "trad_best.pt")
        else:
            sem_melhora += 1
            if sem_melhora >= cfg["patience"]:
                logger.info("Early stopping."); break
        # A cada 5 épocas mede a métrica que importa (top-1 canônica em 100 pares)
        if ep % 5 == 0:
            avaliar_traducao(model, reag_val, prod_val, tok2id, id2tok, cfg, n=100, mostrar=3)

    pd.DataFrame(hist).to_csv("historico_traducao.csv", index=False)
    ck = torch.load("trad_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    logger.info("Avaliação final (melhor checkpoint) na VALIDAÇÃO:")
    avaliar_traducao(model, reag_val, prod_val, tok2id, id2tok, cfg, n=cfg["n_eval_traducao"])
    return model, tok2id, id2tok, hist

def carregar_modelo(path, vocab_path):
    tok2id, id2tok = carregar_vocab(vocab_path)
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(len(tok2id), ck['config'])
    model.load_state_dict(ck['model_state_dict']); model.eval()
    return model, tok2id, id2tok, ck['config']

if __name__ == "__main__":
    treinar_traducao(CONFIG)
