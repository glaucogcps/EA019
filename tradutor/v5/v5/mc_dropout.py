# ============================================================
# 9. MONTE CARLO DROPOUT — inferência com incerteza
# ============================================================
# Ideia (Gal & Ghahramani, 2016; proposta): manter o dropout LIGADO na
# inferência, gerar várias vezes e votar. Cada rodada é uma "versão" ligeiramente
# diferente da rede. A fração de concordância entre as rodadas é uma medida de
# confiança do modelo naquela previsão.
#
# Três modos:
#   "beam"     -> beam search determinístico (o que já existia)         [referência]
#   "mc_greedy"-> n rodadas greedy com dropout, voto simples            [barato]
#   "mc_beam"  -> n rodadas de beam search com dropout, cada uma devolve
#                 top-k candidatos com score; agrega score por molécula
#                 canônica e ranqueia (proposta)              [mais caro]

def _set_dropout(model, ativo: bool):
    """Liga/desliga só o dropout. O modelo não tem BatchNorm, então model.train()
    dentro de no_grad afeta apenas o dropout. Não use módulos individuais em .train():
    em modo eval o TransformerEncoder usa um caminho rápido que ignora o dropout."""
    model.train() if ativo else model.eval()

@torch.no_grad()
def traduzir_com_scores(model, smiles_reagentes, tok2id, id2tok, max_len=300, beam_width=5,
                        alpha=0.6, n_best=5, gram=None, dropout_ativo=False):
    """Igual a traduzir(), mas devolve [(smiles, log_score_normalizado)] e permite dropout."""
    _set_dropout(model, dropout_ativo)
    sos, eos = tok2id[SOS], tok2id[EOS]
    src = torch.tensor([codificar(smiles_reagentes, tok2id, max_len)], device=device)
    mem, src_pad = model.encode(src)
    mem = mem.expand(beam_width, -1, -1); src_pad = src_pad.expand(beam_width, -1)
    seqs = torch.full((1, 1), sos, dtype=torch.long, device=device)
    scores = torch.zeros(1, device=device)
    estados = [gram.estado_inicial()] if gram else None
    terminadas = []
    for t in range(max_len):
        k = seqs.size(0)
        logits = model.decode(seqs, mem[:k], src_pad[:k])[:, -1]
        logp = F.log_softmax(logits.float(), -1)
        logp[:, 0] = -1e9; logp[:, sos] = -1e9
        if gram is not None:
            logp = logp.masked_fill(~gram.mascara(estados, logp.device), -1e9)
        cand = (scores.unsqueeze(1) + logp).view(-1)
        top = cand.topk(min(beam_width * 2, cand.numel()))
        novas_seqs, novos_scores, novos_estados = [], [], []
        for s, idx in zip(top.values.tolist(), top.indices.tolist()):
            if s < -1e8:
                continue
            b, tok = divmod(idx, logp.size(1))
            if tok == eos:
                L = seqs.size(1)
                terminadas.append((s / (((5 + L) / 6) ** alpha), seqs[b, 1:].tolist()))
            else:
                novas_seqs.append(torch.cat([seqs[b], torch.tensor([tok], device=device)]))
                novos_scores.append(s)
                if gram is not None:
                    novos_estados.append(gram.avancar(estados[b], tok))
            if len(novas_seqs) == beam_width:
                break
        if not novas_seqs:
            break
        seqs = torch.stack(novas_seqs); scores = torch.tensor(novos_scores, device=device)
        estados = novos_estados if gram is not None else None
        if len(terminadas) >= beam_width:
            melhor_viva = scores.max().item() / (((5 + max_len) / 6) ** alpha)
            if melhor_viva < max(terminadas)[0]:
                break
    if not terminadas:
        terminadas = [(s / (((5 + seqs.size(1)) / 6) ** alpha), seq[1:].tolist())
                      for s, seq in zip(scores.tolist(), seqs)]
    terminadas.sort(key=lambda x: -x[0])
    model.eval()
    return [(''.join(id2tok[i] for i in toks), sc) for sc, toks in terminadas[:n_best]]

def _canon_ou_none(smi):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi) if smi else None
    return Chem.MolToSmiles(m) if m is not None else None

@torch.no_grad()
def traduzir_mc(model, smiles_reagentes, tok2id, id2tok, cfg, modo="mc_beam",
                n_rodadas=10, beam_width=5, n_best=5, gram=None, seed=None):
    """
    Devolve (lista ranqueada [(smiles_canônico, score, concordância)], concordância_top1).
      concordância = fração das rodadas em que a molécula apareceu como 1ª escolha.
      score        = soma sobre as rodadas de exp(log_score) da molécula (mc_beam),
                     ou nº de votos (mc_greedy).
    modo="beam" reproduz o beam determinístico (1 rodada, dropout desligado).
    """
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    if seed is not None:
        torch.manual_seed(seed)
    if modo == "beam":
        n_rodadas, dropout, bw = 1, False, beam_width
    elif modo == "mc_greedy":
        dropout, bw = True, 1
    elif modo == "mc_beam":
        dropout, bw = True, beam_width
    else:
        raise ValueError(modo)
    soma_score, votos_top1 = Counter(), Counter()
    for _ in range(n_rodadas):
        cands = traduzir_com_scores(model, smiles_reagentes, tok2id, id2tok, cfg["max_len"],
                                    bw, n_best=n_best, gram=gram, dropout_ativo=dropout)
        vistos = set()
        for r, (smi, sc) in enumerate(cands):
            c = _canon_ou_none(smi)
            if c is None or c in vistos:          # SMILES inválido ou duplicado nesta rodada
                continue
            vistos.add(c)
            soma_score[c] += math.exp(sc)
            if r == 0:
                votos_top1[c] += 1
    model.eval()
    if not soma_score:
        return [], 0.0
    # ranqueamento: primeiro por votos de 1º lugar, depois pelo score agregado
    rank = sorted(soma_score, key=lambda c: (votos_top1[c], soma_score[c]), reverse=True)
    saida = [(c, soma_score[c], votos_top1[c] / n_rodadas) for c in rank]
    return saida, saida[0][2]

def avaliar_mc(model, reag, prod, tok2id, id2tok, cfg, modos=("beam", "mc_greedy", "mc_beam"),
               n=200, n_rodadas=10, beam_width=5, gram=None, seed=0, csv_path=None):
    """Compara os modos nos mesmos pares e devolve um DataFrame com uma linha por (par, modo).
    Colunas: acerto_top1, acerto_top3, concordancia, n_candidatos, pred, esperado."""
    n = min(n, len(reag)); linhas = []
    for modo in modos:
        t0 = time.time(); ac1 = ac3 = 0
        for i in range(n):
            esp = _canon_ou_none(prod[i])
            rank, conc = traduzir_mc(model, reag[i], tok2id, id2tok, cfg, modo, n_rodadas,
                                     beam_width, n_best=beam_width, gram=gram, seed=seed + i)
            preds = [c for c, _, _ in rank]
            ok1 = bool(preds) and preds[0] == esp
            ok3 = esp in preds[:3]
            ac1 += ok1; ac3 += ok3
            linhas.append({'i': i, 'modo': modo, 'acerto_top1': ok1, 'acerto_top3': ok3,
                           'concordancia': conc, 'n_candidatos': len(preds),
                           'pred': preds[0] if preds else '', 'esperado': esp, 'reag': reag[i]})
        logger.info(f"[{modo:9s}] top-1 {ac1}/{n} = {100*ac1/n:.1f}% | top-3 {100*ac3/n:.1f}% | {time.time()-t0:.0f}s")
    df = pd.DataFrame(linhas)
    if csv_path:
        df.to_csv(csv_path, index=False)
    return df

def relatorio_concordancia(df, modo="mc_beam"):
    """Acurácia por faixa de concordância: se a curva for crescente, a concordância é
    uma medida de confiança calibrada (útil para o tutor)."""
    d = df[df.modo == modo].copy()
    faixas = pd.cut(d.concordancia, [-0.01, 0.5, 0.7, 0.9, 1.0],
                    labels=['≤50%', '51–70%', '71–90%', '91–100%'])
    tab = d.groupby(faixas, observed=False).agg(n=('acerto_top1', 'size'),
                                                acuracia=('acerto_top1', 'mean'))
    logger.info(f"\n{tab}")
    return tab
