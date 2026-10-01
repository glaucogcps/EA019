# ============================================================
# 11. GALERIA VISUAL — reagentes -> esperado -> previsto (RDKit)
# ============================================================
# Gera um HTML único (auto-contido, abre em qualquer navegador) com uma linha por reação:
#   reagentes | produto esperado | produto previsto | veredito + concordância
# Os átomos em que previsto e esperado diferem são destacados em vermelho
# (subestrutura comum máxima, MCS). Serve para diagnosticar erros em segundos.
#
# Uso:
#   df = pd.read_csv("avaliacao_mc.csv")            # saída de avaliar_mc()
#   gerar_galeria(df[df.modo == "mc_beam"], "galeria_mc_beam.html", tipos=None)
# ou, com casos manuais (CSV com colunas reag, esperado, tipo):
#   gerar_galeria_casos(model, casos_df, tok2id, id2tok, cfg, "galeria_casos.html", gram=gram)

import base64, html as _html
from rdkit import Chem
from rdkit.Chem import Draw, rdFMCS, AllChem
from rdkit.Chem.Draw import rdMolDraw2D

def _mol(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
    return m

def _atomos_diferentes(m_ref, m_pred):
    """Índices dos átomos de m_pred que NÃO estão na subestrutura comum com m_ref (e vice-versa)."""
    if m_ref is None or m_pred is None:
        return set(), set()
    try:
        res = rdFMCS.FindMCS([m_ref, m_pred], timeout=5, matchChiralTag=False,
                             ringMatchesRingOnly=True, completeRingsOnly=False)
        q = Chem.MolFromSmarts(res.smartsString) if res.numAtoms else None
    except Exception:
        q = None
    if q is None:
        return set(range(m_ref.GetNumAtoms())), set(range(m_pred.GetNumAtoms()))
    a_ref = set(m_ref.GetSubstructMatch(q)); a_pred = set(m_pred.GetSubstructMatch(q))
    return (set(range(m_ref.GetNumAtoms())) - a_ref, set(range(m_pred.GetNumAtoms())) - a_pred)

def _svg(m, w=300, h=200, destaque=(), cor=(1.0, 0.55, 0.55), legenda=""):
    if m is None:
        return f'<div style="width:{w}px;height:{h}px;display:flex;align-items:center;justify-content:center;color:#b00">SMILES inválido</div>'
    d = rdMolDraw2D.MolDraw2DSVG(w, h)
    opts = d.drawOptions(); opts.addStereoAnnotation = True; opts.legendFontSize = 12
    AllChem.Compute2DCoords(m)
    destaque = list(destaque)
    rdMolDraw2D.PrepareAndDrawMolecule(d, m, highlightAtoms=destaque,
                                       highlightAtomColors={i: cor for i in destaque}, legend=legenda)
    d.FinishDrawing()
    return d.GetDrawingText()

def _svg_reagentes(smi, w=420, h=200):
    mols = [_mol(s) for s in smi.split('.')]
    mols = [m for m in mols if m is not None]
    if not mols:
        return '<div>reagentes inválidos</div>'
    # remove sais/solventes pequenos da figura (<= 3 átomos pesados) se sobrar algo maior
    grandes = [m for m in mols if m.GetNumHeavyAtoms() > 3] or mols
    img = Draw.MolsToGridImage(grandes, molsPerRow=min(4, len(grandes)), subImgSize=(w // min(4, len(grandes)), h),
                               useSVG=True)
    return img.data if hasattr(img, 'data') else str(img)

_CSS = """
<style>
body{font-family:system-ui,Segoe UI,Arial;margin:16px;color:#222;background:#fafafa}
h1{font-size:20px} .resumo{margin:8px 0 16px;color:#444}
table{border-collapse:collapse;width:100%;background:#fff}
th,td{border:1px solid #ddd;padding:6px;vertical-align:top;text-align:center;font-size:12px}
th{background:#f0f0f0;position:sticky;top:0}
tr.ok td.ver{background:#e6f4ea;color:#137333;font-weight:600}
tr.err td.ver{background:#fce8e6;color:#c5221f;font-weight:600}
.smi{font-family:ui-monospace,Consolas,monospace;font-size:10px;word-break:break-all;color:#555;max-width:300px}
.tipo{display:inline-block;background:#e8f0fe;color:#1a56db;border-radius:8px;padding:1px 6px;font-size:11px}
.conc{font-size:14px}
</style>"""

def gerar_galeria(df, path, titulo="Galeria de reações", col_reag="reag", col_esp="esperado",
                  col_pred="pred", col_tipo="tipo", col_conc="concordancia", col_top3="acerto_top3"):
    """df precisa ter reag, esperado, pred (SMILES) e opcionalmente tipo, concordancia, acerto_top3."""
    linhas = []; acertos = 0
    for _, r in df.iterrows():
        m_esp, m_pred = _mol(r[col_esp]), _mol(r[col_pred])
        ok = (m_esp is not None and m_pred is not None and
              Chem.MolToSmiles(m_esp) == Chem.MolToSmiles(m_pred))
        acertos += ok
        d_esp, d_pred = (set(), set()) if ok else _atomos_diferentes(m_esp, m_pred)
        tipo = f'<span class="tipo">{_html.escape(str(r[col_tipo]))}</span>' if col_tipo in df.columns else ''
        conc = f'<div class="conc">concordância<br><b>{100*float(r[col_conc]):.0f}%</b></div>' if col_conc in df.columns else ''
        top3 = '' if col_top3 not in df.columns else ('<div>no top-3</div>' if (not ok and r[col_top3]) else '')
        veredito = 'ACERTOU' if ok else ('SMILES inválido' if m_pred is None else 'ERROU')
        linhas.append(f"""
<tr class="{'ok' if ok else 'err'}">
 <td>{tipo}<div>{_svg_reagentes(r[col_reag])}</div><div class="smi">{_html.escape(r[col_reag])}</div></td>
 <td>{_svg(m_esp, destaque=d_esp, cor=(0.6,0.8,1.0))}<div class="smi">{_html.escape(str(r[col_esp]))}</div></td>
 <td>{_svg(m_pred, destaque=d_pred)}<div class="smi">{_html.escape(str(r[col_pred]))}</div></td>
 <td class="ver">{veredito}{conc}{top3}</td>
</tr>""")
    n = len(df)
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>{_html.escape(titulo)}</title>{_CSS}</head><body>
<h1>{_html.escape(titulo)}</h1>
<div class="resumo">{n} reações · top-1 canônica: <b>{acertos}/{n} = {100*acertos/max(n,1):.1f}%</b>.
Azul = átomos do esperado que faltam no previsto; vermelho = átomos do previsto que não estão no esperado.</div>
<table><tr><th>Reagentes</th><th>Produto esperado</th><th>Produto previsto (top-1)</th><th>Veredito</th></tr>
{''.join(linhas)}</table></body></html>"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(doc)
    logger.info(f"Galeria salva em {path} ({n} reações, {acertos} acertos)")
    return path

def gerar_galeria_casos(model, casos, tok2id, id2tok, cfg, path, modo="mc_beam", n_rodadas=10,
                        gram=None, titulo="Casos de teste (fora do treino)"):
    """casos: DataFrame com colunas reag, esperado e (opcional) tipo. Traduz e monta a galeria.
    Devolve o DataFrame com pred, concordancia e acerto_top1/top3, além da tabela por tipo."""
    linhas = []
    for i, r in casos.reset_index(drop=True).iterrows():
        esp = _canon_ou_none(r["esperado"])
        rank, conc = traduzir_mc(model, r["reag"], tok2id, id2tok, cfg, modo, n_rodadas,
                                 cfg["beam_width"], n_best=cfg["beam_width"], gram=gram, seed=i)
        preds = [c for c, _, _ in rank]
        linhas.append({**r.to_dict(), "esperado": esp, "pred": preds[0] if preds else "",
                       "concordancia": conc, "acerto_top1": bool(preds) and preds[0] == esp,
                       "acerto_top3": esp in preds[:3]})
    df = pd.DataFrame(linhas)
    gerar_galeria(df, path, titulo=titulo)
    if "tipo" in df.columns:
        tab = df.groupby("tipo").agg(n=("acerto_top1", "size"), top1=("acerto_top1", "mean"),
                                     top3=("acerto_top3", "mean"), conc_media=("concordancia", "mean")).round(2)
        logger.info(f"\n{tab}")
        return df, tab
    return df, None
