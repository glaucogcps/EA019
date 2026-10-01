# v5 — Monte Carlo dropout, galeria visual e casos fora do treino

Data: 30/09/2026. Modelo testado: `trad_best.pt` (época 10, 7,47 M parâmetros) + `vocab_tokens.json` da pasta 30092026.

## Arquivos

| arquivo | o que é |
|---|---|
| `transformer_v5_mc.ipynb` | notebook v4 + célula 9 (MC dropout), célula 10 (avaliação nos 500 pares de validação), célula 11 (galeria visual e casos fora do treino). Roda no Kaggle como antes. |
| `mc_dropout.py`, `galeria_reacoes.py` | o mesmo código das células, em .py |
| `casos_teste.csv` | 20 reações de livro-texto montadas à mão, com tipo e produto esperado (para subir como dataset no Kaggle) |
| `resultado_casos_livro_texto.csv` + `galeria_casos_livro_texto.html` | resultado do modelo atual nesses 20 casos (abrir o HTML no navegador) |
| `mc_teste33.csv` + `galeria_30_pares_uspto.html` | comparação dos 3 modos nos 30 pares do teste anterior |

## 1. Qual é o número real do modelo

No log de treino do v4, a avaliação final em **500 pares de validação** (nunca vistos no treino) deu **53,6 % top-1**, com e sem SBS.
Os 72,7 % (24/33) vieram de pares tirados das **primeiras 50 linhas do CSV bruto**, que podem estar no treino. O número a reportar é ~54 %.
Referência da literatura no mesmo tipo de base (USPTO_STEREO): Molecular Transformer 76 % top-1 (modelo maior, base inteira, mais épocas).

## 2. MC dropout — o que é e o que deu

Manter o dropout ligado na inferência e gerar N vezes; cada rodada é uma "versão" ligeiramente diferente da rede.
Três modos implementados em `traduzir_mc(..., modo=)`:

- `beam`: beam search determinístico (o que já existia).
- `mc_greedy`: N rodadas greedy com dropout, voto simples.
- `mc_beam` (proposta): N rodadas de beam search com dropout; cada rodada guarda top-k com score; agrega o score por molécula canônica e ranqueia.

Resultado nos 30 pares (10 rodadas):

| modo | top-1 | top-3 | tempo (CPU) |
|---|---|---|---|
| beam | 23/30 | 26/30 | 27 s |
| mc_greedy | 23/30 | 24/30 | 179 s |
| mc_beam | 22/30 | 25/30 | 635 s |

**A acurácia não muda** (esperado: o dropout adiciona ruído a um modelo otimizado para rodar sem ele). **O que muda é que agora existe uma medida de confiança:**

| concordância entre as 10 rodadas (mc_beam) | n | acurácia |
|---|---|---|
| ≤ 50 % | 6 | 17 % |
| 71–90 % | 11 | 73 % |
| 91–100 % | 13 | **100 %** |

Ou seja: quando as 10 rodadas concordam, o modelo acertou sempre; quando discordam, quase sempre errou. Isso é o que o tutor precisa: o Transformer pode dizer ao Llama "confiança alta/baixa" em vez de entregar um produto sem qualificação. Falta confirmar em 500 pares (célula 10 do notebook).

## 3. Casos de livro-texto (fora do treino): 14/20

Acertou: SN2 com cianeto, Williamson, hidrogenação, esterificação de Fischer, acilação de fenol, redução com NaBH4, nitração, Friedel-Crafts, Wittig, amida, proteção Boc, Suzuki, Grignard, hidrólise de éster.

Errou (todos visíveis na galeria):

| caso | previu | leitura química | concordância |
|---|---|---|---|
| 1-bromobutano + NaOH | éter dibutílico | fez a substituição, mas usou o produto como nucleófilo (Williamson em vez de hidrólise) | 60 % |
| t-BuBr + KOtBu (E2) | éter di-t-butílico | ignorou a eliminação; tratou como substituição | **100 %** (erro confiante) |
| propeno + HBr | 4-bromobut-1-eno | não aprendeu adição eletrofílica a alceno simples | 30 % |
| butadieno + anidrido maleico (Diels-Alder) | devolveu o anidrido | não reconheceu a cicloadição | 50 % |
| 1-butanol + PCC | clorossulfonato | reagente PCC raro na base; alucinou | 30 % |
| benzaldeído + acetona (aldol) | produto sem a geometria E | acertou a conectividade; errou só a estereo (correto no top-3) | 100 % |

Padrão: o modelo vai bem em reações de acoplamento e de grupo funcional em moléculas "de patente" (Suzuki, amida, Boc), e mal em reações clássicas de Orgânica I com moléculas pequenas (E2, adição a alceno, Diels-Alder), que são raras no USPTO. Para o tutor, isso define em quais tipos de reação o Transformer pode ser usado como juiz hoje.

## 4. Próximos passos sugeridos

1. Rodar a célula 10 no Kaggle (500 pares) para confirmar a calibração da concordância. Custo: ~30 min na T4.
2. Ampliar `casos_teste.csv` para 50–100 casos (Gabriel), cobrindo cada tipo com 3–5 exemplos. É a tabela mais convincente para banca e revisor.
3. Para subir a acurácia do tradutor: base inteira (1,35 M), 20+ épocas, SMILES augmentation no treino (`doRandom=True`), modelo 6 camadas. É o caminho conhecido para chegar perto dos 76 % da literatura. MC dropout não faz isso.
4. No tutor: usar a concordância como sinal. ≥ 90 % → mostrar produto ao aluno; < 70 % → dizer que o modelo está inseguro e pedir ao Llama que raciocine só pela química.
