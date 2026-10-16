# Tutor socrático de Química Orgânica (parte do chat)

Usa o modelo que já está no GitHub (`tradutor/v4 best`). Nada de treino.

## Montar a pasta

```
tutor_quimica/
├── app.py, tutor.py, llm.py, verificador.py, quimica_tradutor.py
├── exercicios.json, requirements.txt, test_tutor.py
├── trad_best.pt          <- baixado de tradutor/v4 best no GitHub
└── vocab_tokens.json     <- baixado de tradutor/v4 best no GitHub
```

## Rodar

```bash
pip install -r requirements.txt
# instalar o Ollama: https://ollama.com
ollama pull llama3.1:8b
streamlit run app.py
```

## O que cada arquivo faz

| Arquivo | Função |
|---|---|
| `app.py` | Interface: escolha do exercício, editor de estrutura, chat |
| `tutor.py` | Decide o veredito, monta o prompt do Llama, filtra a resposta |
| `llm.py` | Conversa com o Llama via Ollama |
| `quimica_tradutor.py` | Carrega o `trad_best.pt` e prevê produtos (cópia da inferência do notebook) |
| `verificador.py` | Liga o modelo ao tutor |
| `exercicios.json` | Exercícios com gabarito e nota sobre o mecanismo |
| `test_tutor.py` | Testa a lógica sem precisar do Llama nem do modelo |

## Modos

- **Exercícios**: reações do `exercicios.json`, com gabarito.
- **Reação livre**: o aluno digita os reagentes em SMILES (separados por ponto) e clica em
  Começar. O modelo calcula o produto na hora. O aluno pode tentar desenhar a resposta e
  conversar com o tutor, ou clicar em "Ver produto calculado pelo modelo" para ver a
  previsão e as alternativas.

## Como o tutor decide

- **Exercício com gabarito**: o gabarito decide se o aluno acertou. O modelo não interfere.
- **Exercício sem gabarito**: quem decide é o modelo v4. Como ele acerta cerca de metade
  das reações e não mede a própria incerteza, o tutor fala em tom de "provavelmente" e
  conduz a discussão pelo mecanismo, sem afirmar certo ou errado.

## Um turno do chat

1. Aluno desenha a estrutura (ou só escreve uma dúvida).
2. RDKit valida e padroniza a estrutura.
3. Compara com o gabarito (ou com a previsão do modelo).
4. O resultado escolhe uma instrução da tabela `POLITICA` em `tutor.py`.
5. Llama recebe essa instrução e responde.
6. Se o Llama escrever uma estrutura ou entregar o nome do produto antes da hora,
   a resposta é descartada e gerada de novo.
