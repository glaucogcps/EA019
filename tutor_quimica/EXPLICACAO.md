# Como funciona o chat do tutor de Química Orgânica

## 1. O problema que o chat resolve

Modelos de linguagem como o Llama conversam bem em português, mas inventam química com frequência: escrevem estruturas impossíveis ou erram o produto de uma reação. Num tutor isso é grave, porque o aluno aprende errado.

A solução é dividir o trabalho. Cada peça faz só o que sabe fazer bem:

| Peça | Papel | Analogia |
|---|---|---|
| **RDKit** | Confere se a molécula existe e padroniza a escrita | O corretor ortográfico |
| **Transformer (v4 do GitHub)** | Prevê o produto a partir dos reagentes | O especialista que já viu muitas reações |
| **Llama** | Conversa com o aluno em português | O professor que fala |

**Regra central do sistema: o Llama nunca decide se a química está certa.** Quem julga é o código Python, usando o RDKit e o gabarito ou o Transformer. O Llama recebe o veredito pronto e só o transforma numa pergunta socrática.

## 2. Como as moléculas viram texto (SMILES)

Todas as peças falam a mesma "língua": SMILES, um jeito de escrever moléculas como texto.

- Etanol: `CCO`
- Ácido acético: `CC(=O)O`
- Fenol: `Oc1ccccc1` (letras minúsculas indicam anel aromático)
- Vários reagentes juntos são separados por ponto: `CC(=O)Cl.Oc1ccccc1`

A mesma molécula pode ser escrita de vários jeitos (`CC(C)Br` e `BrC(C)C` são o mesmo 2-bromopropano). Por isso o RDKit converte tudo para uma **forma canônica**, uma escrita única e padronizada, antes de qualquer comparação.

## 3. De onde vem o conhecimento do Transformer

### O treino (já realizado)

O Transformer foi treinado com cerca de **400 mil reações de patentes americanas** (base USPTO). Cada exemplo era um par: reagentes → produto, em SMILES. Durante o treino, o modelo ajustou seus 7,5 milhões de parâmetros para aprender regularidades, como "quando aparece `C(=O)Cl` junto de um `OH`, o produto costuma ter `C(=O)O`".

O resultado do treino são dois arquivos da pasta `tradutor/v4 best`:

- **`trad_best.pt`**: os parâmetros aprendidos (os "pesos"). Não contém as reações, só os padrões.
- **`vocab_tokens.json`**: o dicionário que converte cada pedaço de SMILES num número.

A base de 400 mil reações **não está no chat** e não é usada em momento nenhum depois do treino.

> **Analogia:** é como um aluno que resolveu milhares de exercícios e depois faz a prova sem consulta. Ele não lembra de cada exercício, mas aprendeu os padrões.

### Por que se chama "tradutor"

A arquitetura é a mesma usada para traduzir inglês para alemão. Aqui, a "frase de entrada" são os reagentes e a "tradução" é o produto. Não é um autoencoder: um autoencoder reconstrói a própria entrada, e aqui a saída é uma molécula diferente.

## 4. Como o Transformer calcula um produto

Quando o modelo recebe reagentes:

1. **RDKit padroniza** o SMILES, porque o modelo foi treinado com SMILES nesse formato.
2. **O texto é quebrado em pedaços** (tokens): `C`, `C`, `(`, `=`, `O`, `)`, `Cl`... Átomos de duas letras como `Cl` e `Br` viram um pedaço só. Cada pedaço vira um número pelo `vocab_tokens.json`.
3. **O encoder lê os reagentes** e monta uma representação interna deles.
4. **O decoder escreve o produto um pedaço por vez**, sempre escolhendo os pedaços mais prováveis com base no que já escreveu. É parecido com o teclado do celular prevendo a próxima palavra.
5. **Uma gramática de SMILES** impede o modelo de escrever coisas sem sentido, como fechar um parêntese que nunca abriu.
6. **Beam search**: em vez de seguir um só caminho, o modelo mantém as 5 melhores versões em paralelo e no fim devolve as 3 melhores.
7. **RDKit canonicaliza** os candidatos e remove repetidos (às vezes o modelo escreve a mesma molécula de dois jeitos).

O modelo **gera** o produto. Ele não procura a reação numa tabela.

## 5. O que o Transformer sabe e não sabe

- **Acerta mais** reações comuns em patentes: formação de amidas e ésteres, acoplamento de Suzuki, remoção de Boc. Ele viu muitos exemplos desses padrões, então acerta mesmo com moléculas que nunca viu.
- **Erra mais** reações clássicas de Orgânica I: HBr em alceno, Diels-Alder, oxidação com PCC, desidratação de álcool. Essas reações quase não aparecem em patentes com moléculas pequenas.
- **Acerta cerca de metade** das reações no geral (53,6% na validação).
- **Não diz quando está inseguro.** Sempre devolve alguma resposta, mesmo para reações sem sentido.
- **Depende dos reagentes completos.** Aprendeu com reações que trazem catalisador e solvente. Com `CCO` sozinho, sem ácido sulfúrico, ele erra.
- **Não enxerga condições.** Calor, luz e temperatura não cabem no SMILES.
- **Não conhece intermediários.** Só conhece reagentes e produto final, então não sabe nada de carbocátions ou estados de transição.

## 6. Os dois modos do chat

### Modo Exercícios

- Reações escritas à mão no `exercicios.json`.
- Cada exercício tem: reagentes, **gabarito**, nome do produto e uma nota sobre o mecanismo, que o Llama usa para fazer perguntas melhores.
- **Quem decide o certo é o gabarito.** O Transformer também roda, mas só para registro (aparece no Modo professor).
- Os 6 exercícios atuais são exemplos. O ideal é um professor revisar e ampliar.

### Modo Reação livre

- O aluno digita os reagentes em SMILES e clica em Começar.
- **O Transformer calcula o produto na hora**, e esse produto vira a referência.
- O aluno pode tentar desenhar a resposta e conversar com o tutor, ou clicar em "Ver produto calculado pelo modelo" para ver a previsão e as alternativas.
- Antes de rodar, o sistema avisa se o SMILES é inválido ou se tem pedaços que o modelo nunca viu no treino.

### Por que os modos falam com segurança diferente

| | Reagentes vêm de | Referência vem de | Como o tutor fala |
|---|---|---|---|
| Exercícios | `exercicios.json` | Gabarito conferido por uma pessoa | Afirma se está certo ou errado |
| Reação livre | O aluno | Palpite do Transformer | Fala em "provavelmente" e puxa para o mecanismo |

Como o v4 não mede a própria incerteza, o código atribui a ele uma confiança fixa de 0,7. Esse valor fica entre os limiares de "certeza" (0,9) e "não sei" (0,5), então o tutor sempre trata a previsão do modelo como provável. Isso evita o pior erro de um tutor: dizer ao aluno que uma resposta certa está errada.

## 7. O que acontece a cada mensagem do aluno

1. **O aluno envia** uma estrutura (SMILES ou desenho) ou só escreve uma dúvida.
2. **RDKit valida a estrutura.** Se for impossível (carbono com 5 ligações, parêntese aberto), o veredito já é `invalido`. Se for válida, é canonicalizada.
3. **O código compara** com a referência (gabarito ou Transformer).
4. **Sai um veredito:**

| Veredito | Quando acontece |
|---|---|
| `correto` / `incorreto` | Com gabarito, resposta igual ou diferente |
| `provavelmente_correto` / `provavelmente_incorreto` | Sem gabarito, comparando com o Transformer |
| `invalido` | Molécula impossível |
| `igual_reagente` | O aluno enviou um dos reagentes, ou seja, nada reagiu |
| `sem_estrutura` | O aluno só escreveu texto |
| `incerto` | O modelo não conseguiu prever nada |
| `resolvido` | O aluno já acertou e agora a conversa é livre |

5. **O veredito escolhe uma instrução** da tabela `POLITICA` em `tutor.py`. Por exemplo, para `incorreto`: "não revele o produto, faça uma pergunta sobre a etapa do mecanismo onde ele errou".
6. **O Llama recebe um prompt oculto** com os fatos (enunciado, veredito, número de tentativas, nota sobre o mecanismo), a instrução e o histórico da conversa. Ele escreve a resposta.
7. **Um filtro confere a resposta.** Se o Llama escreveu uma estrutura em SMILES ou o nome do produto antes do aluno acertar, a resposta é descartada e gerada de novo, até 2 vezes. Se continuar vazando, entra uma resposta segura padrão.

```
Aluno envia estrutura
        │
        ▼
RDKit valida e canonicaliza ──── inválida? → veredito "invalido"
        │
        ▼
Compara com a referência (gabarito ou Transformer)
        │
        ▼
Veredito → instrução da tabela POLITICA
        │
        ▼
Llama escreve a resposta socrática
        │
        ▼
Filtro: vazou SMILES ou o produto? → gera de novo
        │
        ▼
Resposta aparece para o aluno
```

### Exemplo completo: acetilação do fenol

- Reagentes: `CC(=O)Cl.Oc1ccccc1` (cloreto de acetila e fenol).
- Gabarito: `CC(=O)Oc1ccccc1` (acetato de fenila). O oxigênio do fenol ataca a carbonila, forma um intermediário tetraédrico e o cloreto sai.
- O aluno envia `CC(=O)c1ccc(O)cc1` (acilação no anel). O RDKit diz que é válida, o código compara com o gabarito e o veredito é `incorreto`.
- O Llama recebe a instrução de não revelar o produto e pergunta algo como: "Sem um ácido de Lewis, qual átomo do fenol é o nucleófilo mais disponível?".
- O aluno envia `CC(=O)Oc1ccccc1`. Veredito `correto`. O tutor pede que ele justifique pelo mecanismo.

## 8. Os arquivos do projeto

| Arquivo | Responsabilidade |
|---|---|
| `app.py` | A tela: modos, editor de estrutura, chat, Modo professor |
| `tutor.py` | O cérebro: veredito, prompt do Llama, tabela `POLITICA`, filtro de saída |
| `verificador.py` | A ponte entre o tutor e o Transformer |
| `quimica_tradutor.py` | A parte de inferência copiada do `transformer_v4.ipynb` |
| `llm.py` | A conexão com o Llama via Ollama |
| `exercicios.json` | O conteúdo pedagógico |
| `trad_best.pt` + `vocab_tokens.json` | O modelo treinado, baixado do GitHub |
| `test_tutor.py` | Testes da lógica, sem precisar do Llama nem do modelo |

## 9. Modo professor

Um painel de depuração na barra lateral. Mostra a referência, de onde ela veio, a previsão do Transformer, as alternativas, se o modelo discorda do gabarito e o último veredito.

Serve para descobrir de quem é a culpa quando algo sai estranho:

- **Veredito certo, resposta ruim:** o problema é o Llama ou o prompt.
- **Veredito errado:** o problema é o gabarito ou a previsão do Transformer.

O aluno não deve usar, porque ele mostra a resposta.

## 10. Limitações atuais

- O sistema confere **só o produto final**, não o mecanismo passo a passo, porque o Transformer não conhece intermediários.
- Reações em que a **condição decide o produto** (E1 vs SN1, controle cinético vs termodinâmico) ficam fora do alcance do modelo.
- Na reação livre, o modelo **erra cerca de metade** das vezes e erra mais justamente o conteúdo de graduação.
- A qualidade das perguntas depende do Llama. O `llama3.2:3b` é fraco em português e química; o `llama3.1:8b` é bem melhor.
- O aluno precisa informar a estrutura em SMILES ou desenhá-la. Nome em texto ("tolueno") não é avaliado, porque converter nome em estrutura com o Llama reintroduziria a alucinação que o sistema quer evitar.

## 11. Onde está a alavanca para melhorar

O que mais melhora o tutor hoje, sem depender de novo treino:

1. **Ampliar o `exercicios.json`** com reações revisadas por um professor. Com gabarito, o tutor fica confiável independentemente do Transformer.
2. **Refinar a tabela `POLITICA`** testando com o Llama real, para que as perguntas socráticas fiquem boas.
3. **Usar um Llama maior** se o computador aguentar.

As melhorias previstas na proposta do CENAPAD (modelo maior, medição de incerteza, ajuste com reações didáticas) entram depois e se encaixam sem mudar o chat: basta trocar o `trad_best.pt` e ajustar a confiança no `verificador.py`.
