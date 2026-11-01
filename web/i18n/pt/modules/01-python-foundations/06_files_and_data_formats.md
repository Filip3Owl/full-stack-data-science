<!-- en:453d65755c -->
# 06 · Arquivos e Formatos de Dados

**Módulo 01 — Fundamentos de Python para Ciência de Dados**

---

## O que você vai aprender

1. `pathlib`: o jeito moderno de lidar com caminhos, e por que concatenar strings quebra.
2. Gerenciadores de contexto (`with`) e como escrever os seus.
3. CSV: o formato que todo mundo usa e cuja especificação ninguém lê.
4. JSON: dados aninhados, e os tipos que ele não consegue representar, sem nenhum aviso.
5. **Parquet**: armazenamento colunar, e por que ele é 10× menor e 50× mais rápido que CSV.
6. Uma comparação medida, para você escolher um formato com base em evidências.

**Tempo estimado:** 75 minutos

<!-- en:4deb25fce5 -->
## 1. `pathlib`: pare de construir caminhos com strings

Concatenar strings para formar caminhos é frágil: quebra entre sistemas operacionais, separadores duplicados aparecem e não há validação. `pathlib.Path` trata um caminho como um objeto estruturado.

<!-- en:c862dc572c -->
### Como encontrar arquivos

`glob` busca em um nível de diretório; `rglob` percorre recursivamente. Os dois devolvem geradores, então continuam baratos mesmo em árvores de diretórios enormes.

<!-- en:cf5cf9119a -->
> [!TIP]
> Defina seus caminhos **uma única vez**, perto do início do notebook, em relação a uma raiz de projeto que você calcula em vez de escrever à mão. Caminhos absolutos como `/Users/ana/Desktop/project/data.csv` são o motivo mais comum para um colega não conseguir executar o seu notebook.

<!-- en:cc223ebd80 -->
## 2. Gerenciadores de contexto

Um arquivo precisa ser fechado. Deixar arquivos abertos desperdiça recursos do sistema operacional e, pior para o trabalho com dados, gravações em buffer podem nunca chegar ao disco.

A instrução `with` garante a limpeza **mesmo que uma exceção seja gerada**.

<!-- en:a2bab8bff9 -->
Para leituras e gravações simples do arquivo inteiro, `pathlib` é ainda mais curto:

<!-- en:6ff454d74c -->
### Como escrever o seu próprio gerenciador de contexto

Qualquer par de preparação e limpeza é um candidato. `@contextmanager` transforma um gerador em um gerenciador: tudo antes do `yield` é preparação, e tudo depois é a limpeza garantida.

<!-- en:4d30b79a3d -->
## 3. Construa um conjunto de dados realista

Vamos criar algo grande o bastante para que a comparação de formatos faça sentido.

<!-- en:15e250737f -->
## 4. CSV: universal e cheio de armadilhas

CSV é uma *convenção*, não um padrão rigoroso. Todo mundo aceita esse formato, e essa é sua única vantagem real.

<!-- en:1a62d9e1f5 -->
### O que o CSV perde

Um arquivo CSV é *só texto*. Toda informação de tipo é descartada na gravação, e o pandas precisa adivinhar tudo de novo na leitura.

<!-- en:67e5fe0716 -->
> [!WARNING]
> `order_date` voltou como um simples `object` (string). Qualquer cálculo de datas com essa coluna agora vai falhar ou, pior, ordenar de forma lexicográfica, colocando `2024-10-01` antes de `2024-2-01`.

A solução é dizer ao pandas o que você espera, em vez de deixá-lo adivinhar:

<!-- en:fd005f54be -->
### As outras armadilhas do CSV

| Armadilha | Sintoma | Solução |
| :--- | :--- | :--- |
| Codificação errada | `UnicodeDecodeError` | `encoding="latin-1"` ou `"cp1252"` |
| Decimais no formato europeu | `1.234,56` lido como texto | `decimal=","`, `sep=";"` |
| Zeros à esquerda | `00123` → `123` | `dtype={"zip": str}` |
| IDs numéricos grandes | precisão perdida acima de 2⁵³ | `dtype={"id": str}` |
| `"NA"`, `"N/A"`, `"-"` | não reconhecidos como ausentes | `na_values=["NA", "-"]` |
| Arquivo maior que a RAM | `MemoryError` | `chunksize=100_000` |

<!-- en:45b1809c1c -->
## 5. JSON: dados aninhados

JSON lida com hierarquia, algo que o CSV não consegue. É o formato de praticamente toda API web da qual você vai buscar dados.

<!-- en:9c96a05756 -->
### Como achatar JSON aninhado para análise

`pd.json_normalize` transforma estruturas aninhadas em uma tabela plana: é a ponte entre a resposta de uma API e um DataFrame.

<!-- en:57ebc5bc41 -->
> [!WARNING]
> **JSON não tem tipo de data nem NaN.** Datas viram strings, e `NaN`/`Infinity` não são JSON válido. O módulo `json` de Python os grava mesmo assim, produzindo arquivos que outras linguagens rejeitam.

<!-- en:5db66a36e5 -->
## 6. Parquet: o formato para ciência de dados

O Parquet guarda os dados **por coluna** em vez de por linha, com compressão por coluna e um esquema embutido.

```text
CSV (orientado a linhas)        Parquet (orientado a colunas)
─────────────────────           ─────────────────────────
1,organic,BR,54.20              order_id : [1, 2, 3, …]      ← um só tipo, comprime bem
2,paid,US,120.00                channel  : [organic, paid, …] ← codificado com dicionário
3,paid,BR,17.50                 country  : [BR, US, BR, …]
                                amount   : [54.20, 120.00, …]
```

Três consequências decorrem diretamente desse layout:

1. **A compressão é muito melhor**: uma coluna tem um só tipo, muitas vezes com poucos valores distintos.
2. **Ler um subconjunto de colunas quase não custa nada**: as outras colunas nem são tocadas.
3. **O esquema fica guardado no arquivo**: os dtypes sobrevivem exatamente à ida e à volta.

<!-- en:33ed9cc643 -->
### Seleção de colunas: o recurso que muda o seu jeito de trabalhar

Ler duas colunas de um arquivo Parquet de 50 colunas custa aproximadamente o que essas duas colunas custam. Com CSV, você precisa interpretar cada byte de cada linha de qualquer forma.

<!-- en:11e858cefc -->
## 7. Comparação medida

Chega de afirmações: vamos medir o tamanho e a velocidade dos três formatos.

<!-- en:b2bbfafdb1 -->
## 8. Como escolher um formato

| Formato | Use quando | Evite quando |
| :--- | :--- | :--- |
| **CSV** | Uma pessoa precisa abrir o arquivo; você troca dados com uma equipe não técnica | Os dados são grandes, tipados ou aninhados |
| **JSON** | Dados aninhados ou hierárquicos; payloads de API; configuração | Dados tabulares em qualquer escala real |
| **Parquet** | **Qualquer conjunto de dados analítico que você controla**: é o padrão | Uma pessoa precisa ler em um editor de texto |
| **Pickle** | Cache de curta duração, só em Python | *Qualquer coisa compartilhada ou de longa duração*; veja abaixo |
| **Feather** | Troca temporária muito rápida entre sessões de pandas e R | Arquivamento de longo prazo |

> [!WARNING]
> **Nunca carregue um pickle de uma fonte não confiável.** Carregar um pickle executa código arbitrário por definição, então um `.pkl` malicioso é simplesmente um programa que você aceitou executar. Pickles também são frágeis entre versões de bibliotecas: um arquivo gravado com pandas 2.x pode não carregar no 3.x.

<!-- en:0b0cb5e368 -->
### A convenção que este curso segue

```text
data/raw/         ← exatamente como foi recebido, nunca modificado (CSV, JSON, o que tiver chegado)
data/processed/   ← limpo, tipado, pronto para análise  → sempre Parquet
```

<!-- en:17abb9f0c1 -->
## 9. Exercícios

### Exercício 6.1: manipulação de caminhos

Dado `Path("data/raw/sales_2024_q1.csv")`, use `pathlib` (sem métodos de string no caminho completo) para produzir:

1. `data/processed/sales_2024_q1.parquet`
2. O ano e o trimestre como strings separadas, extraídas do stem.
3. Um caminho irmão `sales_2024_q1_backup.csv`.

### Exercício 6.2: diagnostique um CSV quebrado

Grave um arquivo com este conteúdo:

```text
id;name;price;date
001;Café;1.234,56;15/03/2024
002;Naïve;987,65;16/03/2024
```

Depois, leia-o em um DataFrame com os dtypes corretos: `id` como string com os zeros à esquerda, `price` como float e `date` como datetime.

### Exercício 6.3: JSON aninhado

Achate isto em uma linha por item, levando os campos do pedido para cada linha:

```python
{"order_id": 1, "customer": {"name": "Ana", "city": "SP"},
 "items": [{"sku": "A", "qty": 2}, {"sku": "B", "qty": 1}]}
```

### Exercício 6.4: comprove a seleção de colunas

Construa um DataFrame com 50 colunas e 500 000 linhas. Meça o tempo de leitura de **uma** coluna em Parquet e em CSV. Explique a proporção em termos do layout de armazenamento.

### Exercício 6.5: gravação segura

Estenda `atomic_write` para também receber um argumento `backup: bool`, que mantém a versão anterior como `<name>.bak` antes de substituí-la.

<!-- en:46f92dab9f -->
## Principais lições

- Use **`pathlib`**, nunca concatenação de strings. Calcule a raiz do projeto; nunca escreva caminhos absolutos à mão.
- **Sempre use `with`** com arquivos: ele os fecha mesmo quando uma exceção acontece.
- **O CSV perde todos os dtypes.** Passe `parse_dates` e um mapa explícito de `dtype`, e sempre grave com `index=False`.
- **JSON não tem tipo de data nem NaN.** Converta deliberadamente antes de gravar.
- **Parquet é o padrão para dados analíticos**: menor, mais rápido, autodescritivo e com seleção de colunas.
- **Nunca carregue um pickle de dados não confiáveis**: isso executa código arbitrário.
- Convenção: `data/raw/` é a entrada imutável; `data/processed/` é a saída em Parquet.

---

**Próximo:** [07 · Erros, Logging e Testes](07_errors_logging_and_testing.ipynb): como fazer um pipeline falhar de forma clara, e não em silêncio.

<!-- comments -->
EN: # The `/` operator joins path components, using the correct separator for the OS.
PT: # O operador `/` junta partes do caminho, usando o separador correto para o sistema.
EN: # `parents=True` creates intermediate directories; `exist_ok=True` makes the call
PT: # `parents=True` cria os diretórios intermediários; `exist_ok=True` torna a chamada
EN: # idempotent, so re-running the notebook is safe.
PT: # idempotente, então executar o notebook de novo é seguro.
EN: # Paths decompose into named parts — no string slicing required.
PT: # Caminhos se decompõem em partes nomeadas, sem precisar fatiar strings.
EN: # Deriving a related path is one call, not string surgery.
PT: # Derivar um caminho relacionado é uma chamada, não uma cirurgia em strings.
EN: # Every notebook source in this module.
PT: # Todos os arquivos-fonte de notebook deste módulo.
EN: # ✘ Fragile: if the write raises, close() never runs.
PT: # ✘ Frágil: se a gravação gerar um erro, close() nunca é executado.
EN: # ✔ Correct: closed automatically, exception or not.
PT: # ✔ Correto: fechado automaticamente, com ou sem exceção.
EN: # the body of the `with` block runs here
PT: # o corpo do bloco `with` roda aqui
EN: # clean up the partial file
PT: # limpa o arquivo parcial
EN: # re-raise so the caller still sees it
PT: # gera o erro de novo, para quem chamou ainda vê-lo
EN: # atomic on POSIX: readers see old or new, never partial
PT: # atômico em POSIX: quem lê vê o antigo ou o novo, nunca um parcial
EN: # Now simulate a crash halfway through.
PT: # Agora simula uma falha no meio do caminho.
EN: # Low-cardinality strings — the case where columnar formats shine.
PT: # Strings de baixa cardinalidade: o caso em que formatos colunares brilham.
EN: # Always check dtypes after building or loading a DataFrame. A numeric column
PT: # Sempre confira os dtypes depois de construir ou carregar um DataFrame. Uma coluna
EN: # that arrived as `object` is the single most common cause of a slow, wrong
PT: # numérica que chegou como `object` é a causa mais comum de uma análise lenta e
EN: # analysis — and it is invisible in `.head()`.
PT: # errada, e ela é invisível em `.head()`.
EN: # `index=False` matters: without it pandas writes the RangeIndex as an unnamed
PT: # `index=False` importa: sem ele, o pandas grava o RangeIndex como uma primeira
EN: # first column, which reappears as "Unnamed: 0" the next time anyone reads it.
PT: # coluna sem nome, que reaparece como "Unnamed: 0" na próxima leitura.
EN: # An explicit dtype map is faster and safer than inference. `category` also
PT: # Um mapa explícito de dtypes é mais rápido e seguro que a inferência. `category`
EN: # cuts memory dramatically for repeated strings.
PT: # também reduz drasticamente a memória de strings repetidas.
EN: # Reading in chunks: the CSV equivalent of the generator pattern from notebook 04.
PT: # Leitura em blocos: o equivalente em CSV do padrão de geradores do notebook 04.
EN: # `indent` makes it human-readable and diffable; `ensure_ascii=False` keeps
PT: # `indent` deixa legível e fácil de comparar; `ensure_ascii=False` mantém os
EN: # accented characters as themselves rather than \uXXXX escapes.
PT: # caracteres acentuados como são, em vez de escapes \uXXXX.
EN: # the list to turn into rows
PT: # a lista que vai virar linhas
EN: # fields to broadcast onto every row
PT: # campos replicados em cada linha
EN: # Python happily writes invalid JSON by default…
PT: # Python grava JSON inválido sem reclamar, por padrão…
EN: # …but this is what a strict parser elsewhere will do with it:
PT: # …mas é isto que um parser rigoroso fará com ele em outro lugar:
EN: # One colour per format, held constant across all three panels so the reader
PT: # Uma cor por formato, mantida nos três painéis, para quem lê
EN: # tracks a format by colour rather than by position.
PT: # acompanhar um formato pela cor em vez da posição.
EN: # Direct labels: with three bars a legend would be redundant ink.
PT: # Rótulos diretos: com três barras, uma legenda seria tinta desperdiçada.
EN: # Demonstrate the convention: read raw, clean, write processed.
PT: # Demonstra a convenção: lê o bruto, limpa, grava o processado.
EN: # drop non-positive amounts
PT: # remove valores não positivos
EN: # Tidy up the benchmark artefacts so the repo stays clean.
PT: # Limpa os arquivos do teste de desempenho para manter o repositório limpo.
