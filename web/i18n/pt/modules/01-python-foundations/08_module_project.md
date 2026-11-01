<!-- en:e466904063 -->
# 08 · Projeto do Módulo: um Pipeline de ETL Testado

**Módulo 01 — Fundamentos de Python para Ciência de Dados**

---

## O briefing

Você entrou na equipe de dados da **Nordwind Retail**. Dois sistemas exportam dados toda noite, e nenhum deles se comporta bem:

- `orders.csv`: o sistema de gestão de pedidos. Codificação Latin-1, vírgula decimal no formato europeu, maiúsculas e minúsculas inconsistentes, `"N/A"` para nulos e linhas duplicadas sempre que uma nova tentativa é disparada.
- `customers.json`: o CRM, como uma exportação aninhada de API.

**Sua tarefa:** construir um pipeline que transforme isso em uma única tabela Parquet limpa, tipada e pronta para análise, e produzir a partir dela um breve relatório de receita.

Ele precisa ser **defensivo** (dados ruins falham de forma clara), **observável** (com logs) e **testado**.

## O que este projeto exercita

| Notebook | Conceito usado aqui |
| :--- | :--- |
| 02 | Imutabilidade, comparação de floats, limpeza de strings, semântica de `nan` |
| 03 | Buscas em `set`, `defaultdict`, comprehensions, ordenação |
| 04 | Funções puras, argumentos somente por palavra-chave, geradores, decoradores |
| 05 | Configuração com `@dataclass`, uma classe com estado ajustado |
| 06 | `pathlib`, gerenciadores de contexto, codificações, CSV → Parquet |
| 07 | Exceções personalizadas, logging, validação, `pytest` |

**Tempo estimado:** 120 minutos

<!-- en:56932c2ef1 -->
## Passo 0: gere os dados de origem (propositalmente bagunçados)

Em um trabalho real, isso chega de um sistema anterior. Aqui sintetizamos os dados para o projeto ser autossuficiente e para a bagunça ser reproduzível.

<!-- en:aa26bf6d6f -->
## Passo 1: o contrato

Antes de escrever qualquer transformação, defina o que "limpo" significa. Registrar o contrato em uma `@dataclass` o torna explícito, testável e fácil de mudar em um só lugar.

<!-- en:617f8e16ae -->
## Passo 2: exceções de domínio

Uma classe base compartilhada permite capturar todas as falhas do pipeline de uma vez, sem impedir um tratamento específico onde ele importa (notebook 07).

<!-- en:fb3165b7ba -->
## Passo 3: extração

Leia as origens, valide o formato imediatamente e registre o que aconteceu. Repare que estas funções são a **casca impura**: elas acessam o sistema de arquivos. Toda função depois desta é pura.

<!-- en:c1a5c2fca3 -->
> Repare que todo `amount` é uma string com **vírgula** decimal, e `quantity` também é string. Nada está utilizável ainda, e é exatamente por isso que lemos tudo como `str` em vez de deixar o pandas adivinhar.

<!-- en:c6b0d8981a -->
## Passo 4: transformação

Cada passo abaixo é uma **função pura**: entra um DataFrame, sai um novo DataFrame, sem modificar a entrada e sem entrada e saída. Isso torna cada uma delas testável de forma independente, e esse é o objetivo.

<!-- en:4bce964882 -->
## Passo 5: orquestração

A função do pipeline combina os passos puros e impõe o controle de qualidade.

<!-- en:32d03d136c -->
## Passo 6: o relatório

Um pipeline que ninguém consome não está terminado. Produza a resposta que o negócio realmente pediu.

<!-- en:d9eddadb8d -->
> [!NOTE]
> A mediana fica bem abaixo da média de novo: é a assimetria à direita do notebook 01. Relatar apenas o valor médio do pedido superestimaria como é um pedido típico.

<!-- en:9953ad3d6a -->
## Passo 7: teste

O pipeline só é confiável se suas partes forem testadas. Estes testes cobrem as categorias do notebook 07: respostas conhecidas, casos extremos, ausência de modificação e invariantes.

<!-- en:8d7ee56f21 -->
## Passo 8: verifique se a saída carrega e está correta

A última verificação: leia de volta o que gravamos, exatamente como faria quem consome a tabela.

<!-- en:e32394a468 -->
## Passo 9: prove que o controle de qualidade funciona

Uma rede de segurança que você nunca viu disparar não é uma rede de segurança. Force o disparo.

<!-- en:8898b421cd -->
## Sua vez: extensões

O pipeline funciona. Agora, melhore-o. Cada uma destas é uma exigência real que você receberia no trabalho.

### Extensão 1: promova código para o `dstoolkit` ⭐

Mova `parse_amount` e `clean_text_column` para `src/dstoolkit/cleaning.py`, exporte-as em `__init__.py`, adicione testes em `tests/test_cleaning.py` e importe-as de volta aqui. Este é o ciclo de promoção do notebook 01: faça uma vez, e ele se torna automático.

### Extensão 2: processe a entrada em fluxo ⭐⭐

Reescreva `extract_orders` para processar o CSV em blocos (`chunksize=1000`), de modo que o pipeline consiga lidar com um arquivo maior que a RAM. Verifique se os totais não mudam.

### Extensão 3: um relatório de linhas rejeitadas ⭐⭐

Em vez de descartar linhas ruins sem aviso, grave-as em `data/processed/rejected_rows.parquet` com uma coluna `rejection_reason`. A engenharia de dados precisa disso para corrigir o sistema de origem.

### Extensão 4: idempotência ⭐⭐

Executar o pipeline duas vezes deve produzir uma saída idêntica byte a byte. Verifique isso e use o padrão `atomic_write` do notebook 06, para que uma falha no meio da gravação não possa corromper a tabela publicada.

### Extensão 5: um decorador `@monitored` ⭐⭐⭐

Escreva um decorador que registre o nome de cada passo, a duração e a contagem de linhas de entrada e de saída, e aplique-o a todas as funções de transformação. Você acabou de construir uma versão em miniatura do que Airflow e Dagster fazem.

### Extensão 6: validação de esquema ⭐⭐⭐

Substitua as verificações manuais por um esquema declarativo (um dict de coluna → dtype esperado e restrições) e um único validador genérico. Compare o seu design com o `pandera`, que faz exatamente isso de forma profissional.

<!-- en:1f789849e0 -->
## O que você construiu

Um pipeline que:

- lê origens mal formatadas de forma **defensiva**, declarando explicitamente suas peculiaridades;
- separa uma fina **casca impura** (entrada e saída) de um **núcleo puro** (transformações);
- guarda a configuração em uma **dataclass congelada**, para nada mudar no meio da execução;
- **conta e registra** cada linha descartada, com o motivo;
- **se recusa a publicar** uma saída pela qual não pode responder;
- é **testado**, inclusive nos casos extremos em que moram os bugs de verdade;
- grava **Parquet** tipado e comprimido, que recarrega exatamente como foi gravado.

Esse é um formato realmente profissional para um pipeline de dados. Todos os módulos seguintes do curso se encaixam nele: o Módulo 03 substitui as transformações por pandas mais rico, o Módulo 10 adiciona engenharia de atributos, o Módulo 21 adiciona rastreamento de experimentos e o Módulo 22 coloca um modelo atrás de uma API.

<!-- en:336c8c1c5e -->
## Módulo 01 concluído: autoavaliação

Você está pronto para o Módulo 02 quando conseguir responder a estas perguntas sem consultar nada:

- [ ] Por que `b = a` seguido de `b.append(1)` às vezes altera `a`?
- [ ] Por que `0.1 + 0.2 == 0.3` é `False`, e o que você deve escrever no lugar?
- [ ] Por que `x in my_set` é mais rápido que `x in my_list`, e por quanto?
- [ ] Qual é a diferença entre uma list comprehension e uma expressão geradora?
- [ ] O que `def f(x=[])` faz, e por que é um bug?
- [ ] Quando você usaria `@dataclass` em vez de escrever `__init__`?
- [ ] Por que `except Exception:` costuma estar errado?
- [ ] Por que um pipeline deve usar `logging` em vez de `print`?
- [ ] O que torna o Parquet uma escolha melhor que o CSV para um conjunto de dados de 5 GB?
- [ ] Por que `assert` nunca deve ser usado para validar dados de entrada?

---

**Próximo:** Módulo 02 — NumPy e Computação Numérica. Você vai aprender por que `sum(my_list)` é 50× mais lento que `arr.sum()` e como a vetorização realmente funciona por baixo.

<!-- comments -->
EN: # Defect 1: inconsistent casing and stray whitespace.
PT: # Defeito 1: maiúsculas e minúsculas inconsistentes e espaços soltos.
EN: # Defect 2: some amounts are missing, encoded inconsistently.
PT: # Defeito 2: alguns valores estão ausentes, codificados de forma inconsistente.
EN: # Defect 3: European decimal comma instead of a point.
PT: # Defeito 3: vírgula decimal europeia em vez de ponto.
EN: # Defect 4: a handful of negative quantities (bad returns handling upstream).
PT: # Defeito 4: algumas quantidades negativas (devoluções mal tratadas na origem).
EN: # Defect 5: duplicated rows from retry logic.
PT: # Defeito 5: linhas duplicadas pela lógica de novas tentativas.
EN: # Defect 6: orders referencing customers that do not exist in the CRM.
PT: # Defeito 6: pedidos que citam clientes que não existem no CRM.
EN: # Defect 7: semicolon separator and Latin-1 encoding.
PT: # Defeito 7: separador ponto e vírgula e codificação Latin-1.
EN: # --- The CRM export, as nested JSON ------------------------------------
PT: # --- A exportação do CRM, em JSON aninhado ------------------------------
EN: # Look at the raw bytes before assuming anything. This habit catches encoding and
PT: # Olhe os bytes brutos antes de supor qualquer coisa. Esse hábito pega problemas de
EN: # separator problems in ten seconds instead of thirty minutes.
PT: # codificação e de separador em dez segundos, em vez de trinta minutos.
EN: # Business rules, named rather than scattered as magic numbers.
PT: # Regras de negócio, com nome em vez de espalhadas como números mágicos.
EN: # A run fails outright if more than this share of rows is unusable.
PT: # A execução falha de vez se mais que esta parcela das linhas for inutilizável.
EN: # Read everything as a string. We would rather do the type
PT: # Lê tudo como string. Preferimos fazer a conversão de tipos
EN: # conversion ourselves, explicitly, than let pandas guess and
PT: # nós mesmos, de forma explícita, a deixar o pandas adivinhar e
EN: # silently coerce "N/A" into something unexpected.
PT: # transformar "N/A" sem aviso em algo inesperado.
EN: # json_normalize flattens `profile.segment` → a `profile.segment` column.
PT: # json_normalize achata `profile.segment` → uma coluna `profile.segment`.
EN: # The set of tokens upstream systems use for "nothing here". Membership
PT: # O conjunto de valores que os sistemas de origem usam para "nada aqui". O
EN: # against a set is O(1) — the notebook-03 habit, applied.
PT: # pertencimento em um set é O(1): o hábito do notebook 03, aplicado.
EN: # "1.234,56" → thousands separator is ".", decimal is ",".
PT: # "1.234,56" → o separador de milhar é ".", o decimal é ",".
EN: # Genuinely unparseable — surface it rather than defaulting to 0.
PT: # Realmente impossível de interpretar: expõe o problema em vez de assumir 0.
EN: # Quick check against the cases we care about.
PT: # Verificação rápida com os casos que nos interessam.
EN: # collapse internal whitespace
PT: # reduz os espaços internos
EN: # more aggressive than .lower()
PT: # mais agressivo que .lower()
EN: # never mutate the caller's frame (notebook 02)
PT: # nunca modifica o DataFrame de quem chama (notebook 02)
EN: # `.map` applies our pure parser element-wise.
PT: # `.map` aplica o nosso parser puro elemento a elemento.
EN: # closure over `out` (notebook 04)
PT: # closure sobre `out` (notebook 04)
EN: # O(1) membership (notebook 03)
PT: # pertencimento O(1) (notebook 03)
EN: # raises if `customers` has duplicate ids
PT: # gera um erro se `customers` tiver ids duplicados
EN: # Guard against the classic silent row explosion.
PT: # Protege contra a clássica explosão silenciosa de linhas.
EN: # Downcast to the smallest sensible types — the memory lesson from notebook 06.
PT: # Reduz para os menores tipos sensatos: a lição de memória do notebook 06.
EN: #
PT: #
EN: # The `.astype(str)` before `.astype("category")` is deliberate. Our text
PT: # O `.astype(str)` antes de `.astype("category")` é proposital. A nossa limpeza
EN: # cleaning produced pandas' nullable `string` dtype, so the resulting
PT: # de texto produziu o dtype `string` anulável do pandas, então as categorias
EN: # categories would carry that dtype too — and Parquet normalises it back to
PT: # resultantes também teriam esse dtype, e o Parquet o normaliza de volta para
EN: # plain `str` on read. The round trip would then be *almost* lossless, which
PT: # `str` simples na leitura. A ida e volta ficaria *quase* sem perdas, o que é
EN: # is the worst kind: it passes a glance and fails an equality check.
PT: # o pior caso: passa em uma olhada rápida e falha em uma verificação de igualdade.
EN: # Normalising here makes the written schema exactly what we get back.
PT: # Normalizar aqui faz o esquema gravado ser exatamente o que recebemos de volta.
EN: # --- Extract ---------------------------------------------------------
PT: # --- Extração ---------------------------------------------------------
EN: # --- Transform -------------------------------------------------------
PT: # --- Transformação ----------------------------------------------------
EN: # --- Quality gate ----------------------------------------------------
PT: # --- Controle de qualidade --------------------------------------------
EN: # Fail the run rather than publish a table nobody should trust.
PT: # Falha a execução em vez de publicar uma tabela em que ninguém deveria confiar.
EN: # --- Load ------------------------------------------------------------
PT: # --- Carga ------------------------------------------------------------
EN: # --- Left: monthly revenue by channel --------------------------------------
PT: # --- Esquerda: receita mensal por canal -------------------------------------
EN: # Colour follows the channel (the entity), fixed regardless of rank — so a
PT: # A cor acompanha o canal (a entidade), fixa seja qual for a posição, para que
EN: # filtered view never repaints the survivors.
PT: # uma visão filtrada nunca repinte os que restam.
EN: # --- Right: total revenue share --------------------------------------------
PT: # --- Direita: participação na receita total ---------------------------------
EN: # Direct labels instead of a second legend — the values are the message.
PT: # Rótulos diretos em vez de uma segunda legenda: os valores são a mensagem.
EN: # When a schema check fails, "the dtypes differ" is useless on its own — show
PT: # Quando uma verificação de esquema falha, "os dtypes diferem" sozinho não ajuda:
EN: # exactly which column drifted and how. Build this into your own pipelines.
PT: # mostre exatamente qual coluna mudou e como. Coloque isso nos seus pipelines.
EN: # far stricter than the real reject rate
PT: # muito mais rigoroso que a taxa real de rejeição
