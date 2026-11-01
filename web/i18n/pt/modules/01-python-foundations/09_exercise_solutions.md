<!-- en:a1b412ebaa -->
# 09 · Soluções dos Exercícios

**Módulo 01 — Fundamentos de Python para Ciência de Dados**

---

> [!WARNING]
> **Tente cada exercício antes de ler a solução.** Ler uma solução pronta produz uma sensação de entendimento que desaparece no momento em que você encara uma célula em branco. Dez minutos de esforço ensinam mais do que sessenta de leitura.

Quando uma solução tem mais de uma forma razoável, as alternativas são mostradas e os prós e contras são explicados, porque no trabalho a pergunta interessante raramente é "funciona?", e sim "qual versão eu gostaria de manter?"

<!-- en:3711aa869e -->
---
# Notebook 01 — Introdução ao Curso e Configuração

<!-- en:8d405cc57a -->
## Exercício 1.1: auditoria do ambiente

<!-- en:2c63f90a34 -->
## Exercício 1.2: reproduza com outra seed

O objetivo deste exercício é desenvolver a intuição sobre **variabilidade amostral**: uma única amostra é um sorteio dentro de uma distribuição de amostras possíveis.

<!-- en:32b5acab78 -->
**Respostas**

- **As médias mudam?** Sim, em vários dólares de uma seed para outra.
- **A conclusão muda?** Não. O gasto orgânico supera o pago em todas as amostras, e a diferença mantém o mesmo sinal e aproximadamente a mesma magnitude.
- **O que isso diz?** Uma única amostra dá uma estimativa *ruidosa* da quantidade que importa. A direção é estável aqui porque o efeito real é grande em relação ao ruído. Quando um efeito é pequeno, uma única amostra pode facilmente apontar para o lado errado, e é justamente por isso que o Módulo 08 ensina intervalos de confiança em vez de estimativas pontuais.

<!-- en:eac92fb726 -->
## Exercício 1.3: quebre o notebook de propósito

**O que acontece:** `summary` foi calculado a partir do `customers` de 2 000 linhas. Depois de reconstruir `customers` com 50 linhas, `summary` continua com o resultado *antigo*: é um objeto separado, não uma visão viva. O notebook agora mostra estatísticas de dados que já não estão na memória.

**O comando de menu:** *Kernel → Restart Kernel and Run All Cells*. Ele reconstrói todas as variáveis a partir de um estado limpo, na ordem do documento, que é a única forma de ter certeza de que o que você vê corresponde ao que o código diz.

<!-- en:94fee3c94b -->
## Exercício 1.4: mapeamento do ciclo de vida (exemplo: um modelo de previsão de entrega)

| Etapa | O que a equipe precisou fazer |
| :--- | :--- |
| 1. Entendimento do negócio | Decidir se o objetivo é precisão ou *não atrasar*: os dois implicam funções de perda diferentes. |
| 2. Coleta de dados | Juntar viagens históricas, trajetos de GPS, tempos de preparo dos restaurantes e clima. |
| 3. Limpeza | Tratar falhas de GPS, pedidos cancelados durante a entrega e diferenças de relógio entre sistemas. |
| 4. Análise exploratória | Descobrir que o erro da previsão é muito pior em horários de pico e em centros urbanos densos. |
| 5. Engenharia de atributos | Criar atributos para hora do dia, carga do entregador, distância e atraso histórico do restaurante. |
| 6. Modelagem | Gradient boosting com atributos tabulares; possivelmente um modelo de quantis para mirar o atraso. |
| 7. Avaliação | Otimizar uma perda de quantil, não o MAE: superestimar em 5 minutos custa muito menos que subestimar em 5 minutos. |
| 8. Implantação e monitoramento | Servir com latência abaixo de 100 ms; monitorar o drift quando uma nova cidade entrar em operação. |

<!-- en:4e3261c0aa -->
---
# Notebook 02 — Fundamentos de Python

<!-- en:ef87ff3dca -->
## Exercício 2.1: preveja antes de executar

<!-- en:396388a17d -->
**Por que (a) e (b) diferem**

- `b = b + [4]` chama `__add__`, que **constrói uma nova lista** e religa o nome `b` a ela. `a` continua apontando para a original.
- `b += [4]` chama `__iadd__`, que em uma lista é **`extend` no lugar**. Os dois nomes continuam apontando para o único objeto modificado.

Portanto, `+=` *não* é só uma forma curta de `x = x + y` quando o objeto é mutável: é uma operação realmente diferente. Em tipos imutáveis (`int`, `str`, `tuple`) não existe opção no lugar, então `+=` recorre à religação, e por isso a diferença fica invisível até você encontrar uma lista.

**Por que (c) funciona**

Uma tupla congela suas *referências*, não os objetos para os quais elas apontam. `t[0]` é uma lista, e a lista é mutável. O que a tupla proíbe é `t[0] = something`, que mudaria o objeto ao qual a posição se refere.

<!-- en:2c13d5c574 -->
## Exercício 2.2: um bug de comparação de floats

<!-- en:d25c5df5d3 -->
**O bug:** `0.1 + 0.2 + 0.3 + 0.4` acumula quatro erros de representação e termina a algumas vezes 10⁻¹⁶ de distância de exatamente 1.0. Usar `==` em floats calculados está quase sempre errado.

> Especificamente para dinheiro, a melhor resposta é `Decimal`, mas para *pesos* uma tolerância é a ferramenta certa.

<!-- en:fca5db1fe4 -->
## Exercício 2.3: enriquecimento seguro

<!-- en:0a37803192 -->
## Exercício 2.4: limpeza de texto

<!-- en:eee3a6c3fd -->
> [!NOTE]
> `.title()` é imperfeito para nomes reais: ele produz `"Rio De Janeiro"`, quando o português pede `"de"` em minúscula. Tratar isso direito exige uma lista de partículas que ficam em minúscula. O objetivo do exercício é a expressão `" ".join(raw.split())`; a nuance dos nomes é um bom lembrete de que normalizar texto sempre depende do idioma e da região.

<!-- en:55e36099f3 -->
## Exercício 2.5: auditoria de valores ausentes

<!-- en:0b60cc2c95 -->
**Por que `.count(np.nan)` não é confiável**

`list.count` compara com `==`, e `nan == nan` é `False` pelo IEEE 754. Aqui ele devolve 1 apenas porque `count` verifica primeiro a **identidade** (`x is nan or x == nan`), e esta lista por acaso guarda o *mesmo* objeto `np.nan`. Construa a lista de outra forma, por exemplo com valores `float("nan")`, que são objetos distintos, e a contagem vira 0 sem aviso.

<!-- en:79b53daff8 -->
---
# Notebook 03 — Estruturas de Dados

<!-- en:35dc1d3ac7 -->
## Exercício 3.1: escolha a estrutura

| # | Requisito | Escolha | Por quê |
| :-: | :--- | :--- | :--- |
| 1 | 50 milhões de IDs de usuário processados, para pular duplicatas | **`set`** | Pertencimento O(1); uma lista faria cada verificação percorrer 50 milhões de elementos. |
| 2 | Últimas 100 leituras de sensor para uma média móvel | **`collections.deque(maxlen=100)`** | O(1) nas duas pontas, e `maxlen` descarta os itens antigos automaticamente. |
| 3 | Código de país → nome do país | **`dict`** | É exatamente um mapeamento chave-valor, com busca O(1). |
| 4 | Uma cor RGB que nunca pode mudar | **`tuple`** | Imutável, hasheável, e a posição tem significado. |
| 5 | Frequência de palavras em 10 000 avaliações | **`collections.Counter`** | Feito para isso; `.most_common()` já vem incluído. |

<!-- en:9cdc897609 -->
## Exercício 3.2: conserte a função lenta

<!-- en:2382457ab9 -->
## Exercício 3.3: prática de comprehensions

<!-- en:65030a0af3 -->
## Exercício 3.4: agrupe sem pandas

<!-- en:8d927f40f3 -->
## Exercício 3.5: ordenação por várias chaves

<!-- en:5d8bb83d5f -->
> [!TIP]
> Inverter o sinal só funciona para números. Para ordenar um campo de *texto* em ordem decrescente enquanto os outros ficam em ordem crescente, faça uma ordenação estável em várias passadas (notebook 03, seção 9): ordene primeiro pela chave menos significativa e depois pela mais significativa.

<!-- en:fa21516546 -->
---
# Notebook 04 — Funções e Programação Funcional

<!-- en:10f6c55b8c -->
## Exercício 4.1: conserte a assinatura

<!-- en:0769f073d9 -->
**O que mudou e por quê**

- **Opções somente por palavra-chave** (`*`): `summarise(df, True, 0.95, None)` era ilegível e quebraria se a ordem dos parâmetros mudasse.
- **Type hints** documentam o contrato e permitem que editores peguem erros.
- **`columns: list[str] | None = None`**: sentinela `None` em vez de um padrão mutável.
- **Validação logo no início**, com mensagens informativas.
- **Uma docstring** que diz quais erros a função gera, para que quem chama possa tratar as falhas.

<!-- en:9a919b3dd4 -->
## Exercício 4.2: encontre a armadilha

<!-- en:5d2fa989cd -->
**O que está acontecendo:** a lista padrão é criada **uma única vez**, quando a instrução `def` é executada. Toda chamada que não passa `new_cols` adiciona itens a essa mesma lista, que só cresce de uma chamada para outra.

<!-- en:47a41f30a0 -->
## Exercício 4.3: escreva um decorador

<!-- en:c050157178 -->
## Exercício 4.4: pipeline com geradores

<!-- en:6ffe38b2d8 -->
## Exercício 4.5: contador com closure

<!-- en:edffd54363 -->
---
# Notebook 05 — Programação Orientada a Objetos

<!-- en:23035b4036 -->
## Exercício 5.1: classe ou função?

| # | Tarefa | Escolha | Por quê |
| :-: | :--- | :--- | :--- |
| 1 | Celsius → Fahrenheit | **função** | Sem estado, uma entrada, uma saída. Uma classe seria pura cerimônia. |
| 2 | Configurações de conexão do banco de dados | **`@dataclass` congelada** | Um conjunto fixo de campos, sem comportamento, que não pode mudar no meio da execução. |
| 3 | Tokenizador que aprende um vocabulário e depois codifica | **classe** | Tem *estado ajustado* (o vocabulário) do qual `encode` depende: é o caso típico. |
| 4 | Mediana de uma lista | **função** | Cálculo sem estado. |

<!-- en:4f2c6b7bae -->
## Exercício 5.2: prática de métodos dunder

<!-- en:29645588a3 -->
## Exercício 5.3: conversão para dataclass

<!-- en:3f567ead57 -->
**Linhas eliminadas:** `__init__` (3 linhas), e ganhamos `__repr__` e `__eq__`, que a versão original nunca teve: cerca de 15 linhas de código equivalente escrito à mão.

> [!WARNING]
> Uma diferença de comportamento: a original fazia `self.values = list(values)`, copiando defensivamente a lista de quem chamava. A dataclass guarda a referência recebida. Se isso importar, adicione um `__post_init__` que faça a cópia.

<!-- en:6ffe9ce02e -->
## Exercício 5.4: construa um estimador

<!-- en:d402327278 -->
## Exercício 5.5: encontre o bug

<!-- en:3e852908da -->
**Explicação:** `history` é definido no corpo da classe, então existe exatamente uma lista, e ela pertence ao próprio `Model`. `self.history.append(...)` não atribui nada a `self`: ele *procura* `history` (encontrando o atributo de classe) e o modifica. Assim, as duas instâncias compartilham uma única lista crescente, que persiste durante toda a vida do processo.

É a lição de mutabilidade do notebook 02 com uma roupa de classe.

<!-- en:1e1a5e6ba2 -->
---
# Notebook 06 — Arquivos e Formatos de Dados

<!-- en:21f8bd95b2 -->
## Exercício 6.1: manipulação de caminhos

<!-- en:49266fda2a -->
## Exercício 6.2: diagnostique um CSV quebrado

<!-- en:06689c0b95 -->
> [!WARNING]
> Sem `date_format`, `15/03/2024` pode ser lido como *15 de março* ou falhar completamente, dependendo da versão do pandas, e `05/03/2024` viraria sem aviso **3 de maio** em vez de 5 de março. Sempre informe o formato de datas fora do padrão ISO.

<!-- en:7c5e41decd -->
## Exercício 6.3: JSON aninhado

<!-- en:d977cd9e6d -->
## Exercício 6.4: comprove a seleção de colunas

<!-- en:cbab721b71 -->
**A explicação.** O CSV é orientado a linhas: os valores de todas as colunas ficam intercalados em cada linha, então o leitor precisa percorrer e interpretar os bytes do arquivo inteiro só para encontrar a coluna que quer. O Parquet guarda cada coluna como um bloco contíguo separado, com um índice no rodapé, então o leitor salta direto para aquele bloco e não toca em mais nada. A economia é proporcional ao número de colunas que você ignora.

<!-- en:a3507c009b -->
## Exercício 6.5: gravação segura com backup

<!-- en:3f99a5e68c -->
---
# Notebook 07 — Erros, Logging e Testes

<!-- en:f01f2be543 -->
## Exercício 7.1: restrinja o except

**Três falhas distintas que o `except:` sem classe esconde:**

1. `FileNotFoundError`: o caminho está errado. Quem chama pode se recuperar disso, e a resposta precisa ser completamente diferente da de um problema de interpretação.
2. `KeyError` / `pd.errors.ParserError`: o arquivo existe, mas tem as colunas erradas ou está malformado. É um problema de esquema para a engenharia de dados.
3. `ValueError` vindo de `astype(float)`: uma célula específica contém lixo. É um problema de qualidade de dados que vale reportar por linha.

Pior: um `except:` sem classe também engole `KeyboardInterrupt`, então o processo não pode ser interrompido, e devolver um DataFrame vazio faz com que todas as agregações seguintes virem zero sem nenhum aviso.

<!-- en:0c4f116b2f -->
## Exercício 7.2: hierarquia de exceções personalizada

<!-- en:85117a447b -->
## Exercício 7.3: instrumente uma função

<!-- en:c2ca630828 -->
## Exercício 7.4: escreva os testes primeiro

Os testes definem o contrato, incluindo as decisões que o enunciado deixou em aberto: o que um denominador zero deve produzir? E uma série vazia? Responder a isso *antes* de implementar é todo o valor do design guiado por testes.

<!-- en:e70aab4451 -->
> [!NOTE]
> `test_negative_values` documenta uma escolha realmente discutível. Passar de −100 para −50 é uma *melhora*, mas a fórmula informa −50%. A variação percentual fica mal definida quando o sinal muda; o certo é registrar o comportamento em um teste, em vez de deixar cada pessoa que chama a função descobri-lo por conta própria.

<!-- en:6bf3e528c2 -->
## Exercício 7.5: capture um bug real

<!-- en:6a411ccaa1 -->
**A lição.** `validate="one_to_one"` (ou `"many_to_one"`) transforma uma explosão de linhas silenciosa e catastrófica em um `MergeError` claro, no ponto do erro. Adicione-o a todo `merge` em que você acredita que as chaves são únicas: custa um argumento e já pegou mais bugs de produção que qualquer outro hábito isolado de pandas.

A regra geral: **verifique o shape esperado depois de cada join.**

<!-- en:f02f53c84c -->
---

## Módulo 01 concluído

Se você fez os exercícios em vez de só ler as soluções, agora tem os fundamentos sobre os quais o resto do curso é construído.

**Próximo:** Módulo 02 — NumPy e Computação Numérica.

<!-- comments -->
EN: # `getattr` with a default keeps this working for the occasional package
PT: # `getattr` com um padrão mantém isto funcionando para o pacote eventual
EN: # that does not expose __version__.
PT: # que não expõe __version__.
EN: # --- (a) b = b + [4] --------------------------------------------------------
PT: # --- (a) b = b + [4] --------------------------------------------------------
EN: # --- (b) b += [4] -----------------------------------------------------------
PT: # --- (b) b += [4] -----------------------------------------------------------
EN: # --- (c) mutating a list inside a tuple -------------------------------------
PT: # --- (c) modificar uma lista dentro de uma tupla ----------------------------
EN: # The identity check makes the difference concrete.
PT: # A verificação de identidade torna a diferença concreta.
EN: # It must still reject genuinely wrong input.
PT: # Ela ainda precisa rejeitar uma entrada realmente errada.
EN: # `{**order, ...}` builds a NEW dict — the key to not mutating the input.
PT: # `{**order, ...}` constrói um NOVO dict: é a chave para não modificar a entrada.
EN: # `.split()` with no argument splits on ANY run of whitespace (including
PT: # `.split()` sem argumento separa em QUALQUER sequência de espaços (inclusive
EN: # newlines and tabs) and discards empties — so join+split collapses
PT: # quebras de linha e tabulações) e descarta vazios, então join+split reduz
EN: # everything in one step.
PT: # tudo em um só passo.
EN: # The first word is always capitalised, even if it is a particle.
PT: # A primeira palavra sempre fica com inicial maiúscula, mesmo sendo uma partícula.
EN: # `x != x` is True only for NaN — the defining property from notebook 02.
PT: # `x != x` só é True para NaN: a propriedade definidora do notebook 02.
EN: # Why .count() fails:
PT: # Por que .count() falha:
EN: # Proof that the behaviour depends on object identity, not value:
PT: # Prova de que o comportamento depende da identidade do objeto, não do valor:
EN: # one object, repeated
PT: # um objeto, repetido
EN: # three separate objects
PT: # três objetos separados
EN: # The reliable ways:
PT: # Os jeitos confiáveis:
EN: # the entire fix is this line
PT: # a correção inteira é esta linha
EN: # Deliberately modest sizes so the slow version still finishes in this notebook.
PT: # Tamanhos propositalmente modestos, para a versão lenta ainda terminar neste notebook.
EN: # 1. A list of revenues.
PT: # 1. Uma lista de receitas.
EN: # 2. A dict mapping product → revenue.
PT: # 2. Um dict que associa produto → receita.
EN: # 3. The set of products with revenue above 100.
PT: # 3. O set de produtos com receita acima de 100.
EN: # Return a plain dict: a defaultdict silently creates entries on lookup,
PT: # Devolve um dict simples: um defaultdict cria entradas sem aviso na consulta,
EN: # which surprises callers who expect a KeyError for a missing group.
PT: # o que surpreende quem espera um KeyError para um grupo inexistente.
EN: # It agrees with pandas, which is the point.
PT: # Concorda com o pandas, e esse é o objetivo.
EN: # A single sorted() call. Negating the salary flips just that field to descending
PT: # Uma única chamada a sorted(). Inverter o sinal do salário deixa só esse campo decrescente,
EN: # while the others stay ascending — the standard trick for mixed-direction sorts
PT: # enquanto os outros ficam crescentes: o truque padrão para ordenar em direções
EN: # on numeric keys.
PT: # mistas com chaves numéricas.
EN: # Standard error of the mean, then a t-based interval.
PT: # Erro padrão da média, e depois um intervalo baseado em t.
EN: # preserve __name__, __doc__, signature
PT: # preserva __name__, __doc__ e a assinatura
EN: # `bool` is a subclass of int — exclude it, since False would
PT: # `bool` é uma subclasse de int; é excluído, porque False seria
EN: # otherwise be a perfectly valid "non-negative" flag.
PT: # um indicador "não negativo" perfeitamente válido.
EN: # Memory proof: build the chain over 10 million lines and measure it.
PT: # Prova de memória: monta a cadeia sobre 10 milhões de linhas e mede.
EN: # Makes `2 * m` work as well as `m * 2`.
PT: # Faz `2 * m` funcionar tanto quanto `m * 2`.
EN: # Restore the defensive copy the hand-written __init__ had.
PT: # Restaura a cópia defensiva que o __init__ escrito à mão tinha.
EN: # scikit-learn contract: store arguments unchanged, under the same names.
PT: # Contrato do scikit-learn: guarda os argumentos sem alteração, com os mesmos nomes.
EN: # A set gives O(1) membership in transform (notebook 03).
PT: # Um set dá pertencimento O(1) em transform (notebook 03).
EN: # A realistic long tail: 3 common categories, 60 rare ones.
PT: # Uma cauda longa realista: 3 categorias comuns, 60 raras.
EN: # The payoff: it composes with the rest of scikit-learn.
PT: # A recompensa: ela se combina com o resto do scikit-learn.
EN: # BUG: a class attribute, shared by every instance
PT: # BUG: um atributo de classe, compartilhado por todas as instâncias
EN: # a fresh list per instance
PT: # uma lista nova por instância
EN: # 1. Move to processed/ and change the extension.
PT: # 1. Move para processed/ e muda a extensão.
EN: # 2. Extract year and quarter from the stem.
PT: # 2. Extrai o ano e o trimestre do stem.
EN: # 3. A sibling backup path.
PT: # 3. Um caminho irmão para o backup.
EN: # semicolon separator
PT: # separador ponto e vírgula
EN: # keep the leading zeros
PT: # mantém os zeros à esquerda
EN: # comma is the decimal mark
PT: # a vírgula é o separador decimal
EN: # dot is the thousands separator
PT: # o ponto é o separador de milhar
EN: # day-first, or 15/03 would be misread
PT: # dia primeiro, ou 15/03 seria mal interpretado
EN: # one row per item
PT: # uma linha por item
EN: # scaled down so the notebook stays quick
PT: # reduzido para o notebook continuar rápido
EN: # Take the backup only once we know the new content is complete.
PT: # Só faz o backup quando sabemos que o novo conteúdo está completo.
EN: # A failure leaves both the current file and the backup untouched.
PT: # Uma falha deixa intactos tanto o arquivo atual quanto o backup.
EN: # `errors="coerce"` turns unparseable values into NaT/NaN rather than
PT: # `errors="coerce"` transforma valores impossíveis de interpretar em NaT/NaN em vez
EN: # raising — which is right ONLY because we then count and report them.
PT: # de gerar um erro, o que só está certo PORQUE depois os contamos e reportamos.
EN: # Demonstrate each failure mode.
PT: # Demonstra cada modo de falha.
EN: # The payoff: one handler can route by retryability.
PT: # A recompensa: um único tratador pode decidir com base na possibilidade de nova tentativa.
EN: # Clean up the temporary directory used by the exercises above.
PT: # Limpa o diretório temporário usado pelos exercícios acima.
