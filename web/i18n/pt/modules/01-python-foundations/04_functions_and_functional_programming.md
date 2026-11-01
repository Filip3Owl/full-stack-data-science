<!-- en:dc4f51c323 -->
# 04 · Funções e Programação Funcional

**Módulo 01 — Fundamentos de Python para Ciência de Dados**

---

## O que você vai aprender

1. Como projetar uma assinatura de função difícil de usar errado.
2. A armadilha do argumento padrão mutável, um bug que sobrevive por anos em código de produção.
3. Type hints e docstrings no estilo NumPy, as convenções que este curso (e o ecossistema científico de Python) usa.
4. Escopo: a regra LEGB, `global`, `nonlocal` e closures.
5. Decoradores: como funcionam e como escrever decoradores úteis.
6. **Geradores**: como processar um arquivo maior que a sua memória RAM, uma linha por vez.
7. `functools` e `itertools`: cache, aplicação parcial e iteração preguiçosa.

**Tempo estimado:** 90 minutos

<!-- en:c017411e1d -->
## 1. Anatomia de uma boa função

Abaixo está o formato que toda função em `src/` segue neste curso. Leia as anotações: cada elemento justifica seu lugar.

<!-- en:50a4dba090 -->
### Por que o `*` na assinatura?

Tudo o que vem depois de um `*` isolado passa a ser **somente por palavra-chave**. Isso transforma uma chamada ilegível em uma que se explica sozinha e, mais importante, permite reordenar ou inserir parâmetros depois sem quebrar nenhum código que chama a função.

```python
normalise(data, "minmax", 0)          # ✘ impossível: o que são esses argumentos?
normalise(data, method="minmax", ddof=0)   # ✔ óbvio no ponto da chamada
```

<!-- en:8372c8fc26 -->
> [!TIP]
> Uma regra útil: **todo parâmetro booleano ou de "modo" deve ser somente por palavra-chave.** Uma chamada como `f(df, True, False)` é ilegível na hora da revisão.

<!-- en:941ca29ed1 -->
## 2. Parâmetros por completo

O sistema de parâmetros de Python é excepcionalmente expressivo. Este é o quadro completo:

<!-- en:9d05e00140 -->
### `*args` e `**kwargs` na prática

O uso legítimo mais comum é repassar argumentos para outra função sem precisar repetir toda a assinatura dela.

<!-- en:1b69b8286b -->
## 3. A armadilha do argumento padrão mutável

Esta é a pegadinha mais famosa da linguagem, e ela decorre diretamente do modelo de objetos do notebook 02.

**Argumentos padrão são avaliados uma única vez, quando a função é definida**, e não a cada chamada. Portanto, um padrão mutável é compartilhado por todas as chamadas, para sempre.

<!-- en:85dab012f3 -->
### A solução: `None` como sentinela

<!-- en:c27b01c21b -->
> [!WARNING]
> A mesma armadilha vale para `{}`, `set()` e qualquer objeto que você construa na assinatura, incluindo `pd.DataFrame()` e `datetime.now()`. A regra: **nunca coloque um valor mutável ou dependente do tempo em um padrão.**

<!-- en:cd609cbf20 -->
## 4. Escopo: a regra LEGB

Quando Python encontra um nome, ele procura em quatro espaços de nomes, em uma ordem fixa, e para na primeira ocorrência.

![A regra LEGB](../../../assets/images/legb_scope.png)

<!-- en:6be65fde12 -->
### A atribuição cria um nome local, o que causa um erro clássico

<!-- en:0360dbd654 -->
> [!TIP]
> Quase nunca recorra a `global`. Uma função cujo resultado depende só dos argumentos pode ser testada isoladamente; uma que lê e grava o estado do módulo, não.

<!-- en:390c1655ce -->
### Closures: funções que lembram

Uma função interna mantém uma referência viva às variáveis do escopo envolvente, mesmo depois que a função externa retornou. Esse ambiente capturado é uma **closure**.

<!-- en:b932f4eea0 -->
## 5. Decoradores

Um decorador é uma função que recebe uma função e devolve uma substituta. `@decorator` acima de um `def` é exatamente `f = decorator(f)`.

Você vai encontrá-los o tempo todo: `@property`, `@staticmethod`, `@pytest.fixture`, `@app.get` no FastAPI (Módulo 22), `@task` no Airflow.

<!-- en:ad3e639d44 -->
### Um decorador que recebe argumentos

Isso exige mais uma camada: uma função que devolve um decorador que devolve um wrapper. Um decorador de nova tentativa é um exemplo realista: chamadas de rede instáveis a uma API de dados fazem parte do dia a dia.

<!-- en:eb1ccf7986 -->
### `functools.lru_cache`: memoização de graça

Guarda os resultados em cache por argumento. Ideal para funções caras, determinísticas e chamadas repetidamente, como buscas de atributos ou cálculos recursivos.

<!-- en:5d2f504f93 -->
> [!WARNING]
> `lru_cache` exige argumentos **hasheáveis**, então não consegue guardar o cache de uma função que recebe um DataFrame ou uma lista. Ele também mantém os resultados na memória: defina `maxsize` deliberadamente em vez de usar `None` em um trecho muito executado.

<!-- en:3190b71843 -->
## 6. Geradores: a ferramenta de memória

Um gerador produz valores **um de cada vez, sob demanda**, em vez de construir uma coleção inteira na memória. Em ciência de dados, essa é a diferença entre processar um arquivo de 50 GB e receber um `MemoryError`.

Qualquer função que contenha `yield` é uma função geradora.

<!-- en:7f21f27e3f -->
### Como medir a diferença

<!-- en:1eb21c52d1 -->
### O uso realista: ler um arquivo em fluxo

Este é o padrão para qualquer arquivo grande demais para carregar. Repare que toda a cadeia é preguiçosa: nada é lido até que o `for` final comece a puxar os dados.

<!-- en:12b5d8172a -->
### Expressões geradoras

Idênticas a uma list comprehension, mas com parênteses. Prefira-as sempre que estiver alimentando um consumidor como `sum`, `max`, `any` ou `all`.

<!-- en:824df7290f -->
> [!WARNING]
> Um gerador é de **uso único**. Depois de esgotado, ele não produz mais nada, e faz isso sem aviso, o que torna esse bug bem traiçoeiro.

<!-- en:02867d46d5 -->
## 7. `map`, `filter`, `reduce` e `lambda`

<!-- en:24ef95d3c1 -->
### `functools.partial`: argumentos pré-preenchidos

Cria uma nova função com alguns argumentos já fornecidos. É muito útil para construir pequenas transformações nomeadas e passá-las adiante.

<!-- en:6d5c12f681 -->
## 8. `itertools`: ferramentas de iteração preguiçosa

<!-- en:741b930f3a -->
> [!TIP]
> `itertools.product` é exatamente como uma busca em grade de hiperparâmetros enumera seus candidatos. Você vai revê-lo, embutido no `GridSearchCV`, no Módulo 13.

<!-- en:fa31f6e588 -->
## 9. Funções puras e efeitos colaterais

Uma função **pura** devolve a mesma saída para a mesma entrada e não altera nada fora de si mesma. Funções puras são muito fáceis de testar, seguras para paralelizar e fáceis de colocar em cache.

<!-- en:4fc8a188ae -->
Nem tudo pode ser puro: ler arquivos, gravar em um banco de dados e registrar logs são efeitos colaterais, e são a razão de ser do programa. O padrão profissional é **empurrar os efeitos colaterais para as bordas**: uma casca fina e impura que faz a entrada e saída, em torno de um núcleo puro que faz os cálculos.

<!-- en:28393d591e -->
## 10. Exercícios

### Exercício 4.1: conserte a assinatura

```python
def summarise(data, True, 0.95, None):
    ...
```

Reescreva isto como uma assinatura bem projetada, com opções somente por palavra-chave, valores padrão, type hints e uma docstring no estilo NumPy. A função deve calcular estatísticas-resumo com um nível de confiança opcional e um subconjunto opcional de colunas.

### Exercício 4.2: encontre a armadilha

```python
def add_feature(df_columns, new_cols=[]):
    new_cols.append("engineered")
    return df_columns + new_cols
```

Chame-a três vezes só com o primeiro argumento. Explique a saída e depois corrija a função.

### Exercício 4.3: escreva um decorador

Escreva `@validate_positive`, que gera `ValueError` se algum argumento posicional *numérico* passado à função decorada for negativo. Preserve os metadados da função original.

### Exercício 4.4: pipeline com geradores

Escreva três geradores encadeados para processar linhas de log:

1. `parse(lines)` → dicts com `level` e `message`;
2. `only_errors(records)` → mantém `level == "ERROR"`;
3. `take(records, n)` → produz no máximo `n` registros.

Verifique com `sys.getsizeof` que a memória não cresce com o tamanho da entrada.

### Exercício 4.5: contador com closure

Escreva `make_counter()`, que devolve uma função que produz 1, 2, 3, … em chamadas sucessivas, sem usar `global`. *Dica:* `nonlocal`.

<!-- en:7eff543f67 -->
## Principais lições

- Torne parâmetros de "modo" e booleanos **somente por palavra-chave** com `*`.
- **Nunca use um argumento padrão mutável.** Use `None` como sentinela.
- Type hints e docstrings no estilo NumPy são as convenções do ecossistema: siga-as.
- Nomes são resolvidos pela regra **LEGB**; a atribuição dentro de uma função torna o nome local.
- Um **decorador** envolve uma função; sempre aplique `functools.wraps`.
- `lru_cache` transforma uma função determinística cara em uma simples consulta.
- **Geradores trocam tempo por memória**: o consumo é constante, seja qual for o tamanho da entrada, e eles são de uso único.
- Prefira **funções puras**; empurre entrada e saída e mutações para as bordas do programa.

---

**Próximo:** [05 · Programação Orientada a Objetos](05_object_oriented_programming.ipynb), e como escrever uma classe que se encaixa direto em um pipeline do scikit-learn.

<!-- comments -->
EN: # Validate early and fail loudly. A function that returns nonsense for bad
PT: # Valide cedo e falhe de forma clara. Uma função que devolve absurdos para uma entrada
EN: # input is far more expensive to debug than one that refuses it.
PT: # ruim custa muito mais para depurar do que uma que a recusa.
EN: # Naming the valid options in the message saves the caller a trip to the docs.
PT: # Citar as opções válidas na mensagem poupa quem chama de ir até a documentação.
EN: # positional — now rejected
PT: # posicional: agora é rejeitado
EN: # Caller-supplied kwargs come last, so they win over our defaults.
PT: # Os kwargs de quem chama vêm por último, então prevalecem sobre os nossos padrões.
EN: # The shared object is visible on the function itself.
PT: # O objeto compartilhado fica visível na própria função.
EN: # GLOBAL
PT: # GLOBAL
EN: # ENCLOSING (from inner's point of view)
PT: # ENVOLVENTE (do ponto de vista de inner)
EN: # LOCAL
PT: # LOCAL
EN: # `base` found locally, `factor` in the enclosing scope,
PT: # `base` é encontrado localmente, `factor` no escopo envolvente,
EN: # `threshold` globally, `max` in built-ins.
PT: # `threshold` no global, `max` nas funções embutidas.
EN: # Python sees an assignment → treats `counter` as local
PT: # Python vê uma atribuição → trata `counter` como local
EN: # `threshold` comes from the enclosing scope
PT: # `threshold` vem do escopo envolvente
EN: # call the original
PT: # chama a função original
EN: # noqa: BLE001 — deliberate here
PT: # noqa: BLE001 — proposital aqui
EN: # A function that fails the first two times it is called.
PT: # Uma função que falha nas duas primeiras vezes em que é chamada.
EN: # A list comprehension materialises everything…
PT: # Uma list comprehension materializa tudo…
EN: # …while a generator expression (round brackets) does not.
PT: # …enquanto uma expressão geradora (parênteses) não.
EN: # Stand-in for a file object — a real one iterates line by line the same way.
PT: # Substituto de um objeto de arquivo; um real percorre linha a linha do mesmo jeito.
EN: # Build the pipeline. No work has happened yet.
PT: # Monta o pipeline. Nenhum trabalho aconteceu ainda.
EN: # Only now does data start flowing, one record at a time.
PT: # Só agora os dados começam a fluir, um registro por vez.
EN: # ✘ Builds a 1M-element list, then sums it and throws the list away.
PT: # ✘ Constrói uma lista de 1 milhão de elementos, soma e joga a lista fora.
EN: # ✔ Streams values into sum, holding one at a time.
PT: # ✔ Passa os valores em fluxo para sum, guardando um de cada vez.
EN: # `any` and `all` also short-circuit: they stop at the first decisive element.
PT: # `any` e `all` também interrompem cedo: param no primeiro elemento decisivo.
EN: # `lambda` is a one-expression anonymous function. Use it only for short callables
PT: # `lambda` é uma função anônima de uma só expressão. Use-a apenas em funções curtas
EN: # passed to another function; anything longer deserves a `def` and a name.
PT: # passadas a outra função; qualquer coisa mais longa merece um `def` e um nome.
EN: # In idiomatic Python, comprehensions usually read better than map/filter:
PT: # Em Python idiomático, comprehensions costumam ser mais legíveis que map/filter:
EN: # chain: concatenate iterables without building a combined list.
PT: # chain: concatena iteráveis sem construir uma lista combinada.
EN: # islice: slice any iterable, including an infinite or streaming one.
PT: # islice: fatia qualquer iterável, inclusive um infinito ou em fluxo.
EN: # groupby: group CONSECUTIVE equal items — so sort by the key first!
PT: # groupby: agrupa itens iguais CONSECUTIVOS, então ordene pela chave antes!
EN: # combinations / product: for grid searches and pairwise comparisons.
PT: # combinations / product: para buscas em grade e comparações par a par.
EN: # ✘ Impure: mutates its argument and depends on module state.
PT: # ✘ Impura: modifica o argumento e depende do estado do módulo.
EN: # ✔ Pure: everything it needs comes in; everything it produces comes out.
PT: # ✔ Pura: tudo de que precisa entra; tudo o que produz sai.
