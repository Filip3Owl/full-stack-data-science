<!-- en:3ddbe7474f -->
# 03 · Estruturas de Dados

**Módulo 01 — Fundamentos de Python para Ciência de Dados**

---

## O que você vai aprender

1. As quatro coleções embutidas, e como escolher entre elas **pela complexidade**, não pelo hábito.
2. Por que `x in my_set` é milhares de vezes mais rápido que `x in my_list`: medido, não apenas afirmado.
3. O módulo `collections`: `Counter`, `defaultdict`, `deque`, `namedtuple`.
4. Comprehensions, a expressão idiomática que substitui a maioria dos seus loops `for`.
5. Ordenação com funções-chave, e por que a estabilidade da ordenação importa no trabalho com dados.
6. Desempacotamento e expressões com asterisco.

**Tempo estimado:** 90 minutos

<!-- en:b0cea20804 -->
## 1. O mapa

Antes dos detalhes, o resumo que você deve guardar. Quase todo problema de desempenho no Python do dia a dia vem de uma leitura errada da linha "Membership" (pertencimento) desta tabela.

![Como escolher uma coleção embutida](../../../assets/images/data_structures_comparison.png)

<!-- en:659c1e9e0f -->
## 2. Listas: o burro de carga ordenado

Uma `list` é uma sequência mutável e ordenada. Internamente, ela é um **array de ponteiros** para objetos, o que explica todas as suas características de desempenho.

<!-- en:836b03c6db -->
### Fatiamento: `[start:stop:step]`

`stop` é sempre **excluído**. Isso é proposital: faz com que `len(x[a:b]) == b - a` e com que `x[:n] + x[n:] == x` seja verdadeiro para qualquer `n`.

<!-- en:83f272219b -->
### Os métodos que vale conhecer

> [!WARNING]
> Métodos que modificam a lista devolvem `None`, não a lista. `sorted_list = my_list.sort()` atribui `None`, um erro que quase todo mundo comete pelo menos uma vez.

<!-- en:5d78dd45d7 -->
## 3. Tuplas: imutáveis, e mais que "uma lista somente leitura"

Tuplas têm dois usos distintos, e misturá-los causa confusão:

1. **Registros**: um conjunto fixo de campos em que *a posição tem significado* (`(lat, lon)`).
2. **Imutabilidade**: quando você precisa de uma sequência que possa ser chave de um `dict` ou membro de um `set`.

<!-- en:e8a74c1a08 -->
> [!NOTE]
> **Imutável não significa "congelado por inteiro".** Uma tupla garante que suas *referências* nunca mudam; ela não diz nada sobre os objetos para os quais essas referências apontam.

<!-- en:d671eb0908 -->
## 4. Dicionários: a estrutura mais importante de Python

Um `dict` associa **chaves hasheáveis** a valores arbitrários, com busca **O(1)** em média. Desde o Python 3.7, ele preserva a ordem de inserção.

<!-- en:7a7767657b -->
### Acesso seguro: `.get()` e `.setdefault()`

Indexar uma chave inexistente gera `KeyError`. No trabalho com dados, em que um campo pode simplesmente faltar em alguns registros, normalmente não é isso que você quer.

<!-- en:1470dc4571 -->
### Como juntar dicionários

O operador `|` (Python 3.9+) é o jeito mais limpo de juntar dicionários. As chaves da direita prevalecem, o que o torna perfeito para sobrepor configurações aos valores padrão.

<!-- en:cc1648e292 -->
## 5. Sets: pertencimento e remoção de duplicatas

Um `set` é uma coleção não ordenada de elementos únicos e hasheáveis, com teste de pertencimento **O(1)**. Ele é a resposta certa com muito mais frequência do que as pessoas imaginam.

<!-- en:bd2a4e580e -->
### Álgebra de conjuntos

Estas operações substituem muito código de loop e comparação. Elas correspondem diretamente aos joins de SQL, que você vai ver no Módulo 05.

<!-- en:8502b973e6 -->
## 6. Por que a complexidade não é teoria

Eis o objetivo de todo este notebook. Verificamos o pertencimento de 1 000 itens em contêineres de tamanho crescente: uma vez com uma `list`, outra com um `set`.

A lista precisa percorrer todos os elementos (**O(n)**); o set calcula o hash do valor e vai direto até ele (**O(1)**).

<!-- en:2ef4a1f11d -->
> [!TIP]
> **A regra prática:** no momento em que você escrever `if x in some_list` *dentro de um loop*, converta a lista para set antes. Esse único hábito costuma transformar um processo de várias horas em um de poucos segundos.

```python
# ✘ O(n × m)
valid = load_valid_ids()                    # uma lista de 500.000 ids
clean = [r for r in records if r["id"] in valid]

# ✔ O(n + m): só uma linha a mais
valid = set(load_valid_ids())
clean = [r for r in records if r["id"] in valid]
```

<!-- en:9c8aee48a9 -->
### O custo: memória

Sets e dicts compram velocidade com espaço. Eles mantêm a tabela hash deliberadamente esparsa (cerca de um terço vazia) para que as colisões continuem raras.

<!-- en:ce81dc3345 -->
## 7. Comprehensions

Uma comprehension constrói uma coleção a partir de um iterável em uma única expressão. Ela é mais legível que o loop equivalente e mensuravelmente mais rápida, porque o loop roda em C, não no interpretador.

```text
[ expressão    for item in iterável   if condição ]
  └─ o que       └─ de onde os         └─ filtro
     produzir       itens vêm             opcional
```

<!-- en:75b9692ff2 -->
### Comprehensions de dict e de set

<!-- en:2a4ab0f3aa -->
### Expressões condicionais dentro de uma comprehension

Repare na posição: um filtro (`if` no final) *remove* itens; uma expressão condicional (`if/else` no início) os *transforma*. Os dois são úteis, e confundi-los é um deslize comum.

<!-- en:982559410b -->
> [!WARNING]
> **Não aninhe mais de dois níveis.** Comprehensions deixam de ser legíveis rapidamente. Se você precisar de um comentário para explicar uma, escreva o loop: clareza vale mais que esperteza em um código que alguém vai depurar às 2 da manhã.

<!-- en:ed17104eba -->
## 8. O módulo `collections`

Quatro contêineres especializados que eliminam muito código repetitivo.

<!-- en:16d45887a3 -->
### `Counter`: contagem de frequências em uma linha

<!-- en:a068ece092 -->
### `defaultdict`: chega de "if key not in dict"

<!-- en:4c81942aff -->
### `deque`: uma fila com inserções O(1) nas *duas* pontas

`list.insert(0, x)` e `list.pop(0)` são **O(n)**: todos os elementos restantes precisam se deslocar. Um `deque` torna as duas pontas O(1).

<!-- en:ccdca129f9 -->
### `namedtuple`: registros com nomes em vez de posições

<!-- en:5e3ded7646 -->
> [!TIP]
> Para qualquer coisa além de um registro trivial, prefira `@dataclass`, que aceita type hints, valores padrão, métodos e mutabilidade. O notebook 05 trata dele.

<!-- en:f23ab6cbef -->
## 9. Ordenação

`sorted()` devolve uma nova lista; `list.sort()` ordena no lugar. Os dois aceitam uma função `key`, que calcula *por qual valor* ordenar.

<!-- en:81bb6eb7ef -->
### `operator.itemgetter`: mais rápido e mais claro que um lambda

<!-- en:8f0c910dfa -->
### Estabilidade da ordenação

A ordenação de Python é **estável**: itens considerados iguais mantêm a ordem relativa original. Isso é uma garantia, não um acaso, e permite uma técnica poderosa: ordene primeiro pela chave menos significativa e depois pela mais significativa.

<!-- en:2dcd4662dc -->
## 10. Desempacotamento e expressões com asterisco

<!-- en:b91d9ad9af -->
## 11. Exercícios

### Exercício 3.1: escolha a estrutura

Para cada requisito, indique a melhor coleção embutida e justifique em uma frase:

1. Guardar os 50 milhões de IDs de usuário já processados, para pular duplicatas.
2. Manter as últimas 100 leituras de um sensor para uma média móvel.
3. Associar um código de país ao nome do país.
4. Representar uma cor RGB que nunca pode mudar.
5. Contar quantas vezes cada palavra aparece em 10 000 avaliações de produtos.

### Exercício 3.2: conserte a função lenta

```python
def find_new_customers(all_ids: list[int], known_ids: list[int]) -> list[int]:
    return [i for i in all_ids if i not in known_ids]
```

Com 100 000 `all_ids` e 100 000 `known_ids`, isso leva minutos. Reescreva para rodar em bem menos de um segundo e meça as duas versões com `%timeit` para provar.

### Exercício 3.3: prática de comprehensions

Dado `sales = [{"product": "A", "units": 10, "price": 5.0}, ...]`, escreva comprehensions únicas que produzam:

1. Uma lista de receitas (`units * price`).
2. Um dict que associa produto → receita.
3. O set de produtos com receita acima de 100.

### Exercício 3.4: agrupe sem pandas

Usando `defaultdict`, escreva `group_by_key(records, key)`, que agrupa uma lista de dicts por qualquer campo e devolve `dict[Any, list[dict]]`. Isso é `pandas.groupby` em oito linhas, e escrever uma vez deixa o Módulo 03 muito mais claro.

### Exercício 3.5: ordenação por várias chaves

Ordene a lista `employees` por departamento em ordem crescente, depois por salário em ordem decrescente e depois por nome em ordem crescente, usando uma **única** chamada a `sorted()`. *Dica:* inverta o sinal do campo numérico dentro da tupla da chave.

<!-- en:77e919c425 -->
## Principais lições

- **Escolha contêineres pela complexidade.** O pertencimento em `list` é O(n); em `set` e `dict`, é O(1).
- O hábito que importa: **use `set()` em qualquer coleção na qual você vá testar pertencimento repetidas vezes.**
- Métodos de lista que modificam devolvem `None`. Use `sorted()` quando quiser uma nova lista.
- Tuplas são registros *e* a sequência hasheável: podem ser chaves de dict.
- **Comprehensions** são a expressão padrão para construir coleções; limite-as a dois níveis.
- `Counter`, `defaultdict`, `deque` e `namedtuple` eliminam categorias inteiras de código repetitivo.
- A ordenação de Python é **estável**, o que torna confiável ordenar em várias passadas.

---

**Próximo:** [04 · Funções e Programação Funcional](04_functions_and_functional_programming.ipynb): assinaturas, closures, decoradores e geradores que processam arquivos maiores que a memória RAM.

<!-- comments -->
EN: # Lists hold anything, including mixed types — though in practice, homogeneous
PT: # Listas guardam qualquer coisa, inclusive tipos misturados, mas na prática listas
EN: # lists are what you want (and what NumPy will demand in Module 02).
PT: # homogêneas são o que você quer (e o que o NumPy vai exigir no Módulo 02).
EN: # A slice always produces a NEW list — this is the standard shallow-copy idiom.
PT: # Uma fatia sempre produz uma NOVA lista: é a forma padrão de fazer uma cópia rasa.
EN: # sorts IN PLACE and returns None
PT: # ordena NO LUGAR e devolve None
EN: # When you want a new sorted list and the original preserved, use `sorted()`.
PT: # Quando quiser uma nova lista ordenada e preservar a original, use `sorted()`.
EN: # The in-place mutators, for reference.
PT: # Os métodos que modificam no lugar, para referência.
EN: # add one item to the end          — O(1)
PT: # adiciona um item ao final          — O(1)
EN: # add many items                   — O(k)
PT: # adiciona vários itens              — O(k)
EN: # insert at a position             — O(n), shifts everything
PT: # insere em uma posição              — O(n), desloca tudo
EN: # delete the first matching VALUE  — O(n)
PT: # apaga o primeiro VALOR igual       — O(n)
EN: # remove and return the last item  — O(1)
PT: # remove e devolve o último item     — O(1)
EN: # (latitude, longitude) — position carries meaning
PT: # (latitude, longitude): a posição tem significado
EN: # Unpacking is the idiomatic way to read a record.
PT: # O desempacotamento é o jeito idiomático de ler um registro.
EN: # Tuples are hashable (if their contents are), so they can be dict keys.
PT: # Tuplas são hasheáveis (se o conteúdo for), então podem ser chaves de dict.
EN: # This is genuinely useful for sparse, multi-dimensional lookups.
PT: # Isso é muito útil para buscas esparsas e multidimensionais.
EN: # A list cannot do this.
PT: # Uma lista não pode fazer isso.
EN: # allowed — we mutate the LIST, not the tuple
PT: # permitido: modificamos a LISTA, não a tupla
EN: # not allowed — this would rebind a tuple slot
PT: # não permitido: isso religaria uma posição da tupla
EN: # Iterate over key/value pairs with .items() — the idiomatic loop.
PT: # Percorra pares de chave/valor com .items(): é o loop idiomático.
EN: # ✘ Raises on a missing key.
PT: # ✘ Gera erro com uma chave inexistente.
EN: # ✔ Returns a default instead.
PT: # ✔ Devolve um valor padrão.
EN: # `.setdefault` returns the value, inserting the default first if absent.
PT: # `.setdefault` devolve o valor, inserindo antes o padrão se a chave não existir.
EN: # The equivalent unpacking syntax, which also works in comprehensions:
PT: # A sintaxe equivalente com desempacotamento, que também funciona em comprehensions:
EN: # Look up values that are NOT present — the worst case for a list, since it
PT: # Busca valores que NÃO existem: o pior caso para uma lista, que
EN: # must scan all n elements before concluding "no".
PT: # precisa percorrer os n elementos antes de concluir que "não".
EN: # --- Left: absolute time, log scale ----------------------------------------
PT: # --- Esquerda: tempo absoluto, escala logarítmica ---------------------------
EN: # A log y-axis is essential here: the two series differ by ~4 orders of
PT: # Um eixo y logarítmico é essencial aqui: as duas séries diferem em ~4 ordens de
EN: # magnitude, so on a linear axis the set line would be flat against zero.
PT: # grandeza, então em um eixo linear a linha do set ficaria colada no zero.
EN: # Direct-label the endpoints so identity does not rest on colour alone.
PT: # Rotula as pontas diretamente, para a identificação não depender só da cor.
EN: # --- Right: the speed-up factor --------------------------------------------
PT: # --- Direita: o fator de ganho de velocidade --------------------------------
EN: # The loop version.
PT: # A versão com loop.
EN: # The comprehension — same result, one line, no mutable accumulator.
PT: # A comprehension: mesmo resultado, uma linha, sem acumulador mutável.
EN: # Dict comprehension.
PT: # Comprehension de dict.
EN: # With a filter and a transformation.
PT: # Com um filtro e uma transformação.
EN: # Set comprehension — deduplicates automatically.
PT: # Comprehension de set: remove duplicatas automaticamente.
EN: # removes items
PT: # remove itens
EN: # keeps all, clips them
PT: # mantém todos e limita os valores
EN: # Two levels is the sane maximum. This flattens a nested structure:
PT: # Dois níveis é o máximo sensato. Isto achata uma estrutura aninhada:
EN: # ✘ The manual version.
PT: # ✘ A versão manual.
EN: # ✔ defaultdict calls the factory (`list`) automatically for a missing key.
PT: # ✔ defaultdict chama a fábrica (`list`) automaticamente para uma chave inexistente.
EN: # A defaultdict(float) makes running totals equally clean.
PT: # Um defaultdict(float) deixa os totais acumulados igualmente limpos.
EN: # A rolling window of the last 3 readings — `maxlen` discards old items for free.
PT: # Uma janela móvel das últimas 3 leituras: `maxlen` descarta os itens antigos sem custo.
EN: # A plain tuple forces you to remember what index 2 meant.
PT: # Uma tupla comum obriga você a lembrar o que o índice 2 significava.
EN: # Sort by a single field.
PT: # Ordena por um único campo.
EN: # Sort by several fields: return a tuple from the key function. Tuples compare
PT: # Ordena por vários campos: a função-chave devolve uma tupla. Tuplas são comparadas
EN: # element by element, so this sorts by dept, then by salary descending.
PT: # elemento a elemento, então isto ordena por dept e depois por salário decrescente.
EN: # Equivalent to `lambda e: (e["dept"], e["salary"])`, but implemented in C.
PT: # Equivale a `lambda e: (e["dept"], e["salary"])`, mas é implementado em C.
EN: # Sort by salary first, then by department. Because the sort is stable, the
PT: # Ordena primeiro por salário e depois por departamento. Como a ordenação é estável, a
EN: # salary ordering survives inside each department group.
PT: # ordem de salários se mantém dentro de cada grupo de departamento.
EN: # Basic unpacking.
PT: # Desempacotamento básico.
EN: # Star captures "the rest" — always as a list.
PT: # O asterisco captura "o resto", sempre como lista.
EN: # Swapping needs no temporary variable.
PT: # Trocar valores não exige variável temporária.
EN: # Unpacking into a function call: * for positional, ** for keyword arguments.
PT: # Desempacotamento em uma chamada de função: * para posicionais, ** para palavras-chave.
EN: # keys become keyword arguments
PT: # as chaves viram argumentos por palavra-chave
EN: # items become positional arguments
PT: # os itens viram argumentos posicionais
