<!-- en:136349b81e -->
# 05 · Agregações e Máscaras Booleanas

**Módulo 02 — NumPy e Computação Numérica**

---

## O que você vai aprender

1. `axis=` desmistificado: o único modelo mental que torna tudo óbvio.
2. `keepdims`, e por que ele existe.
3. Máscaras booleanas: o substituto vetorizado para um `if` dentro de um loop.
4. `np.where`, `np.select`, `np.clip`: lógica condicional sem ramificação.
5. Indexação avançada, `argsort`, `argpartition`.
6. Contagem e agrupamento: `unique`, `bincount`, `histogram`.

**Tempo estimado:** 90 minutos

<!-- en:f25cdea3ae -->
## 1. `axis=`: a única regra de que você precisa

Quase todo mundo entende isto ao contrário no início, porque "axis=0" parece "faça nas linhas". A leitura correta é o oposto.

![Como ler axis= corretamente](../../../assets/images/axis_semantics.png)

> **`axis` indica a dimensão que DESAPARECE.**

`arr.sum(axis=0)` colapsa o eixo 0, então as linhas somem e sobra um valor por coluna.

<!-- en:87ec7980f1 -->
A aritmética dos shapes torna isso mecânico: comece em `(3, 4)`, apague a entrada na posição `axis`, e esse é o shape do resultado.

<!-- en:0c3c2318b6 -->
### As reduções que vale conhecer

<!-- en:7a8eb0f031 -->
> [!TIP]
> `ptp` significa "peak to peak" (de pico a pico): `max - min`. É um jeito rápido de conferir a amplitude de uma coluna, e um `ptp` igual a 0 mostra na hora que um atributo é constante e, portanto, inútil para um modelo.

<!-- en:4c7a100859 -->
### `argmin` / `argmax`: *onde*, não *o quê*

<!-- en:7d01394591 -->
### `keepdims`: o argumento que salva os seus broadcasts

Uma redução remove uma dimensão, o que depois quebra o broadcast que você queria fazer. `keepdims=True` mantém no lugar uma dimensão de tamanho 1.

<!-- en:9f83eda148 -->
> [!WARNING]
> Sem `keepdims`, `(4,3) / (4,)` gera um erro aqui, o que é sorte. Em uma matriz **quadrada**, o mesmo erro faz o broadcast em silêncio e normaliza o eixo errado. Use `keepdims=True` sempre que a redução for alimentar um broadcast.

<!-- en:c8e2ccbfef -->
## 2. Máscaras booleanas

Uma comparação em um array produz um array de booleanos. Essa máscara pode então selecionar, contar ou substituir, sem loop e sem `if`.

<!-- en:97c716bb63 -->
### Como combinar condições: atenção aos parênteses

O NumPy usa `&`, `|`, `~` para lógica elemento a elemento. `and`, `or`, `not` de Python **não** funcionam, porque tentam avaliar o array inteiro como um único valor verdadeiro ou falso: é o mesmo erro de ambiguidade que você viu no Módulo 01.

<!-- en:9a4ffd1f72 -->
> [!WARNING]
> **Em Python, `&` tem precedência sobre `>`.** Sem parênteses, `values > 20 & values < 60` é interpretado como `values > (20 & values) < 60` e gera um erro ou, pior, calcula outra coisa sem aviso. Sempre coloque cada comparação entre parênteses.

<!-- en:27f982a817 -->
### `any` e `all`

<!-- en:49aead8ca4 -->
### Atribuição por meio de uma máscara

<!-- en:334a7b11f7 -->
## 3. Lógica condicional sem ramificação

### `np.where`: if/else vetorizado

<!-- en:d7b015372c -->
### `np.select`: mais de dois caminhos

<!-- en:5bdac27f96 -->
### `np.clip`: limita os valores a um intervalo

<!-- en:f22040ed15 -->
## 4. Indexação avançada

Indexar com um array de **inteiros** seleciona elementos arbitrários, em qualquer ordem e com repetição. Ao contrário do fatiamento, sempre devolve uma **cópia**.

<!-- en:98848da850 -->
### `argsort`: ordena um array por outro

<!-- en:8d55f07816 -->
### `argpartition`: os *k* maiores sem ordenar tudo

Quando você só precisa dos poucos maiores, ordenar tudo é trabalho desperdiçado. `argpartition` é **O(n)**, enquanto `argsort` é **O(n log n)**.

<!-- en:75f8e92c5e -->
## 5. Contagem e agrupamento

<!-- en:820fe4d36c -->
### Agregação por grupo sem pandas

Este é o `groupby().sum()` implementado a partir dos princípios básicos. Entendê-lo torna o Módulo 03 muito menos misterioso.

<!-- en:619628f359 -->
### Faixas com `digitize` e `histogram`

<!-- en:fd38f412d5 -->
## 6. Juntando tudo: um relatório de qualidade vetorizado

<!-- en:f1af1ef4b2 -->
## 7. Exercícios

### Exercício 5.1: treino de eixos

Para `a = np.arange(24).reshape(2, 3, 4)`, diga o shape de cada resultado antes de executar:

```python
a.sum(axis=0)      a.sum(axis=1)      a.sum(axis=2)
a.sum(axis=(0, 2)) a.mean(axis=1, keepdims=True)   a.max(axis=-1)
```

### Exercício 5.2: a armadilha do keepdims

Dado `X` de shape `(5, 5)`, escreva `X / X.sum(axis=1)` e `X / X.sum(axis=1, keepdims=True)`. Os dois rodam. Mostre que o primeiro está errado e explique com precisão o que ele calculou.

### Exercício 5.3: prática de máscaras

Para `data = rng.normal(100, 20, 10_000)`:

1. Quantos valores ficam a mais de dois desvios padrão da média?
2. Substitua-os pela média, sem loop.
3. Informe a parcela de valores em `[80, 120]`.

### Exercício 5.4: group-by do zero

Escreva `group_stats(keys, values)`, que devolve, por chave única, a contagem, a soma, a média e o máximo, usando apenas NumPy. Trate chaves de texto com `np.unique(..., return_inverse=True)`. Confira com o pandas.

### Exercício 5.5: top-k por linha

Dado `scores` de shape `(1000, 50)` (1000 usuários × 50 itens), encontre os 5 melhores itens de cada usuário com `argpartition` e devolva um array `(1000, 5)` de índices de itens, ordenados por pontuação decrescente. Esse é o núcleo da entrega de um sistema de recomendação.

<!-- en:1466021bab -->
## Principais lições

- **`axis` indica a dimensão que desaparece.** Apague essa entrada da tupla do shape, e você tem o shape do resultado.
- Use **`keepdims=True`** sempre que a redução alimentar um broadcast: isso evita toda uma classe de erros silenciosos em arrays quadrados.
- `argmin`/`argmax` devolvem **índices**; em 2-D, eles achatam o array, a menos que você passe um eixo. `np.unravel_index` faz a conversão de volta.
- Máscaras booleanas substituem o `if` dentro de um loop. Combine com **`&`, `|`, `~`** e **sempre use parênteses** em cada comparação.
- `mask.sum()` conta e `mask.mean()` dá a proporção: booleanos são apenas 0 e 1.
- **`np.where`** é if/else vetorizado; **`np.select`** trata vários caminhos; **`np.clip`** limita um intervalo.
- **A indexação avançada devolve uma cópia** e permite repetições e ordem arbitrária.
- Use **`argpartition`** em vez de `argsort` quando só precisar dos *k* maiores.
- **`np.bincount(ids, weights=values)`** é uma soma por grupo em uma chamada.

---

**Próximo:** [06 · Álgebra Linear](06_linear_algebra.ipynb): a matemática sobre a qual rodam todos os modelos deste curso.

<!-- comments -->
EN: # Several axes at once.
PT: # Vários eixos ao mesmo tempo.
EN: # In 2-D, argmax flattens by default — a classic surprise.
PT: # Em 2-D, argmax achata o array por padrão: uma surpresa clássica.
EN: # shape (4,)
PT: # shape (4,)
EN: # shape (4, 1)
PT: # shape (4, 1)
EN: # Normalising rows so each sums to 1:
PT: # Normaliza as linhas para que cada uma some 1:
EN: # The idiomatic data-quality check.
PT: # A verificação idiomática de qualidade de dados.
EN: # -999 is a sentinel for "sensor failed" — a real and common convention.
PT: # -999 é um sentinela para "sensor falhou", uma convenção real e comum.
EN: # np.where(condition, value_if_true, value_if_false)
PT: # np.where(condition, value_if_true, value_if_false)
EN: # All three arguments broadcast.
PT: # Os três argumentos fazem broadcast.
EN: # With one argument it returns the INDICES where the condition holds.
PT: # Com um argumento, devolve os ÍNDICES em que a condição é verdadeira.
EN: # Winsorising: clip at percentiles rather than fixed values — the technique
PT: # Winsorização: limita por percentis em vez de valores fixos, a técnica
EN: # behind the OutlierClipper you wrote in Module 01, notebook 05.
PT: # por trás do OutlierClipper que você escreveu no Módulo 01, notebook 05.
EN: # inject outliers
PT: # injeta outliers
EN: # In 2-D, index arrays pair up element-wise.
PT: # Em 2-D, os arrays de índices formam pares elemento a elemento.
EN: # Selecting whole rows in a chosen order — how you shuffle a dataset.
PT: # Selecionar linhas inteiras em uma ordem escolhida: é assim que se embaralha um conjunto de dados.
EN: # Shuffling features and labels together, keeping them aligned.
PT: # Embaralha atributos e rótulos juntos, mantendo o alinhamento.
EN: # descending
PT: # decrescente
EN: # Guarantees the last k positions hold the k largest, unordered among themselves.
PT: # Garante que as últimas k posições têm os k maiores, sem ordem entre eles.
EN: # unique with counts — the NumPy equivalent of collections.Counter.
PT: # unique com contagens: o equivalente em NumPy de collections.Counter.
EN: # bincount is faster for small non-negative integers.
PT: # bincount é mais rápido para inteiros pequenos não negativos.
EN: # unique also returns the inverse mapping — this is label encoding.
PT: # unique também devolve o mapeamento inverso: isso é codificação de rótulos.
EN: # bincount with `weights` sums the weights per bin — a group-by in one call.
PT: # bincount com `weights` soma os pesos por faixa: um group-by em uma chamada.
EN: # Cross-check against pandas.
PT: # Confere com o pandas.
EN: # digitize returns which bin each value falls into (1-indexed by default).
PT: # digitize devolve em qual faixa cada valor cai (começando em 1, por padrão).
EN: # Outliers by the 1.5×IQR rule, computed per column.
PT: # Outliers pela regra de 1.5×IQR, calculados por coluna.
EN: # A realistic messy matrix.
PT: # Uma matriz realista e bagunçada.
EN: # well behaved
PT: # bem-comportado
EN: # skewed, will show outliers
PT: # assimétrico, vai mostrar outliers
EN: # constant — useless feature
PT: # constante: atributo inútil
EN: # inject missingness
PT: # injeta valores ausentes
EN: # Mark the IQR outlier fences so the report's numbers are visible.
PT: # Marca os limites de outliers por IQR, para os números do relatório ficarem visíveis.
