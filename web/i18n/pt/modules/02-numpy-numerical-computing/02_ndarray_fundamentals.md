<!-- en:e455cf9c2a -->
# 02 · Fundamentos do ndarray

**Módulo 02 — NumPy e Computação Numérica**

---

## O que você vai aprender

1. Todas as formas de criar um array, e qual escolher.
2. `shape`, `ndim`, `size`, `dtype`: os quatro atributos que você vai conferir o tempo todo.
3. Indexação e fatiamento em N dimensões.
4. **Views e cópias**: a regra que *se inverte* em relação ao Módulo 01, e a maior fonte de bugs silenciosos em código NumPy.
5. Remodelagem, transposição, empilhamento e divisão.

**Tempo estimado:** 90 minutos

<!-- en:fbab178407 -->
## 1. Como criar arrays

### A partir de dados Python existentes

<!-- en:96a3db0c00 -->
### Do zero: as funções que você vai usar todo dia

<!-- en:14a8121bcf -->
> [!WARNING]
> `np.empty` **não** zera a memória: ele entrega os bytes que já estavam lá. É um pouco mais rápido que `zeros` e só é a escolha certa quando você vai sobrescrever todos os elementos. Usá-lo e esquecer de preencher produz lixo com aparência plausível.

<!-- en:fc78909d12 -->
### Intervalos e grades

<!-- en:095da080d5 -->
> [!TIP]
> **Prefira `linspace` a `arange` com floats.** `arange` acumula erro de ponto flutuante e seu ponto final é imprevisível, consequência direta do problema de representação do Módulo 01.

<!-- en:395865840a -->
### Arrays aleatórios

Use sempre a API moderna de `Generator` (`np.random.default_rng`). As funções antigas `np.random.seed` / `np.random.rand` compartilham um único estado global, o que torna a reprodutibilidade frágil assim que mais de um trecho de código sorteia números.

<!-- en:bcad7f7454 -->
## 2. Os quatro atributos que você vai conferir o tempo todo

Quando algo dá errado em código NumPy, a resposta quase sempre está em um destes. Imprima-os antes de depurar qualquer outra coisa.

<!-- en:e6aab101a7 -->
### A distinção que confunde todo mundo: 1-D e 2-D

`(3,)`, `(3, 1)` e `(1, 3)` guardam os mesmos três números e se comportam de maneira completamente diferente no broadcasting (notebook 04) e na multiplicação de matrizes (notebook 06).

<!-- en:955b088e17 -->
## 3. Indexação e fatiamento

<!-- en:59e11c74be -->
### `...` (Ellipsis): "todos os eixos restantes"

É útil quando você tem muitas dimensões e só se importa com a primeira ou a última.

<!-- en:726bb990b6 -->
## 4. Views e cópias: leia isto duas vezes

![Views e cópias](../../../assets/images/views_vs_copies.png)

No Módulo 01, você aprendeu que fatiar uma lista **copia**. No NumPy, **o fatiamento básico devolve uma view**: uma segunda janela para a *mesma* memória. Gravar por meio da view altera o original.

Essa é uma decisão de design deliberada: torna o fatiamento gratuito, seja qual for o tamanho. Também é a origem de toda uma categoria de bugs silenciosos.

<!-- en:dabc0d21d3 -->
### Como identificar

`.base` aponta para o array do qual uma view empresta os dados, e é `None` em um array dono dos próprios dados.

<!-- en:2fda6650e5 -->
### O que produz uma view e o que produz uma cópia

| Operação | Resultado | |
| :--- | :--- | :-- |
| `arr[1:5]`: fatiamento básico | **view** | compartilha memória |
| `arr[:, 0]` | **view** | compartilha memória |
| `arr.reshape(...)` | **view** *(quando possível)* | compartilha memória |
| `arr.T` / `arr.transpose()` | **view** | compartilha memória |
| `arr.ravel()` | **view** *(quando possível)* | compartilha memória |
| `arr[[0, 2, 4]]`: indexação avançada | **cópia** | independente |
| `arr[arr > 3]`: máscara booleana | **cópia** | independente |
| `arr.flatten()` | **cópia** | sempre |
| `arr.copy()` | **cópia** | sempre |
| `arr + 1`, `np.sqrt(arr)` | **novo array** | independente |

<!-- en:1990f4143b -->
### O bug que isso causa na prática

<!-- en:6436eed9e8 -->
> [!WARNING]
> Repare *por que* a versão quebrada causa problema: `column -= ...` é uma operação **no lugar** sobre uma view. Se fosse `column = column - column.mean()`, apenas religaria um nome local e não faria nada: é a mesma distinção entre `+=` e `= ... +` do Módulo 01, notebook 02, com mais coisa em jogo.

<!-- en:bd894d54b3 -->
## 5. Remodelagem

<!-- en:75c78d34e7 -->
### `ravel` e `flatten`

Os dois produzem um array 1-D. `ravel` devolve uma view quando pode (barato); `flatten` sempre copia (seguro).

<!-- en:f494df912d -->
### Transposição

<!-- en:30a36687cd -->
> [!NOTE]
> Uma transposição não custa nada: ela só reinterpreta os strides. Mas o resultado deixa de ser C-contíguo, o que pode deixar operações seguintes mais lentas. O notebook 08 mostra quando vale pagar por `np.ascontiguousarray`.

<!-- en:5312be2752 -->
## 6. Como combinar e dividir arrays

<!-- en:6d8e968fff -->
## 7. Um checklist prático

Quando uma operação com arrays se comportar mal, siga esta ordem:

<!-- en:7e0e83565b -->
## 8. Exercícios

### Exercício 2.1: preveja os shapes

Sem executar nada, diga o shape de cada resultado:

```python
a = np.arange(24).reshape(2, 3, 4)

a[0]            # ?
a[:, 1]         # ?
a[..., 2]       # ?
a[0, :, ::2]    # ?
a.T             # ?
a.reshape(4, -1)  # ?
```

### Exercício 2.2: view ou cópia?

Para cada expressão, diga se ela compartilha memória com `a` e depois verifique com `np.shares_memory`:

```python
a = np.arange(20).reshape(4, 5)

a[1:3]         a[:, [0, 1]]      a.T[0]
a[a % 2 == 0]  a.reshape(5, 4)   a.ravel()[::2]
```

### Exercício 2.3: conserte o bug de modificação

```python
def scale_features(X, factor):
    for j in range(X.shape[1]):
        col = X[:, j]
        col *= factor
    return X
```

Mostre que ela corrompe o array de quem chama e reescreva para que isso não aconteça. Escreva uma asserção que teria pegado o bug.

### Exercício 2.4: construa um tabuleiro de xadrez

Crie um array 8×8 de 0s e 1s alternados (como um tabuleiro de xadrez) **sem nenhum loop Python**. Depois, faça de uma segunda forma. *Dica:* fatiamento com passo, ou `np.indices`.

### Exercício 2.5: remodele para o scikit-learn

Você tem `y = np.array([1.0, 2.0, 3.0])` e um modelo que exige um array 2-D de shape `(n_samples, 1)`. Produza-o de três formas diferentes e confirme que todas compartilham memória com `y`.

<!-- en:c9f4a426cc -->
## Principais lições

- Ao depurar, confira primeiro **`shape`, `ndim`, `dtype`, `size`**. A maioria dos bugs de NumPy é de shape.
- `(3,)`, `(3, 1)` e `(1, 3)` são três coisas diferentes.
- Prefira **`linspace` a `arange`** em intervalos de ponto flutuante.
- Use **`np.random.default_rng(seed)`**, não as funções globais antigas.
- **O fatiamento básico devolve uma VIEW**, o oposto do fatiamento de listas. Indexação avançada e máscaras booleanas devolvem cópias.
- Verifique com **`.base`** ou **`np.shares_memory`**; force a independência com **`.copy()`**.
- Operadores no lugar (`-=`, `*=`) aplicados a uma view gravam no original.
- `ravel` é uma view barata; `flatten` é sempre uma cópia.
- `reshape(-1, 1)` transforma um array 1-D no vetor coluna que o scikit-learn quer.

---

**Próximo:** [03 · Dtypes e Precisão Numérica](03_dtypes_and_precision.ipynb), onde o aviso do Módulo 01 sobre o estouro silencioso de inteiros é tratado por completo.

<!-- comments -->
EN: # From a list — the dtype is inferred from the contents.
PT: # A partir de uma lista: o dtype é inferido do conteúdo.
EN: # nested lists → 2-D
PT: # listas aninhadas → 2-D
EN: # Being explicit about dtype is a good habit — it documents intent and prevents
PT: # Ser explícito sobre o dtype é um bom hábito: documenta a intenção e evita
EN: # the surprise of getting int64 when you needed floats.
PT: # a surpresa de receber int64 quando você precisava de floats.
EN: # identity matrix
PT: # matriz identidade
EN: # arange: like range(), but returns an array and accepts floats.
PT: # arange: como range(), mas devolve um array e aceita floats.
EN: # linspace: N evenly spaced points, INCLUSIVE of the endpoint.
PT: # linspace: N pontos igualmente espaçados, INCLUINDO o ponto final.
EN: # logspace: evenly spaced on a log scale — ideal for hyperparameter grids.
PT: # logspace: igualmente espaçados em escala logarítmica, ideal para grades de hiperparâmetros.
EN: # The trap, concretely:
PT: # A armadilha, concretamente:
EN: # Same seed → same numbers, every time, on every machine.
PT: # Mesma seed → mesmos números, sempre, em qualquer máquina.
EN: # Single element: [row, col] — one pair of brackets, not arr[row][col].
PT: # Um único elemento: [row, col], um só par de colchetes, não arr[row][col].
EN: # arr[0][0] also works but is slower: it builds an intermediate row array first.
PT: # arr[0][0] também funciona, mas é mais lento: constrói antes um array intermediário da linha.
EN: # Slicing: [start:stop:step] per axis, separated by commas.
PT: # Fatiamento: [start:stop:step] por eixo, separados por vírgulas.
EN: # a view
PT: # uma view
EN: # in-place → writes into `data`
PT: # no lugar → grava em `data`
EN: # -1 means "work this dimension out for me". Use it so the code survives a
PT: # -1 significa "calcule esta dimensão para mim". Use para o código sobreviver a uma
EN: # change in the number of rows.
PT: # mudança no número de linhas.
EN: # The most common reshape in machine learning: turning a 1-D array into a
PT: # A remodelagem mais comum em machine learning: transformar um array 1-D em um
EN: # single-column 2-D array, because scikit-learn requires 2-D feature matrices.
PT: # array 2-D de uma coluna, porque o scikit-learn exige matrizes de atributos 2-D.
EN: # `stack` differs from `concatenate`: it adds a NEW axis rather than extending
PT: # `stack` difere de `concatenate`: adiciona um NOVO eixo em vez de estender
EN: # an existing one. This is how you build a batch out of individual samples.
PT: # um existente. É assim que você monta um lote a partir de amostras individuais.
EN: # new axis 0
PT: # novo eixo 0
EN: # Splitting is the inverse.
PT: # Dividir é a operação inversa.
EN: # into 3 equal parts
PT: # em 3 partes iguais
EN: # Or at explicit indices — this is how a train/validation/test split works
PT: # Ou em índices explícitos: é assim que funciona uma divisão treino/validação/teste
EN: # before scikit-learn does it for you.
PT: # antes de o scikit-learn fazê-la para você.
