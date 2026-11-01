<!-- en:ea52394c9a -->
# 04 · Broadcasting

**Módulo 02 — NumPy e Computação Numérica**

---

## O que você vai aprender

1. As duas regras que determinam por completo se dois shapes são compatíveis.
2. Como ler e corrigir uma mensagem de erro de broadcasting.
3. `np.newaxis` / `reshape` para deixar a sua intenção explícita.
4. O bug silencioso: shapes que fazem broadcast, mas produzem a coisa errada.
5. Aplicações reais: padronização de atributos, distâncias par a par, one-hot encoding.
6. A armadilha de memória, e como `einsum` e o processamento em blocos a evitam.

**Tempo estimado:** 75 minutos

<!-- en:126b3450e8 -->
## 1. As regras

O broadcasting permite ao NumPy combinar arrays de shapes diferentes sem que você escreva um loop ou materialize uma cópia. Existem exatamente duas regras.

![Broadcasting: as duas regras](../../../assets/images/broadcasting_rules.png)

1. **Alinhe os shapes pela direita.** Se um for mais curto, complete-o à **esquerda** com 1s.
2. **Duas dimensões são compatíveis se forem iguais ou se uma delas for 1.** Uma dimensão de tamanho 1 é *esticada* para se igualar à outra.

Se algum par não atender à regra 2, a operação gera um erro.

<!-- en:620b9587cc -->
### Como verificar a compatibilidade sem executar a operação

`np.broadcast_shapes` responde "estes vão se combinar, e em quê?". É útil ao depurar e barato, porque não aloca nada.

<!-- en:1adf3bac19 -->
## 2. Como ler a mensagem de erro

O erro de broadcasting do NumPy é preciso quando você sabe interpretá-lo.

<!-- en:20327739c3 -->
Decifrando:

- `operands could not be broadcast together`: a regra 2 falhou.
- `with shapes (3,4) (3,)`: as duas entradas.

Alinhe-as pela direita, como o NumPy faz:

```text
(3, 4)
   (3)   →  completado para (1, 3)
─────────
 4 e 3   →  nenhum é 1  →  falha
```

A intenção era quase certamente "somar um valor por **linha**", o que significa que `b` precisa ser uma **coluna**:

<!-- en:c6969091a3 -->
## 3. Como deixar a intenção explícita com `np.newaxis`

`np.newaxis` (um apelido para `None`) insere uma dimensão de tamanho 1. É a forma mais clara de dizer "trate isto como uma linha" ou "trate isto como uma coluna".

<!-- en:7ee5300993 -->
## 4. O bug silencioso

O caso perigoso não é o que gera um erro. É o que **faz o broadcast com sucesso para um shape que você não queria**.

<!-- en:69b9f00598 -->
> [!WARNING]
> Este é o bug de broadcasting mais comum em código de machine learning, e geralmente aparece como uma coluna `(n, 1)` vinda de `reshape(-1, 1)`, ou de uma coluna de DataFrame que encontra um array 1-D de previsões. Nada gera erro; a perda simplesmente fica errada.

**A defesa: verifique com uma asserção o shape que você espera.**

<!-- en:d488e15859 -->
## 5. Aplicações reais

### Padronização de atributos: a operação por trás do `StandardScaler`

<!-- en:8f32c181ea -->
> [!TIP]
> Repare que é `axis=0` que faz isto funcionar: ele colapsa as **linhas**, deixando uma estatística por coluna. Erre o eixo, e você normalizaria cada amostra em relação a si mesma, o que faz o broadcast sem problemas e não tem sentido. O notebook 05 aprofunda a semântica dos eixos.

<!-- en:dd43ceec52 -->
### Distâncias par a par sem loop

Um item básico de clustering, vizinhos mais próximos e sistemas de recomendação.

<!-- en:bcbca918de -->
### One-hot encoding em uma linha

<!-- en:aa862998ca -->
## 6. A armadilha de memória

O broadcasting **não** materializa o array esticado: esse é o objetivo dele. Mas o **resultado** é totalmente materializado e pode ser enorme.

<!-- en:566488009e -->
Dois arrays pequenos produziram um resultado quase mil vezes maior. Aumente as entradas para 100 000, e o resultado teria 80 GB: um `MemoryError` imediato.

<!-- en:24f47ddc5c -->
### `np.einsum`: reduções sem o intermediário

A soma de Einstein expressa "multiplique estes eixos e some sobre aquele" em uma única passada, sem construir o produto externo completo.

<!-- en:d856d78d5b -->
### Processamento em blocos quando o intermediário é inevitável

<!-- en:cab6c11e42 -->
## 7. Como visualizar um broadcast

<!-- en:630a178bfe -->
## 8. Exercícios

### Exercício 4.1: compatível ou não?

Para cada par, diga o shape do resultado ou "incompatível" e depois confira com `np.broadcast_shapes`:

```
(5, 4)    e (4,)            (5, 4)    e (5,)
(5, 4)    e (5, 1)          (3, 1, 4) e (2, 4)
(2, 3, 4) e (3, 1)          (8, 1, 6) e (7, 1)
```

### Exercício 4.2: conserte o bug silencioso

```python
y_true = np.array([[1.0], [2.0], [3.0]])
y_pred = np.array([1.1, 1.9, 3.2])
residuals = y_true - y_pred
```

Qual é o shape de `residuals`, e qual deveria ser? Corrija e adicione uma asserção que teria pegado o erro.

### Exercício 4.3: normalize linhas em vez de colunas

Dado `X` de shape `(100, 5)`, escale cada **linha** para que ela some 1. Depois, explique por que `X / X.sum(axis=1)` falha e `X / X.sum(axis=1, keepdims=True)` funciona.

### Exercício 4.4: distância até os centroides

Você tem `X` de shape `(1000, 3)` e `centroids` de shape `(8, 3)`. Usando apenas broadcasting, calcule uma matriz `(1000, 8)` de distâncias e depois o índice do centroide mais próximo de cada ponto. Essa é uma iteração completa de k-means, que você vai reutilizar no projeto do módulo.

### Exercício 4.5: evite o intermediário

`(a[:, None] - b[None, :])**2).sum()` calcula a soma das diferenças ao quadrado entre todos os pares. Para `a` e `b` de 50 000 elementos, o intermediário tem 20 GB. Expanda a álgebra e calcule o mesmo número usando apenas `sum`, `dot` e escalares.

<!-- en:6c609f54ba -->
## Principais lições

- **Alinhe pela direita, complete à esquerda com 1s; as dimensões precisam ser iguais ou 1.** Isso é todo o broadcasting.
- Use **`np.broadcast_shapes`** para verificar a compatibilidade sem alocar nada.
- **`arr[:, None]` cria uma coluna; `arr[None, :]` cria uma linha.** Diga o que você quer em vez de torcer.
- O caso perigoso é o que **funciona com o shape errado**: um `(n,)` com um `(n, 1)` dá `(n, n)` e nenhum erro.
- **Verifique os shapes com asserções** em qualquer função que calcule uma métrica ou uma perda.
- O broadcasting não materializa as entradas, mas **materializa o resultado**: dois vetores de 10 mil elementos criam uma matriz de 800 MB.
- Quando só precisar de uma redução, use **álgebra**, **`einsum`** ou **processamento em blocos** em vez de construir o intermediário.

---

**Próximo:** [05 · Agregações e Máscaras Booleanas](05_aggregations_and_masking.ipynb): `axis=` desmistificado, e as ferramentas de indexação que substituem um `if` dentro de um loop.

<!-- comments -->
EN: # The simplest case: a scalar broadcasts against anything.
PT: # O caso mais simples: um escalar faz broadcast com qualquer coisa.
EN: # A scalar is conceptually shape (), padded to (1,), then stretched to (4,).
PT: # Um escalar tem conceitualmente shape (), completado para (1,) e esticado para (4,).
EN: # The everyday case: a (3, 4) matrix and a (4,) row.
PT: # O caso do dia a dia: uma matriz (3, 4) e uma linha (4,).
EN: # fails
PT: # falha
EN: # The classic outer-product pattern: column ⊗ row → matrix.
PT: # O padrão clássico do produto externo: coluna ⊗ linha → matriz.
EN: # (3, 1)
PT: # (3, 1)
EN: # (1, 3)
PT: # (1, 3)
EN: # shape (4,)
PT: # shape (4,)
EN: # shape (4, 1) — a column!
PT: # shape (4, 1): uma coluna!
EN: # 1000 samples, 4 features, deliberately on wildly different scales.
PT: # 1000 amostras, 4 atributos, propositalmente em escalas bem diferentes.
EN: # mean and std are shape (4,); X is (1000, 4).
PT: # mean e std têm shape (4,); X é (1000, 4).
EN: # (4,) pads to (1, 4) and stretches down all 1000 rows — exactly right.
PT: # (4,) é completado para (1, 4) e esticado pelas 1000 linhas: exatamente o certo.
EN: # The wrong axis broadcasts too — and silently produces nonsense.
PT: # O eixo errado também faz broadcast, e produz absurdos sem aviso.
EN: # points[:, None, :] has shape (5, 1, 2)
PT: # points[:, None, :] tem shape (5, 1, 2)
EN: # points[None, :, :] has shape (1, 5, 2)
PT: # points[None, :, :] tem shape (1, 5, 2)
EN: # Broadcasting gives (5, 5, 2): every point against every other point.
PT: # O broadcasting dá (5, 5, 2): cada ponto contra cada um dos outros.
EN: # Sanity checks that should always hold for a distance matrix.
PT: # Verificações que devem sempre valer para uma matriz de distâncias.
EN: # Verify against SciPy's implementation.
PT: # Confere com a implementação do SciPy.
EN: # labels[:, None] is (6, 1); arange(3) is (3,) → (1, 3).
PT: # labels[:, None] é (6, 1); arange(3) é (3,) → (1, 3).
EN: # Broadcasting compares every label against every class index.
PT: # O broadcasting compara cada rótulo com cada índice de classe.
EN: # 10k elements, 80 KB
PT: # 10 mil elementos, 80 KB
EN: # When you only need a REDUCTION of the big intermediate, avoid building it.
PT: # Quando você só precisa de uma REDUÇÃO do intermediário grande, evite construí-lo.
EN: # Example: the sum of all pairwise products.
PT: # Exemplo: a soma de todos os produtos par a par.
EN: # ✘ Materialises a 2000×2000 intermediate (32 MB) to then collapse it.
PT: # ✘ Materializa um intermediário 2000×2000 (32 MB) para depois colapsá-lo.
EN: # ✔ Algebra: Σᵢ Σⱼ aᵢbⱼ = (Σᵢ aᵢ)(Σⱼ bⱼ). No intermediate at all.
PT: # ✔ Álgebra: Σᵢ Σⱼ aᵢbⱼ = (Σᵢ aᵢ)(Σⱼ bⱼ). Sem nenhum intermediário.
EN: # These agree, but einsum lets you express contractions that have no
PT: # Os resultados concordam, mas einsum expressa contrações que não têm
EN: # matmul equivalent, and it fuses the multiply and the sum.
PT: # equivalente em matmul, e funde a multiplicação com a soma.
EN: # A contraction that is awkward otherwise: row-wise dot product of two matrices.
PT: # Uma contração difícil de outra forma: produto escalar linha a linha de duas matrizes.
EN: # builds a (1000, 50) intermediate
PT: # constrói um intermediário (1000, 50)
EN: # no intermediate
PT: # sem intermediário
EN: # Print the value in each cell — with only 20 cells, the numbers ARE the data.
PT: # Imprime o valor em cada célula: com só 20 células, os números SÃO os dados.
