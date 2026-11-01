<!-- en:9596a3b93e -->
# 06 · Álgebra Linear

**Módulo 02 — NumPy e Computação Numérica**

---

## O que você vai aprender

1. Vetores: produtos escalares, normas e o que "similaridade" significa geometricamente.
2. Multiplicação de matrizes: `@` e `*`, e por que a distinção importa.
3. Como resolver sistemas lineares, e por que **você nunca deve inverter uma matriz**.
4. Mínimos quadrados: como deduzir a regressão linear do zero.
5. Autovalores e SVD, a engrenagem por trás do PCA.
6. Números de condição: como reconhecer um problema numericamente sem salvação.

**Tempo estimado:** 90 minutos

> Este é o núcleo matemático dos Módulos 11 a 19. Todo modelo que você vai ajustar é álgebra linear por baixo.

<!-- en:2d62cb0862 -->
## 1. Vetores

### O produto escalar

`a · b = Σ aᵢbᵢ`. Geometricamente, é `|a| |b| cos θ`: mede o quanto dois vetores apontam na mesma direção.

<!-- en:9d41130a53 -->
### Normas: como medir o comprimento

<!-- en:0c472c0799 -->
> [!NOTE]
> L1 e L2 são exatamente as penalidades das regressões Lasso e Ridge (Módulo 11). Os cantos da L1 levam os coeficientes a exatamente zero, produzindo seleção de atributos; a bola suave da L2 apenas os reduz.

<!-- en:c093fb9405 -->
### Similaridade de cosseno

Normalize para eliminar a magnitude, e só a direção permanece. É assim que embeddings de texto, sistemas de recomendação e buscas por vizinhos mais próximos comparam itens.

<!-- en:41a2a491cc -->
## 2. Multiplicação de matrizes

### `@` não é `*`

Esta é a fonte mais comum de respostas erradas para quem vem do MATLAB ou da matemática.

<!-- en:b39a10ef04 -->
### A regra dos shapes

`(m, n) @ (n, p) → (m, p)`. As **dimensões internas precisam ser iguais** e desaparecem; as externas sobrevivem.

<!-- en:075f2b6e5a -->
### Arrays 1-D recebem tratamento especial

<!-- en:aebec0de4c -->
> [!TIP]
> Essa conveniência também é uma armadilha: como arrays 1-D se adaptam à orientação que faz a multiplicação funcionar, um erro real de shape pode passar despercebido. Quando os shapes importam, seja explícito com `reshape(-1, 1)`.

<!-- en:b92248ec4d -->
### Multiplicação em lote

`@` opera nos **dois últimos eixos** e faz broadcast de tudo o que vem antes deles, que é exatamente como redes neurais processam um lote de amostras de uma vez.

<!-- en:fdbccac88d -->
## 3. Como resolver sistemas lineares

### Nunca inverta uma matriz

Para resolver `Ax = b`, a álgebra diz `x = A⁻¹b`. **Não calcule dessa forma.** `np.linalg.solve` usa a decomposição LU: é mais rápido e bem mais exato.

<!-- en:599eb48c67 -->
### O número de condição

`cond(A)` estima o quanto uma pequena mudança na entrada pode ser amplificada na saída. Como regra prática, você perde cerca de `log10(cond)` dígitos significativos.

<!-- en:29789cf5db -->
> [!WARNING]
> Um número de condição alto é o motivo de você **padronizar atributos** antes de ajustar um modelo linear, e de colunas perfeitamente correlacionadas quebrarem uma regressão. Não é uma curiosidade numérica: é a razão de o `Ridge` somar `alpha` à diagonal (Módulo 11).

<!-- en:a95e495fe3 -->
## 4. Mínimos quadrados: regressão linear a partir dos princípios básicos

Em um sistema sobredeterminado (mais linhas que incógnitas), geralmente não há solução exata. Os mínimos quadrados encontram o `x` que minimiza `‖Ax − b‖²`.

As **equações normais** dão a forma fechada:

$$\hat{\beta} = (X^T X)^{-1} X^T y$$

<!-- en:5f658b1396 -->
Os três concordam aqui porque o problema é bem condicionado. Ainda assim, `lstsq` é a escolha certa: ele usa SVD, então lida com matrizes de posto incompleto (o caso em que dois atributos são perfeitamente colineares), em vez de gerar um erro ou devolver lixo.

<!-- en:1717ccd9d1 -->
## 5. Autovalores e autovetores

Um autovetor de `A` é uma direção que `A` apenas *escala*, sem nunca girar: `Av = λv`. O autovalor `λ` é esse fator de escala.

<!-- en:18cbf90164 -->
### Por que isso importa: PCA

A Análise de Componentes Principais é a autodecomposição da matriz de covariância. Os autovetores são as direções principais; os autovalores são a variância explicada ao longo de cada uma.

<!-- en:293b735311 -->
## 6. Decomposição em Valores Singulares

A SVD fatora **qualquer** matriz, quadrada ou não, invertível ou não, como `A = U Σ Vᵀ`. Ela sustenta o PCA, sistemas de recomendação, compressão de imagens e o próprio `lstsq`.

<!-- en:5f51ab0cd6 -->
### Aproximação de posto baixo

Manter apenas os `k` maiores valores singulares dá a melhor aproximação possível de posto `k`. Isso é compressão, e também é remoção de ruído.

<!-- en:fbcd449a6b -->
## 7. Referência rápida

<!-- en:62acef0661 -->
## 8. Exercícios

### Exercício 6.1: shapes

Diga o shape do resultado ou "erro":

```python
(3, 4) @ (4, 5)      (3, 4) @ (3, 4)      (4,) @ (4,)
(3, 4) @ (4,)        (10, 3, 4) @ (4, 2)  (3,) @ (3, 4)
```

### Exercício 6.2: matriz de similaridade de cosseno

Dado `E` de shape `(1000, 128)` (1000 documentos, embeddings de 128 dimensões), calcule a matriz completa `(1000, 1000)` de similaridade de cosseno **sem loop**. *Dica:* normalize as linhas primeiro; depois, um produto de matrizes faz o resto.

### Exercício 6.3: implemente mínimos quadrados

Escreva `fit_linear(X, y, fit_intercept=True)`, que devolve os coeficientes por meio de `np.linalg.lstsq`. Valide com `sklearn.linear_model.LinearRegression` em dados aleatórios. Depois, adicione um parâmetro `alpha` que implemente Ridge: `β = (XᵀX + αI)⁻¹Xᵀy`, e confirme que o resultado coincide com `sklearn.linear_model.Ridge`.

### Exercício 6.4: número de condição e colinearidade

Construa `X` com duas colunas quase idênticas (correlação de 0.9999). Informe `cond(XᵀX)`. Ajuste pelas equações normais e por `lstsq` e compare os coeficientes. Depois, padronize as colunas e informe de novo o número de condição.

### Exercício 6.5: compressão de imagens com SVD

Construa uma "imagem" em tons de cinza de 200×200 (qualquer padrão estruturado). Reconstrua-a nos postos 5, 20 e 50, e faça um gráfico das quatro versões lado a lado, com o armazenamento que cada uma exige.

<!-- en:5a1da29c0f -->
## Principais lições

- **`@` é multiplicação de matrizes; `*` é elemento a elemento.** Confundi-los é o bug de álgebra linear mais comum em Python.
- `(m, n) @ (n, p) → (m, p)`: as dimensões internas precisam ser iguais e desaparecem.
- **Nunca use `np.linalg.inv` para resolver um sistema.** `solve` é mais rápido e muito mais exato; `lstsq` também sobrevive a posto incompleto.
- Mínimos quadrados *são* regressão linear. `lstsq` é o que o scikit-learn chama.
- **`cond(A)`** diz quantos dígitos um problema vai custar. Números de condição altos são o motivo de padronizar atributos e de o Ridge existir.
- **PCA é a autodecomposição da matriz de covariância.** Use `eigh` em matrizes simétricas.
- **A SVD funciona em qualquer matriz** e dá a melhor aproximação de posto baixo: compressão, remoção de ruído e recomendação se apoiam nela.

---

**Próximo:** [07 · Aleatoriedade e Simulação](07_random_and_simulation.ipynb): números aleatórios reproduzíveis e como responder a perguntas difíceis por meio de simulação.

<!-- comments -->
EN: # Row-wise norms of a matrix — the everyday case.
PT: # Normas por linha de uma matriz: o caso do dia a dia.
EN: # same direction, 5× the length
PT: # mesma direção, 5× o comprimento
EN: # (2, 3)
PT: # (2, 3)
EN: # (3,)
PT: # (3,)
EN: # (2,)
PT: # (2,)
EN: # vector @ vector gives a scalar, not a matrix.
PT: # vetor @ vetor dá um escalar, não uma matriz.
EN: # 32 matrices of shape (4, 3)
PT: # 32 matrizes de shape (4, 3)
EN: # one shared weight matrix
PT: # uma matriz de pesos compartilhada
EN: # On an ill-conditioned system the difference stops being academic.
PT: # Em um sistema mal condicionado, a diferença deixa de ser teórica.
EN: # A Hilbert matrix is the textbook example of near-singularity.
PT: # Uma matriz de Hilbert é o exemplo clássico de quase singularidade.
EN: # Simulate data with a known answer, so we can check our work.
PT: # Simula dados com resposta conhecida, para podermos conferir o trabalho.
EN: # Add a column of ones so the intercept is just another coefficient.
PT: # Adiciona uma coluna de uns, para o intercepto ser só mais um coeficiente.
EN: # --- Method 1: the normal equations, literally ------------------------------
PT: # --- Método 1: as equações normais, literalmente ----------------------------
EN: # --- Method 2: the same equations, solved rather than inverted --------------
PT: # --- Método 2: as mesmas equações, resolvidas em vez de invertidas ----------
EN: # --- Method 3: lstsq — what you should actually use -------------------------
PT: # --- Método 3: lstsq, o que você deve realmente usar ------------------------
EN: # Perfect collinearity: feature 3 is an exact copy of feature 1.
PT: # Colinearidade perfeita: o atributo 3 é uma cópia exata do atributo 1.
EN: # Cross-check against scikit-learn — the same computation it performs.
PT: # Confere com o scikit-learn: é o mesmo cálculo que ele faz.
EN: # Verify the definition for the first eigenpair.
PT: # Verifica a definição para o primeiro par de autovalor e autovetor.
EN: # Correlated 2-D data, so there is an obvious principal direction.
PT: # Dados 2-D correlacionados, para haver uma direção principal óbvia.
EN: # eigh: symmetric input
PT: # eigh: entrada simétrica
EN: # eigh returns ascending order; reverse so component 1 is the largest.
PT: # eigh devolve em ordem crescente; inverte para o componente 1 ser o maior.
EN: # --- Left: the data with its principal axes -------------------------------
PT: # --- Esquerda: os dados com seus eixos principais ---------------------------
EN: # Scale each axis by its standard deviation so the arrow length is meaningful.
PT: # Escala cada eixo pelo desvio padrão, para o comprimento da seta ter significado.
EN: # --- Right: the data rotated onto its principal axes ----------------------
PT: # --- Direita: os dados girados para seus eixos principais -------------------
EN: # The projected components are uncorrelated — that is the point of PCA.
PT: # Os componentes projetados não têm correlação: esse é o objetivo do PCA.
EN: # Cross-check with scikit-learn.
PT: # Confere com o scikit-learn.
EN: # A matrix that is genuinely low-rank plus a little noise.
PT: # Uma matriz realmente de posto baixo, mais um pouco de ruído.
