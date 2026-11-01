<!-- en:150e2b87d2 -->
# 09 · Projeto do Módulo: Dois Algoritmos do Zero

**Módulo 02 — NumPy e Computação Numérica**

---

## O briefing

Implemente **regressão linear** e **clustering k-means** usando apenas NumPy: nada de scikit-learn até o final, quando vamos usá-lo para conferir o trabalho.

Isto não é tarefa para passar o tempo. Quando `LinearRegression().fit()` devolver um coeficiente suspeito no Módulo 11, ou quando o `KMeans` der clusters diferentes a cada execução, você vai saber exatamente por quê, porque terá escrito o loop.

## O que este projeto exercita

| Notebook | Conceito usado aqui |
| :--- | :--- |
| 01 | Vetorização em vez de loops Python |
| 02 | Shapes, views e cópias, `reshape(-1, 1)` |
| 03 | float64 para acumulação, números de condição |
| 04 | Broadcasting para distâncias par a par |
| 05 | `argmin`, máscaras booleanas, `bincount`, `keepdims` |
| 06 | `lstsq`, equações normais, autodecomposição |
| 07 | Geradores com seed, inicialização reproduzível |
| 08 | Evitar temporários, escolher a operação barata |

**Tempo estimado:** 120 minutos

<!-- en:020e256dec -->
---
# Parte 1: Regressão Linear

## O problema

Dada uma matriz de atributos `X` de shape `(n, p)` e alvos `y` de shape `(n,)`, encontre os coeficientes `β` que minimizam a soma dos resíduos ao quadrado:

$$\hat{\beta} = \arg\min_\beta \; \lVert X\beta - y \rVert^2$$

Vamos implementar três métodos de solução e compará-los em exatidão, velocidade e robustez.

<!-- en:b1eda8ef69 -->
## Como testar

<!-- en:d0ac851279 -->
## Onde os métodos diferem

Em dados bem condicionados, os três concordam. As diferenças aparecem exatamente onde o notebook 06 disse que apareceriam.

<!-- en:886911f71a -->
Os dois se ajustam igualmente bem aos dados, mas os coeficientes são absurdos: o modelo não consegue distinguir as colunas 0 e 2, então atribui pesos enormes que se compensam. **É assim que a multicolinearidade se manifesta**, e é por isso que os coeficientes de um modelo sem regularização em atributos correlacionados nunca devem ser interpretados.

O Ridge resolve isso:

<!-- en:0b10626cb2 -->
---
# Parte 2: Clustering k-means

## O algoritmo

1. Escolha `k` centroides iniciais.
2. **Atribua** cada ponto ao centroide mais próximo.
3. **Atualize** cada centroide para a média dos pontos atribuídos a ele.
4. Repita os passos 2 e 3 até as atribuições pararem de mudar.

O passo 2 é exatamente o exercício de broadcasting do notebook 04, e o passo 3 é o group-by com `bincount` do notebook 05.

<!-- en:d990e292c9 -->
> [!NOTE]
> Comparar `labels_` elemento a elemento estaria errado: o cluster 0 da nossa execução pode ser o cluster 2 do scikit-learn. O **Adjusted Rand Index** mede se os mesmos pontos terminaram juntos, que é a única coisa que importa.

<!-- en:8b546f28fb -->
## Por que `n_init` não é opcional

O k-means converge para um ótimo **local** que depende inteiramente de onde os centroides começaram. Executar uma só vez é apostar.

<!-- en:59ec596e59 -->
## Desempenho: a nossa versão e o scikit-learn

<!-- en:ca47a8a682 -->
Ser algumas vezes mais lento que o scikit-learn é um resultado **bom** para NumPy puro. A diferença vem de loops internos em Cython, do algoritmo de Elkan, que usa a desigualdade triangular para pular cálculos de distância, e de multithreading, não de algo errado na nossa abordagem.

A lição é a do notebook 01: **use a biblioteca em produção, mas saiba o que ela está fazendo.**

<!-- en:b9b77ae740 -->
## Verificação

<!-- en:f1884c5263 -->
## Sua vez: extensões

### Extensão 1: promova para o `dstoolkit` ⭐

Mova as duas classes para `src/dstoolkit/models.py`, adicione testes em `tests/test_models.py` que cubram as proteções de shape e a concordância com o sklearn, e importe-as de volta aqui.

### Extensão 2: mínimos quadrados ponderados ⭐⭐

Adicione um parâmetro `sample_weight`. A solução passa a ser `β = (XᵀWX)⁻¹XᵀWy`, com `W` diagonal. Implemente **sem** formar a matriz `W` de `(n, n)`: um conjunto de 100 000 amostras precisaria de 80 GB.

### Extensão 3: k-means em mini-lotes ⭐⭐

Atualize os centroides a partir de subconjuntos aleatórios em vez do conjunto completo. Compare a velocidade de convergência e a inércia final com a versão completa em 500 000 pontos.

### Extensão 4: silhouette score ⭐⭐⭐

Implemente o coeficiente de silhueta em NumPy vetorizado e use-o para escolher `k` em vez do cotovelo. Valide com `sklearn.metrics.silhouette_score`. Atenção à memória: a versão ingênua constrói uma matriz de distâncias `(n, n)`.

### Extensão 5: atributos polinomiais ⭐⭐

Adicione um parâmetro `degree` que expanda `X` em termos polinomiais antes do ajuste. Mostre como o número de condição explode com o grau, e que é o Ridge que torna o grau 10 utilizável.

### Extensão 6: o algoritmo de Elkan ⭐⭐⭐

Use a desigualdade triangular para pular cálculos de distância que não podem alterar uma atribuição. Meça o quanto isso reduz a diferença para o scikit-learn.

<!-- en:1c826c3fa1 -->
## O que você construiu

Dois estimadores no formato de produção que:

- seguem o **contrato do scikit-learn**, então se encaixam em um `Pipeline`;
- **validam as entradas** e falham com mensagens úteis;
- usam **`lstsq` em vez de `inv`** e continuam corretos em dados de posto incompleto;
- calculam distâncias par a par **por expansão em vez de um tensor de diferenças**, reduzindo a memória em 50× com dados realistas;
- **reiniciam a partir de fluxos independentes com seed** e mantêm o melhor resultado;
- concordam com o scikit-learn dentro da tolerância de ponto flutuante.

## Módulo 02 concluído: autoavaliação

- [ ] Por que uma lista Python de números é mais lenta *e* maior que um ndarray?
- [ ] Por que `arr[1:3]` se comporta de forma diferente de `lst[1:3]`?
- [ ] O que `axis=0` realmente colapsa?
- [ ] Quando `keepdims=True` importa, e o que quebra sem ele?
- [ ] Quais são as duas regras de broadcasting?
- [ ] Por que `int32_arr.sum()` não estoura, mas `int32_arr * 400` estoura?
- [ ] Por que você nunca deve chamar `np.linalg.inv` para resolver um sistema?
- [ ] O que um número de condição de 1e16 diz a você?
- [ ] Por que `np.random.seed` é uma escolha ruim para a reprodutibilidade?
- [ ] O que são strides, e por que `arr.T` não custa nada?

---

**Próximo:** Módulo 03 — pandas e Manipulação de Dados. Tudo o que você acabou de aprender sobre shapes, dtypes, views e vetorização vale diretamente: um DataFrame é um conjunto de arrays NumPy com rótulos e muita conveniência por cima.

<!-- comments -->
EN: # Fitted state. `field(init=False)` keeps these out of the constructor.
PT: # Estado ajustado. `field(init=False)` os mantém fora do construtor.
EN: # -- helpers ------------------------------------------------------------
PT: # -- auxiliares ---------------------------------------------------------
EN: # float64 throughout: notebook 03 showed what float32 does to accumulation.
PT: # float64 em tudo: o notebook 03 mostrou o que o float32 faz com a acumulação.
EN: # -- the three solvers --------------------------------------------------
PT: # -- os três métodos de solução -----------------------------------------
EN: # Ridge as augmented least squares: stack sqrt(alpha)*I below X.
PT: # Ridge como mínimos quadrados aumentados: empilha sqrt(alpha)*I abaixo de X.
EN: # This is more stable than forming XᵀX + alpha*I explicitly.
PT: # Isso é mais estável que formar XᵀX + alpha*I explicitamente.
EN: # never penalise the intercept
PT: # nunca penaliza o intercepto
EN: # ∇ = (2/n) Xᵀr, plus the ridge term.
PT: # ∇ = (2/n) Xᵀr, mais o termo do ridge.
EN: # -- public API ---------------------------------------------------------
PT: # -- API pública --------------------------------------------------------
EN: # Cross-check against scikit-learn — the real test.
PT: # Confere com o scikit-learn: o teste de verdade.
EN: # Ridge, checked the same way.
PT: # Ridge, conferido do mesmo jeito.
EN: # Build a badly conditioned design: two nearly identical columns.
PT: # Constrói um design mal condicionado: duas colunas quase idênticas.
EN: # almost a duplicate
PT: # quase uma duplicata
EN: # --- Left: predicted vs actual --------------------------------------------
PT: # --- Esquerda: previsto e real ----------------------------------------------
EN: # --- Right: gradient descent convergence -----------------------------------
PT: # --- Direita: convergência do gradiente descendente -------------------------
EN: # -- the two core steps -------------------------------------------------
PT: # -- os dois passos centrais --------------------------------------------
EN: # (n, 1)
PT: # (n, 1)
EN: # (k,)
PT: # (k,)
EN: # (n, k)
PT: # (n, k)
EN: # Broadcasting: (n,1) - 2*(n,k) + (k,) → (n,k)
PT: # Broadcasting: (n,1) - 2*(n,k) + (k,) → (n,k)
EN: # Tiny negatives can appear from cancellation; clip them away.
PT: # Pequenos valores negativos podem surgir por cancelamento; são eliminados com clip.
EN: # Distance from each point to its CLOSEST chosen centroid so far.
PT: # Distância de cada ponto até o centroide escolhido MAIS PRÓXIMO até agora.
EN: # Every point coincides with a centroid; fall back to uniform.
PT: # Todo ponto coincide com um centroide; recorre à distribuição uniforme.
EN: # --- ASSIGN: nearest centroid for every point ------------------
PT: # --- ATRIBUIÇÃO: centroide mais próximo de cada ponto -----------
EN: # --- UPDATE: centroid becomes the mean of its members ----------
PT: # --- ATUALIZAÇÃO: o centroide vira a média dos seus membros -----
EN: # An empty cluster: re-seed it at the worst-fitted point,
PT: # Um cluster vazio: reinicia-o no ponto com pior ajuste,
EN: # which is what scikit-learn does too.
PT: # que é o que o scikit-learn também faz.
EN: # One parent seed spawns independent streams — notebook 07's lesson.
PT: # Uma seed pai cria fluxos independentes: a lição do notebook 07.
EN: # Cluster LABELS are arbitrary — only the partition matters. Adjusted Rand
PT: # Os RÓTULOS dos clusters são arbitrários; só a partição importa. O Adjusted Rand
EN: # Index compares partitions while ignoring how they were numbered.
PT: # Index compara partições ignorando como elas foram numeradas.
EN: # --- Ground truth ----------------------------------------------------------
PT: # --- Verdade conhecida ------------------------------------------------------
EN: # --- Our clustering --------------------------------------------------------
PT: # --- O nosso clustering -----------------------------------------------------
EN: # --- The elbow curve -------------------------------------------------------
PT: # --- A curva do cotovelo ----------------------------------------------------
