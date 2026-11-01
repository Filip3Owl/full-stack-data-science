<!-- en:f52a169347 -->
# Módulo 02 — NumPy e Computação Numérica

> **Parte I — Fundamentos** · Tempo estimado: **10–13 horas** · Pré-requisitos: [Módulo 01](../01-python-foundations/)

Toda biblioteca numérica deste curso (pandas, scikit-learn, statsmodels, SciPy, PyTorch) é uma camada sobre o ndarray. Este módulo é onde você para de tratar essa camada como mágica. Você vai aprender *por que* o código vetorizado é rápido, não só que ele é, e vai terminar implementando regressão linear e k-means usando apenas NumPy.

<p align="center">
  <img src="../../assets/images/numpy_memory_layout.png" alt="Uma lista Python guarda ponteiros para objetos inteiros espalhados; um ndarray guarda um único buffer contíguo e tipado" width="760">
</p>

---

## Por que este módulo existe

O Módulo 01 ensinou você a escrever Python correto. Isso não basta: Python correto que percorre 10 milhões de linhas em um loop é inutilizável, e código NumPy que *parece* correto pode estar errado sem nenhum aviso, de maneiras que um loop `for` nunca estaria.

| O hábito | O que ele evita |
| :--- | :--- |
| Pensar em arrays inteiros | Um loop de 40 minutos onde uma expressão de 2 segundos bastaria |
| Distinguir views de cópias | Modificar um array por meio de uma fatia que você achava ser só sua |
| Escolher dtypes deliberadamente | Um contador `int32` que dá a volta e vira um número negativo sem aviso |
| Ler o shape antes de confiar no resultado | Um broadcast `(1000,) + (1000,1)` que devolve um bug de um milhão de elementos |
| Entender `axis=` | Padronizar pela dimensão errada e nunca perceber |
| Resolver em vez de inverter | Coeficientes destruídos por uma matriz mal condicionada |
| Usar seed em um `Generator`, não no estado global | Resultados que ninguém, nem você, consegue reproduzir |

---

## Objetivos de aprendizagem

Ao final deste módulo, você será capaz de:

1. **Explicar a diferença de desempenho** entre uma lista Python e um ndarray em termos de layout de memória, e medi-la você mesmo.
2. **Criar, indexar, remodelar e combinar** arrays em N dimensões, e prever se uma operação devolve uma view ou uma cópia.
3. **Escolher dtypes deliberadamente**, antecipando estouro de inteiros, promoção de tipos e os limites de precisão do `float32`.
4. **Aplicar as duas regras de broadcasting** para alinhar shapes de propósito, e reconhecer shapes que fazem broadcast para a resposta errada.
5. **Reduzir ao longo de qualquer eixo** e substituir loops condicionais por máscaras booleanas, `np.where` e indexação avançada.
6. **Usar álgebra linear corretamente** (`solve` e `lstsq` no lugar de `inv`, autodecomposição, SVD) e interpretar um número de condição.
7. **Simular e fazer bootstrap** com a API moderna de `Generator`, com uma reprodutibilidade que resiste ao paralelismo.
8. **Diagnosticar desempenho e memória** por meio de strides, temporários e arrays mapeados em memória, e saber quando o NumPy chegou ao limite.
9. **Implementar um algoritmo real do zero** seguindo o contrato de estimadores do scikit-learn.

---

## Notebooks

Siga-os em ordem: cada um se apoia no anterior.

| # | Notebook | O que você vai aprender | Tempo |
| :-: | :--- | :--- | :-: |
| 01 | [Por que o NumPy Existe](notebooks/01_why_numpy_exists.ipynb) | Layout de memória, o que "vetorizado" realmente significa, ufuncs, como medir a diferença, quando *não* usar NumPy | 60 min |
| 02 | [Fundamentos do ndarray](notebooks/02_ndarray_fundamentals.ipynb) | Criação, `shape`/`ndim`/`size`/`dtype`, indexação em N dimensões, **views e cópias**, remodelagem, empilhamento | 90 min |
| 03 | [Dtypes e Precisão Numérica](notebooks/03_dtypes_and_precision.ipynb) | O zoológico de dtypes, estouro silencioso, promoção no NumPy 2.0, prós e contras do `float32`, épsilon, `NaN`/`inf`, redução de tipos | 75 min |
| 04 | [Broadcasting](notebooks/04_broadcasting.ipynb) | As duas regras, como ler o erro, `np.newaxis`, o bug silencioso, padronização e distâncias par a par, a armadilha de memória | 75 min |
| 05 | [Agregações e Máscaras Booleanas](notebooks/05_aggregations_and_masking.ipynb) | `axis=` desmistificado, `keepdims`, máscaras, `where`/`select`/`clip`, indexação avançada, `unique`/`bincount`/`histogram` | 90 min |
| 06 | [Álgebra Linear](notebooks/06_linear_algebra.ipynb) | Produtos escalares e normas, `@` e `*`, resolução de sistemas, mínimos quadrados, autovalores, SVD, números de condição | 90 min |
| 07 | [Aleatoriedade e Simulação](notebooks/07_random_and_simulation.ipynb) | A API de `Generator`, `SeedSequence`/`spawn`, distribuições, amostragem, Monte Carlo, o bootstrap | 75 min |
| 08 | [Desempenho e Memória](notebooks/08_performance_and_memory.ipynb) | Strides, ordem C e Fortran, operações in-place, medição honesta de memória, `np.memmap`, como escolher a ferramenta certa | 75 min |
| 09 | [Projeto do Módulo](notebooks/09_module_project.ipynb) | Regressão linear **e** k-means do zero, seguindo o contrato de estimadores do sklearn | 120 min |
| 10 | [Soluções dos Exercícios](notebooks/10_exercise_solutions.ipynb) | Respostas comentadas com o raciocínio — **abra só depois de tentar os exercícios** | — |

---

## Figuras principais

Os diagramas abaixo são usados nos notebooks. Eles são gerados por [`scripts/generate_figures.py`](../../scripts/generate_figures.py), e você pode gerá-los de novo a qualquer momento com `.venv/bin/python scripts/generate_figures.py`.

| Figura | Notebook |
| :--- | :--- |
| ![Layout de memória do NumPy](../../assets/images/numpy_memory_layout.png) | 01 |
| ![Views e cópias](../../assets/images/views_vs_copies.png) | 02 |
| ![Regras de broadcasting](../../assets/images/broadcasting_rules.png) | 04 |
| ![Semântica dos eixos](../../assets/images/axis_semantics.png) | 05 |

---

## Exercícios

Cada um dos notebooks de 01 a 08 termina com cinco exercícios. A [página de exercícios](exercises/README.md) reúne todos os 40, e as soluções comentadas estão no [notebook 10](notebooks/10_exercise_solutions.ipynb). O projeto acrescenta seis extensões abertas.

> **Faça os exercícios antes de ler as soluções.** Ler uma solução dá a sensação de aprender, mas não é aprender.

---

## As três armadilhas que este módulo existe para evitar

Se você não lembrar de mais nada, lembre-se destas. Cada uma tem um notebook dedicado porque cada uma produz *números errados, e não uma exceção*.

1. **Uma fatia é uma view.** `arr[1:3] = 0` grava no array original. Listas copiam ao fatiar; arrays não. Notebook 02.
2. **Inteiros dão a volta sem aviso.** Uma conta com `np.int32` que passa de 2 147 483 647 não gera erro: o resultado fica negativo. Notebook 03.
3. **`(n,)` e `(n, 1)` fazem broadcast para `(n, n)`.** Sem erro, com uma resposta no shape errado e uma média sobre a dimensão errada. Notebook 04.

---

## Checklist de autoavaliação

Você está pronto para o Módulo 03 quando conseguir responder a tudo isto *sem consultar nada*:

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

## Leitura complementar

| Recurso | Por que vale o seu tempo |
| :--- | :--- |
| [NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html) | O tour oficial; uma segunda passada rápida pelos notebooks 01 e 02. |
| [Array programming with NumPy](https://www.nature.com/articles/s41586-020-2649-2) (Harris et al., *Nature* 2020) | O artigo dos mantenedores, o relato mais claro de *por que* o design é como é. |
| [Documentação de broadcasting do NumPy](https://numpy.org/doc/stable/user/basics.broadcasting.html) | As regras formalizadas, com a tabela de shapes que vale memorizar. |
| [What Every Computer Scientist Should Know About Floating-Point](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) (Goldberg) | A referência por trás do notebook 03. Denso, mas vale a pena. |
| [Guia de amostragem aleatória do NumPy](https://numpy.org/doc/stable/reference/random/index.html) | Por que `Generator` substituiu `RandomState` e como `SeedSequence` funciona. |
| [Introduction to Linear Algebra](https://math.mit.edu/~gs/linearalgebra/) (Strang) | Se o notebook 06 pareceu superficial, este é o livro para ler junto com os Módulos 11 a 19. |

---

**Próximo:** Módulo 03 — pandas e Manipulação de Dados.
