<!-- en:bc688fc4f4 -->
# Módulo 02 — Exercícios

Cada um dos notebooks de 01 a 08 termina com cinco exercícios. Esta página reúne todos os 40 em um só lugar, para você trabalhar neles sem precisar rolar a tela, e indica onde estão as soluções comentadas.

> [!WARNING]
> **Tente cada exercício antes de abrir a solução.** Ler uma resposta pronta produz uma sensação convincente de entendimento, que desaparece no momento em que você encara uma célula em branco. Dez minutos empacado ensinam mais do que uma hora de leitura.

---

## Onde estão as soluções

Todas as soluções estão em um único notebook executado:

📓 **[10 · Soluções dos Exercícios](../notebooks/10_exercise_solutions.ipynb)**

Cada solução inclui o raciocínio e, quando mais de uma resposta é defensável, as alternativas com seus prós e contras. Várias vão além da pergunta feita, porque a parte interessante costuma ser *por que* a abordagem óbvia é lenta ou está errada.

---

## A lista de exercícios

### Notebook 01 — Por que o NumPy Existe

| # | Exercício | Habilidade |
| :-- | :--- | :--- |
| 1.1 | Meça você mesmo | Metodologia honesta de medição de tempo |
| 1.2 | A pergunta honesta sobre memória | `nbytes` e `sys.getsizeof` |
| 1.3 | Encontre o ponto de equilíbrio | Custo fixo por chamada de uma ufunc |
| 1.4 | Encontre a desvetorização | Reconhecer um loop Python escondido |
| 1.5 | `out=` em um loop | Reutilização de buffers, custo de alocação |

### Notebook 02 — Fundamentos do ndarray

| # | Exercício | Habilidade |
| :-- | :--- | :--- |
| 2.1 | Preveja os shapes | Aritmética de shapes antes de executar o código |
| 2.2 | View ou cópia? | `.base`, fatiamento e indexação avançada |
| 2.3 | Conserte o bug de modificação | Cópia defensiva nas fronteiras da API |
| 2.4 | Construa um tabuleiro de xadrez | Atribuição em fatias com passo |
| 2.5 | Remodele para o scikit-learn | `reshape(-1, 1)`, `ravel`, o contrato 2-D |

### Notebook 03 — Dtypes e Precisão Numérica

| # | Exercício | Habilidade |
| :-- | :--- | :--- |
| 3.1 | Preveja o dtype | Regras de promoção do NumPy 2.0 |
| 3.2 | Provoque e depois evite um estouro | Inteiros que dão a volta, promoção deliberada de tipo |
| 3.3 | Quando o `float32` não basta? | Orçamento de precisão |
| 3.4 | Construa um relatório de memória | Redução segura de tipos com `iinfo`/`finfo` |
| 3.5 | Cancelamento catastrófico na prática | `np.spacing`, fórmulas numericamente estáveis |

### Notebook 04 — Broadcasting

| # | Exercício | Habilidade |
| :-- | :--- | :--- |
| 4.1 | Compatível ou não? | Aplicar as duas regras à mão |
| 4.2 | Conserte o bug silencioso | `(n,)` e `(n, 1)` |
| 4.3 | Normalize linhas em vez de colunas | `axis=` e `keepdims=True` juntos |
| 4.4 | Distância até os centroides | `np.newaxis`, distâncias par a par |
| 4.5 | Evite o intermediário | Expansão algébrica em vez de um tensor `(n, m)` |

### Notebook 05 — Agregações e Máscaras Booleanas

| # | Exercício | Habilidade |
| :-- | :--- | :--- |
| 5.1 | Treino de eixos | Qual dimensão uma redução colapsa |
| 5.2 | A armadilha do keepdims | Fazer broadcast de um resultado de volta sobre a origem |
| 5.3 | Prática de máscaras | Combinar condições com `&`, `|`, `~` |
| 5.4 | Group-by do zero | `unique(return_inverse=True)` e `bincount` |
| 5.5 | Top-k por linha | `argpartition` e `argsort` |

### Notebook 06 — Álgebra Linear

| # | Exercício | Habilidade |
| :-- | :--- | :--- |
| 6.1 | Shapes | `@` e `*`, compatibilidade de dimensões |
| 6.2 | Matriz de similaridade de cosseno | Normas, normalização de linhas, uma multiplicação de matrizes |
| 6.3 | Implemente mínimos quadrados | Equações normais e `lstsq` |
| 6.4 | Número de condição e colinearidade | `np.linalg.cond`, quando os números não têm salvação |
| 6.5 | Compressão de imagens com SVD | SVD truncada, posto e armazenamento |

### Notebook 07 — Aleatoriedade e Simulação

| # | Exercício | Habilidade |
| :-- | :--- | :--- |
| 7.1 | Quebre e conserte a reprodutibilidade | `Generator` e seed global |
| 7.2 | Escolha a distribuição | Associar um gerador a um processo |
| 7.3 | O problema do aniversário | Monte Carlo contra uma fórmula fechada |
| 7.4 | Bootstrap de uma diferença | Intervalos de confiança por reamostragem |
| 7.5 | Tamanho de amostra por simulação | Poder estatístico sem fórmula |

### Notebook 08 — Desempenho e Memória

| # | Exercício | Habilidade |
| :-- | :--- | :--- |
| 8.1 | Preveja os strides | Strides, `itemsize`, por que `.T` não custa nada |
| 8.2 | Por linhas ou por colunas | Ordem na memória e localidade de cache |
| 8.3 | Elimine os temporários | Operadores in-place, `out=` |
| 8.4 | Estatísticas em blocos | Acumulação em uma passada com memória limitada |
| 8.5 | Saiba quando parar | Reconhecer o que o NumPy ainda consegue fazer em C |

### Notebook 09 — Projeto do Módulo

O projeto termina com seis extensões abertas (de ⭐ a ⭐⭐⭐). Elas não têm uma única resposta certa: são o mais perto que este módulo chega do trabalho real.

| # | Extensão | Dificuldade |
| :-- | :--- | :-- |
| 1 | Promova os dois estimadores para o `dstoolkit` | ⭐ |
| 2 | Mínimos quadrados ponderados sem uma matriz `(n, n)` | ⭐⭐ |
| 3 | k-means em mini-lotes | ⭐⭐ |
| 4 | Silhouette score vetorizado | ⭐⭐⭐ |
| 5 | Atributos polinomiais e o número de condição | ⭐⭐ |
| 6 | O algoritmo de Elkan | ⭐⭐⭐ |

---

## Como conferir o seu próprio trabalho

Além de comparar com as soluções, três hábitos valem para qualquer caso, e o terceiro é específico do NumPy e vai pegar a maioria dos seus bugs:

1. **Escreva um teste.** Se você não consegue dizer qual é a saída correta, ainda não entendeu o problema.
2. **Restart & Run All.** Se a sua solução só funciona quando as células rodam na ordem em que você clicou nelas, ela não funciona.
3. **Imprima o shape e o dtype, sempre.** A maioria dos bugs de NumPy é um shape que fez broadcast sem você querer ou um dtype que foi promovido sem você notar. Nenhum dos dois gera exceção.
