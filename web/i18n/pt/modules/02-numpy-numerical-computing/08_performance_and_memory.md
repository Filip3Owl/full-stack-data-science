<!-- en:e716b7b547 -->
# 08 · Desempenho e Memória

**Módulo 02 — NumPy e Computação Numérica**

---

## O que você vai aprender

1. **Strides**: como um único buffer plano finge ser um array de N dimensões.
2. Ordem C e ordem Fortran, e por que percorrer do jeito errado é mais lento.
3. Operações in-place e como parar de alocar temporários.
4. Como ler o uso de memória com honestidade, incluindo intermediários ocultos.
5. `np.memmap` para arrays maiores que a RAM.
6. Como saber quando o NumPy chegou ao limite, e o que vem depois.

**Tempo estimado:** 75 minutos

<!-- en:1c6a60cc08 -->
## 1. Strides: o truque por trás de tudo

Um ndarray é um **buffer plano** mais uma regra para interpretá-lo. Essa regra é a tupla `strides`: quantos **bytes** avançar para andar uma posição em cada eixo.

<!-- en:4b811d6b72 -->
É por isso que remodelar e transpor não custam nada: eles só reescrevem os strides, nunca os dados.

<!-- en:ebe680850d -->
### Ordem C e ordem Fortran

<!-- en:6e5f5fcf27 -->
### Por que a ordem importa: localidade de cache

CPUs buscam a memória em **linhas de cache** de 64 bytes. Quando você lê dados contíguos, cada busca entrega vários valores úteis. Quando você salta pela memória, a maior parte de cada busca é desperdiçada.

<!-- en:ed9a51c4a2 -->
> [!TIP]
> Você raramente escolhe a ordem de forma explícita, mas a herda. `arr.T`, `df.values` de uma origem orientada a colunas e algumas rotinas do SciPy devolvem arrays que não são C-contíguos. Se uma operação seguinte estiver inesperadamente lenta, confira `.flags` e considere `np.ascontiguousarray`.

<!-- en:7022d0a110 -->
## 2. Temporários: as alocações ocultas

Cada intermediário de uma expressão aloca um array inteiro. Uma cadeia de operações sobre um array grande pode usar várias vezes a memória que você espera.

<!-- en:ff365cadd0 -->
O ganho de velocidade é modesto; o ganho de **memória** é o verdadeiro prêmio. Trabalhar in-place é o que permite a um processo sequer caber na RAM.

<!-- en:c3555c7147 -->
> [!WARNING]
> `y *= 2` modifica no lugar, então, se `y` for uma **view** de outra coisa, você acabou de alterar o original. É a lição do notebook 02 encontrando o conselho de desempenho: in-place é eficiente *e* perigoso, e você precisa saber quais arrays são seus.

<!-- en:85a1ccf226 -->
## 3. Como medir a memória com honestidade

<!-- en:a980ef77f3 -->
## 4. Arrays maiores que a RAM: `np.memmap`

Um array mapeado em memória fica em disco e é carregado por partes sob demanda. Ele se comporta como um ndarray, mas é limitado pelo disco, não pela RAM.

<!-- en:118ae100c8 -->
## 5. Como escolher a ferramenta certa

O NumPy é rápido quando o trabalho pode ser escrito como operações com arrays. Quando não pode, a resposta é outra ferramenta, não um array mais esperto.

<!-- en:457a2e5640 -->
### A escada de opções

| Sintoma | Próximo passo |
| :--- | :--- |
| Um loop `for` de Python sobre os elementos | Reescreva como operações com arrays |
| O loop é realmente sequencial | Procure uma rotina do SciPy (`lfilter`, `cumsum`, `convolve`) |
| Continua sequencial, sem biblioteca | **Numba** `@njit`, geralmente uma mudança de uma linha |
| O pico de memória é a restrição | Operações in-place, blocos, `np.memmap` |
| Os dados não cabem em uma máquina | **Dask** ou **Spark** (Módulo 20) |
| Colunas heterogêneas ou com rótulos | **pandas** (Módulo 03) |
| Matrizes muito grandes, quase todas com zeros | `scipy.sparse` |

<!-- en:142fbafed6 -->
> [!NOTE]
> Atributos categóricos com one-hot encoding e matrizes TF-IDF de texto quase sempre têm mais de 99% de zeros. `scipy.sparse` é o motivo de um modelo bag-of-words com um vocabulário de 50 000 palavras caber na memória (Módulo 18).

<!-- en:e414b7d276 -->
## 6. Um checklist de desempenho

Siga esta lista antes de recorrer a uma máquina maior.

<!-- en:f1aaa9634e -->
## 7. Exercícios

### Exercício 8.1: preveja os strides

Para `a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)`, diga os `strides` de: `a`, `a.T`, `a[::2]`, `a[:, ::2, :]`, `a.reshape(6, 4)`. Verifique cada um.

### Exercício 8.2: por linhas ou por colunas

Construa um array 3000×3000 nas duas ordens. Meça o tempo de `sum(axis=0)` e `sum(axis=1)` em cada um. Explique o padrão em termos de linhas de cache e preveja qual combinação é a mais rápida antes de medir.

### Exercício 8.3: elimine os temporários

Reescreva `result = np.sqrt(a**2 + b**2) / (c + 1)` usando apenas parâmetros `out=` e no máximo dois buffers. Verifique se os resultados coincidem e meça o pico de alocações.

### Exercício 8.4: estatísticas em blocos

Escreva `chunked_stats(path, shape, dtype, chunk_rows)`, que calcula a média e o desvio padrão de um array mapeado em memória **em uma passada**, sem carregá-lo inteiro. *Dica:* acumule a contagem, a soma e a soma dos quadrados, e depois tome cuidado com o cancelamento catastrófico que você viu no notebook 03.

### Exercício 8.5: saiba quando parar

Implemente um máximo acumulado (`out[i] = max(x[:i+1])`) de três formas: um loop Python, `np.maximum.accumulate` e uma versão em blocos. Meça o tempo das três em 10 milhões de elementos e explique a ordem.

<!-- en:3f4dac2c6a -->
## Principais lições

- Um ndarray é um **buffer plano mais strides**. Remodelar e transpor só reescrevem os strides, e por isso não custam nada.
- **A contiguidade rege o comportamento do cache.** Reduza ao longo do eixo que o seu array guarda de forma contígua; confira `.flags` quando algo estiver inesperadamente lento.
- Cada intermediário de uma expressão **aloca um array inteiro**. Use `out=` ou operadores in-place quando a memória for a restrição.
- **Operações in-place em uma view modificam o original**: eficiência e a armadilha do aliasing são o mesmo mecanismo.
- `.base is None` indica se um array é realmente dono da própria memória.
- **`np.memmap`** lida com arrays maiores que a RAM; reduções em blocos generalizam o padrão.
- Quando um loop for realmente sequencial, procure uma **rotina do SciPy** antes de recorrer ao Numba.
- Use **`scipy.sparse`** em matrizes quase todas com zeros, como one-hot e TF-IDF sempre são.

---

**Próximo:** [09 · Projeto do Módulo](09_module_project.ipynb): implemente regressão linear e k-means do zero, usando apenas o que este módulo ensinou.

<!-- comments -->
EN: # row-major (default)
PT: # por linhas (padrão)
EN: # column-major
PT: # por colunas
EN: # Summing along axis 0 walks DOWN columns; axis 1 walks ACROSS rows.
PT: # Somar ao longo do eixo 0 percorre as colunas PARA BAIXO; o eixo 1 percorre as linhas DE LADO.
EN: # When does the copy pay for itself?
PT: # Quando a cópia se paga?
EN: # a view, F-contiguous
PT: # uma view, F-contígua
EN: # pays a copy up front
PT: # paga uma cópia logo de início
EN: # The in-place operators do the same thing more readably, when you own the array.
PT: # Os operadores in-place fazem o mesmo com mais legibilidade, quando o array é seu.
EN: # no temporary
PT: # sem temporário
EN: # in-place through a view
PT: # in-place por meio de uma view
EN: # 40 MB — small enough for a notebook, real in principle
PT: # 40 MB: pequeno o bastante para um notebook, real em princípio
EN: # Create and write in chunks, never holding it all in memory.
PT: # Cria e grava em blocos, sem nunca manter tudo na memória.
EN: # Reopen read-only and compute over it.
PT: # Reabre em modo somente leitura e calcula sobre ele.
EN: # Chunked reduction is the pattern that generalises to genuinely huge files.
PT: # A redução em blocos é o padrão que se generaliza para arquivos realmente enormes.
EN: # A genuinely sequential algorithm: each step depends on the previous one.
PT: # Um algoritmo realmente sequencial: cada passo depende do anterior.
EN: # SciPy exposes the same recurrence as a digital filter, implemented in C.
PT: # O SciPy expõe a mesma recorrência como um filtro digital, implementado em C.
EN: # The first element differs because lfilter starts from zero rather than x[0].
PT: # O primeiro elemento difere porque lfilter parte de zero em vez de x[0].
EN: # Sparse matrices: when most entries are zero, storing them is pure waste.
PT: # Matrizes esparsas: quando a maioria das entradas é zero, guardá-las é puro desperdício.
EN: # Fill 0.1% of the entries.
PT: # Preenche 0.1% das entradas.
EN: # Clean up the memmap file.
PT: # Limpa o arquivo memmap.
