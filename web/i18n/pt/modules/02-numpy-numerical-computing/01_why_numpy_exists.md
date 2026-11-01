<!-- en:df7fbe5542 -->
# 01 · Por que o NumPy Existe

**Módulo 02 — NumPy e Computação Numérica**

---

## O que você vai aprender

1. Por que uma lista Python de números é lenta: a resposta é o **layout de memória**, não só a velocidade do interpretador.
2. O que "vetorizado" realmente significa por baixo, e por que não é apenas "um loop que outra pessoa escreveu".
3. Como medir a diferença você mesmo, em vez de acreditar nela.
4. Funções universais (ufuncs), a interface que toda operação do NumPy compartilha.
5. Quando o NumPy *não* é a resposta certa.

**Tempo estimado:** 60 minutos

> O Módulo 01 prometeu que `arr.sum()` é muito mais rápido que `sum(my_list)` e que `sys.getsizeof` engana sobre contêineres. Este notebook resolve as duas questões.

<!-- en:cd6e9ef66f -->
## 1. O problema de uma lista de números

Uma `list` de Python é um ótimo contêiner de uso geral. Ela guarda qualquer coisa, em qualquer mistura, e cresce sob demanda. Essa flexibilidade tem um custo, e o custo é estrutural.

![Por que o NumPy é mais rápido: layout de memória](../../../assets/images/numpy_memory_layout.png)

Uma lista não guarda os seus números. Ela guarda **ponteiros** para objetos Python completos que estão em outro lugar da memória. Por isso, somar duas listas elemento a elemento significa, para cada elemento:

1. seguir um ponteiro até o objeto A,
2. seguir um ponteiro até o objeto B,
3. verificar os dois tipos durante a execução,
4. despachar para o `__add__` correto,
5. alocar um *novo* objeto Python para o resultado,
6. guardar um ponteiro para ele.

Um ndarray guarda os números brutos, lado a lado, todos do mesmo tipo. Somar dois arrays executa um loop enxuto em C compilado, sem nenhum dos passos de 1 a 4.

<!-- en:59e334a088 -->
### Vendo a sobrecarga

<!-- en:fa471dfe82 -->
> [!NOTE]
> Esses bytes extras não são desperdício: são a contagem de referências, o ponteiro de tipo e os dígitos de precisão arbitrária que tornam os inteiros de Python ilimitados (Módulo 01, notebook 02). Você está pagando por recursos reais. A questão é que, em trabalho numérico, você raramente precisa deles, e o NumPy permite dispensá-los.

<!-- en:78c599fd1b -->
### `sys.getsizeof` engana sobre contêineres

O Módulo 01 já avisou: `getsizeof` é **raso**. Em uma lista, ele conta o array de ponteiros e nada daquilo para o que eles apontam.

<!-- en:1fcd06ae18 -->
`ndarray.nbytes` é exato e não precisa de ressalvas, porque não há mais nada a contar: os dados *são* o buffer.

<!-- en:7d5981cab3 -->
## 2. Como medir a diferença de velocidade

Vamos resolver a afirmação do Módulo 01 com uma medição, e não com outra afirmação. Quatro formas de somar um milhão de números:

<!-- en:6b6411cdc1 -->
Dois resultados merecem uma pausa.

**`sum()` em um ndarray é a opção *mais lenta*.** O `sum` embutido de Python percorre o array elemento a elemento, e o NumPy precisa empacotar cada valor bruto de volta em um objeto Python para entregá-lo. Você tem toda a sobrecarga da lista *mais* o empacotamento. Usar uma função embutida de Python em um array é um erro comum e caro.

**`np.sum` é muito mais rápido** porque o loop inteiro acontece em código compilado, sobre um buffer contíguo: sem empacotamento, sem verificação de tipos, sem perseguir ponteiros.

<!-- en:43c64f49e0 -->
## 3. O que "vetorizado" realmente significa

A palavra costuma ser usada de forma imprecisa. Com precisão, ela significa: **o loop sai do interpretador Python e vai para código compilado.** Daí decorrem três coisas.

<!-- en:836d68d369 -->
> [!TIP]
> Esse último resultado é o motivo de `np.ascontiguousarray` existir, e de o notebook 08 do Módulo 02 dedicar tempo aos strides. Quando um array é fatiado, transposto ou remodelado, ele pode deixar de ser contíguo, e o custo é real.

<!-- en:8a169bb38c -->
## 4. Funções universais (ufuncs)

Uma **ufunc** é uma função que opera elemento a elemento sobre arrays, com broadcasting, promoção de tipos e um buffer de saída opcional, tudo resolvido para você. Quase tudo no NumPy é uma ufunc.

<!-- en:c35217cb3d -->
### Por que `math.sqrt` não serve como substituto

O módulo `math` trabalha com um escalar por vez. Aplicá-lo a um array significa fazer um loop em Python, exatamente o que estamos tentando evitar.

<!-- en:9c8ed2dc09 -->
### O parâmetro `out=`: como evitar uma alocação

Toda ufunc pode gravar em um array existente em vez de alocar um novo. Com arrays grandes em um loop, isso importa.

<!-- en:6ebdee0e99 -->
## 5. Como a vantagem cresce com o tamanho

A diferença não é um multiplicador fixo: ela aumenta com o tamanho do array, porque a sobrecarga do interpretador por elemento é constante, enquanto o NumPy dilui seu custo fixo em mais dados.

<!-- en:d394b21a15 -->
> [!NOTE]
> Em tamanhos muito pequenos, a vantagem diminui e pode até se inverter. O NumPy paga um custo fixo por *chamada* (interpretação de argumentos, resolução de dtype, alocação), que uma operação simples de Python evita. Para poucos valores, Python puro pode ganhar. Esse é um efeito real, não um artefato de arredondamento, e é por isso que você não deve "numpyficar" uma lista de três elementos.

<!-- en:c7ac86b364 -->
## 6. Quando o NumPy não é a resposta

Ser honesto sobre os limites faz parte de conhecer a ferramenta:

| Situação | Por que o NumPy tem dificuldade | Use |
| :--- | :--- | :--- |
| Colunas heterogêneas (strings + números + datas) | Um dtype por array | **pandas** (Módulo 03) |
| Registros realmente de tamanho variável | Só arrays retangulares | listas de dicts, ou Arrow |
| Dados maiores que a RAM | Tudo fica na memória | **Dask / Spark** (Módulo 20) |
| Processamento pesado de texto | Nenhuma vetorização de strings digna do nome | `str` de Python, ou `.str` do pandas |
| Um algoritmo que não pode ser escrito sem loop | A sobrecarga do loop Python volta | **Numba**, **Cython** ou C |
| Menos de ~100 elementos | O custo por chamada domina | Python puro |

<!-- en:a281810c58 -->
## 7. Exercícios

### Exercício 1.1: meça você mesmo

Compare quatro formas de calcular o produto elemento a elemento de duas sequências de um milhão de floats: um loop `for`, uma list comprehension, `zip` + comprehension e `arr_a * arr_b`. Informe cada tempo e o ganho em relação à mais lenta.

### Exercício 1.2: a pergunta honesta sobre memória

Escreva `true_list_size(lst)`, que devolve o total de bytes que uma lista de números realmente ocupa (o array de ponteiros mais cada objeto referenciado). Compare com o array `int64` equivalente para 10, 1 000 e 100 000 elementos. Faça um gráfico da proporção.

### Exercício 1.3: encontre o ponto de equilíbrio

Para `x * 2 + 1`, encontre o tamanho de array em que o NumPy ultrapassa uma list comprehension. Busque tamanhos de 1 a 10 000. Por que não é 1?

### Exercício 1.4: encontre a desvetorização

Cada um destes trechos volta silenciosamente à velocidade de Python. Explique por que e reescreva:

```python
# (a)
total = sum(arr)

# (b)
result = np.array([np.sqrt(v) for v in arr])

# (c)
biggest = max(arr)

# (d)
doubled = np.array(list(map(lambda x: x * 2, arr)))
```

### Exercício 1.5: `out=` em um loop

Simule um cálculo iterativo: 100 rodadas de `y = sqrt(x) + log1p(x)` sobre um array de 1 milhão de elementos. Escreva uma vez alocando novos arrays a cada rodada e outra reutilizando buffers pré-alocados. Meça as duas.

<!-- en:cf282224e2 -->
## Principais lições

- Uma lista guarda **ponteiros para objetos**; um ndarray guarda **valores brutos, lado a lado, com um único dtype**. Todas as diferenças de desempenho vêm disso.
- **`sys.getsizeof` é raso** e subestima muito uma lista. `ndarray.nbytes` é exato.
- "**Vetorizado**" significa que o loop roda em C compilado, não no interpretador.
- **Nunca aplique uma função embutida de Python a um array**: `sum(arr)` e `max(arr)` são mais lentos que as versões com lista, porque o NumPy precisa empacotar cada valor.
- Ufuncs aceitam **`out=`** para reutilizar um buffer em vez de alocar.
- A vantagem **cresce com o tamanho do array** e pode se inverter abaixo de ~100 elementos.
- O NumPy precisa de um único dtype e de um formato retangular. Quando os seus dados não são assim, a resposta é pandas, não um array mais esperto.

---

**Próximo:** [02 · Fundamentos do ndarray](02_ndarray_fundamentals.ipynb): criação, shape, indexação e a regra de views e cópias, que inverte tudo o que você aprendeu sobre fatiamento no Módulo 01.

<!-- comments -->
EN: # The honest figure: the pointer array PLUS every object it references.
PT: # O número honesto: o array de ponteiros MAIS cada objeto que ele referencia.
EN: # Colour by kind of operation, not by rank: red for the pure-Python paths,
PT: # Cor por tipo de operação, não por posição: vermelho para os caminhos em Python puro,
EN: # teal for the vectorised one.
PT: # verde-azulado para o vetorizado.
EN: # --- 1. The operation applies to the whole array at once --------------------
PT: # --- 1. A operação se aplica ao array inteiro de uma vez --------------------
EN: # --- 2. There is no per-element type check ---------------------------------
PT: # --- 2. Não há verificação de tipo por elemento ----------------------------
EN: # The dtype is fixed for the whole array, so the C loop checks it ONCE.
PT: # O dtype é fixo para todo o array, então o loop em C o verifica UMA vez.
EN: # --- 3. Contiguity lets the CPU work on several values per instruction -----
PT: # --- 3. A contiguidade permite à CPU tratar vários valores por instrução ----
EN: # Modern CPUs have SIMD registers (AVX2 holds 4 float64s; AVX-512 holds 8).
PT: # CPUs modernas têm registradores SIMD (AVX2 guarda 4 float64; AVX-512 guarda 8).
EN: # They only help when the data sits back to back in memory.
PT: # Eles só ajudam quando os dados ficam lado a lado na memória.
EN: # Every second element: same number of values, but strided access.
PT: # Um elemento a cada dois: mesma quantidade de valores, mas acesso com passo.
EN: # copy so we compare like with like
PT: # copia para comparar coisas equivalentes
EN: # a view — NOT contiguous
PT: # uma view, NÃO contígua
EN: # writes in place, allocates nothing
PT: # grava no lugar, não aloca nada
EN: # --- Left: absolute times --------------------------------------------------
PT: # --- Esquerda: tempos absolutos ---------------------------------------------
EN: # --- Right: the speed-up ratio ---------------------------------------------
PT: # --- Direita: a proporção de ganho ------------------------------------------
EN: # NumPy CAN hold strings and mixed data — but notice what happens to the dtype.
PT: # O NumPy CONSEGUE guardar strings e dados misturados, mas repare no que acontece com o dtype.
