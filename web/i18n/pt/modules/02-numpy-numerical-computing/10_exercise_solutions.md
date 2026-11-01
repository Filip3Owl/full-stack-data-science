<!-- en:fa676ab56f -->
# 10 · Soluções dos Exercícios

**Módulo 02 — NumPy e Computação Numérica**

---

> [!WARNING]
> **Tente cada exercício antes de ler a solução.** Ler uma solução pronta produz uma sensação de entendimento que desaparece no momento em que você encara uma célula em branco. Dez minutos de esforço ensinam mais do que sessenta de leitura.

Quarenta exercícios, um por seção dos notebooks de 01 a 08. Quando mais de uma resposta é defensável, as alternativas são mostradas com seus prós e contras: no trabalho, a pergunta interessante raramente é "funciona?", e sim "qual versão eu gostaria de manter, e qual continua correta com 100× mais dados?"

<!-- en:fb91fd2442 -->
---
# Notebook 01 — Por que o NumPy Existe

<!-- en:15a368f74e -->
## Exercício 1.1: meça você mesmo

Quatro formas de multiplicar um milhão de pares de floats. A única que sai do interpretador é a última.

<!-- en:1c7c46e2df -->
As três versões com listas ficam próximas umas das outras porque todas fazem a mesma coisa: um milhão de iterações no interpretador, cada uma desempacotando dois objetos `float`, multiplicando e empacotando o resultado. `zip` é a mais rápida delas só porque evita as chamadas a `__getitem__`.

`arr_a * arr_b` é outro tipo de operação. Uma chamada entra em C, e o loop sobre um milhão de valores `float64` contíguos roda lá, sem verificação de tipo por elemento, sem empacotamento e com instruções SIMD tratando vários valores por ciclo de clock.

<!-- en:a0cd4a1e3e -->
## Exercício 1.2: a pergunta honesta sobre memória

`sys.getsizeof` em uma lista é **raso**: conta o array de ponteiros e o cabeçalho do próprio objeto lista, e para aí. Os objetos apontados não entram, e em uma lista de números eles são a maior parte do custo.

<!-- en:a3ee16c689 -->
A proporção **sobe** com `n` e depois se estabiliza perto de 4,5×. Com `n = 10`, o cabeçalho de 56 bytes da lista é uma parcela visível do total; com `n = 100 000`, o custo é inteiramente de 8 bytes de ponteiro mais 28 bytes de objeto `int` por elemento, contra 8 bytes no array.

A lição não é o número exato, e sim que **`sys.getsizeof` subestima uma lista de números em cerca de 4×** e dirá tranquilamente que uma lista é menor que o array com os mesmos dados.

<!-- en:d5a93b1de2 -->
## Exercício 1.3: encontre o ponto de equilíbrio

O NumPy não é gratuito. Toda chamada interpreta argumentos, resolve dtypes, faz broadcast de shapes e aloca a saída antes de tocar em um único elemento. Esse custo fixo é de alguns microssegundos, e uma list comprehension ganha com folga quando há poucos elementos a processar.

<!-- en:15a6d1e979 -->
**Por que não é 1?** Abaixo do ponto de equilíbrio, todo o tempo é a sobrecarga por chamada do NumPy: o trecho plano à esquerda da linha verde-azulada, que não muda quando `n` cresce. A list comprehension quase não tem custo fixo, mas paga por elemento, então começa bem mais baixa e sobe de forma linear. O ponto em que as duas se encontram é o ponto de equilíbrio.

A consequência prática: **chamar o NumPy dentro de um loop Python sobre arrays pequenos é o pior dos dois mundos.** Você paga a sobrecarga milhares de vezes e nunca chega ao tamanho em que o loop interno vetorizado a compensa.

<!-- en:23419ae447 -->
## Exercício 1.4: encontre a desvetorização

Os quatro voltam a passar pelo interpretador. Os três primeiros são piores que os equivalentes com listas, porque o NumPy precisa construir um objeto Python para cada elemento que entrega.

<!-- en:0367b3705f -->
| | Por que é lento | Correção |
| :-- | :--- | :--- |
| **(a)** | O `sum` de Python percorre o array, empacotando cada `float64` em um `float` de Python, e depois soma com o despacho do interpretador. É *mais lento que somar uma lista*, porque uma lista já guarda objetos Python. | `arr.sum()` |
| **(b)** | `np.sqrt` em um **escalar** devolve um `np.float64`: um despacho completo de ufunc por elemento, depois a construção de uma lista, depois a conversão de volta em array. | `np.sqrt(arr)` |
| **(c)** | O mesmo problema de (a): `max` é um loop no interpretador com uma comparação por elemento. | `arr.max()` |
| **(d)** | `map` com um `lambda` de Python é um loop em Python disfarçado de programação funcional. | `arr * 2` |

O sinal é sempre o mesmo: **um `for`, uma comprehension, um `map` ou uma agregação embutida de Python com um array dentro.** Se o array aparece como *iterável* em vez de *operando*, você saiu do caminho rápido.

<!-- en:c2ef2724e9 -->
## Exercício 1.5: `out=` em um loop

Um cálculo iterativo é onde o custo de alocação deixa de ser ruído. Cada rodada da versão ingênua pede ao alocador três novos blocos de 8 MB e depois os joga fora.

<!-- en:6e1c4aefc0 -->
O ganho costuma ser modesto, entre 1,2× e 1,6×, e esse é o resultado honesto que vale guardar. `sqrt` e `log1p` são funções transcendentais; a aritmética realmente domina. `out=` economiza a alocação e o tráfego de memória, não a matemática.

`out=` passa a ser decisivo quando a operação é **barata em relação ao tráfego de memória** (`a + b`, `a * 2`), ou quando o array é grande o bastante para a alocação começar a pedir novas páginas ao sistema operacional. Recorra a ele quando a análise de desempenho apontar para a alocação, não por princípio.

<!-- en:3582cab854 -->
---
# Notebook 02 — Fundamentos do ndarray

<!-- en:3cd0980bf9 -->
## Exercício 2.1: preveja os shapes

| Expressão | Shape | Raciocínio |
| :--- | :-- | :--- |
| `a[0]` | `(3, 4)` | Um índice inteiro **remove** aquele eixo. |
| `a[:, 1]` | `(2, 4)` | O inteiro no eixo 1 o remove; os eixos 0 e 2 sobrevivem. |
| `a[..., 2]` | `(2, 3)` | `...` preenche "todos os eixos restantes"; o último é removido pelo índice. |
| `a[0, :, ::2]` | `(3, 2)` | Eixo 0 removido, eixo 1 mantido inteiro, eixo 2 de dois em dois elementos. |
| `a.T` | `(4, 3, 2)` | `.T` inverte a ordem dos eixos: em 3-D, não é "trocar os dois últimos". |
| `a.reshape(4, -1)` | `(4, 6)` | `-1` significa "o que fizer os 24 elementos caberem". |

<!-- en:d4a14c75ec -->
A regra que resolve quase todos estes casos: **um índice inteiro apaga um eixo; uma fatia o mantém.** `a[0]` tem uma dimensão a menos que `a`; `a[0:1]` tem o mesmo número. Essa única distinção está por trás da maior parte da confusão do tipo "por que meu shape é `(n,)` se eu queria `(n, 1)`".

<!-- en:55d9bc989b -->
## Exercício 2.2: view ou cópia?

A regra: **o fatiamento básico dá uma view; a indexação avançada e as máscaras booleanas dão uma cópia.** Uma view é possível sempre que os elementos selecionados puderem ser descritos por um deslocamento inicial e um passo fixo por eixo, o que uma lista arbitrária de índices não permite.

<!-- en:acf59cadb8 -->
Duas armadilhas escondidas nessa tabela:

- **`reshape` só é uma view quando pode ser.** Em dados contíguos, ele apenas reinterpreta o shape. Em um array não contíguo, precisa copiar, então você não pode contar com nenhum dos dois comportamentos. Verifique, não suponha.
- **`ravel` e `flatten`.** `ravel` devolve uma view quando pode; `flatten` sempre copia. Se você gravar no resultado de `ravel`, pode estar gravando no original.

<!-- en:a51cb38c54 -->
## Exercício 2.3: conserte o bug de modificação

`col = X[:, j]` é uma **view**. `col *= factor` é uma multiplicação no lugar, então grava direto no array de quem chama. A função não tem nenhuma `copy`, então modifica a entrada sem aviso e *também* a devolve: a pior combinação, porque quem chama tem todos os motivos para achar que recebeu algo novo.

<!-- en:2f3f8ea3b5 -->
### A asserção que teria pegado o erro

Em um teste, o padrão é: guarde uma cópia da entrada, chame a função, compare.

<!-- en:ca1023e937 -->
> Se uma função é documentada como pura, `np.shares_memory(out, inp)` é um teste de regressão de uma linha para ela. Esta classe de bug é o equivalente em NumPy da armadilha do argumento padrão mutável do Módulo 01, e é muito mais difícil de notar, porque a corrupção aparece em uma célula que você executou vinte minutos atrás.

<!-- en:1414de5827 -->
## Exercício 2.4: construa um tabuleiro de xadrez

<!-- en:b4610513b5 -->
**Qual eu manteria?** `(np.arange(8)[:, None] + np.arange(8)) % 2`: ela declara a regra ("escuro quando linha + coluna é ímpar") em vez da mecânica, generaliza para qualquer tamanho com uma edição e nunca aloca as duas grades de coordenadas completas que `np.indices` constrói. A versão com fatiamento também serve, mas você precisa ler duas vezes para se convencer de que as duas linhas cobrem cada casa exatamente uma vez.

<!-- en:f8544ac0c1 -->
## Exercício 2.5: remodele para o scikit-learn

<!-- en:23c0828b4a -->
Os três são a mesma operação escrita de três formas, e os três são **views**: adicionar um eixo de tamanho 1 altera só os metadados de shape e strides, nunca o buffer.

Prefira **`y.reshape(-1, 1)`**: funciona sem saber `len(y)`, é a forma que a própria mensagem de erro do scikit-learn sugere, e `-1` deixa explícita a intenção ("uma coluna, com quantas linhas forem necessárias").

`y[:, None]` é o mesmo que `y[:, np.newaxis]`: `np.newaxis` *é* `None`. É mais conciso e muito comum em código de bibliotecas; também é a escolha menos legível em um notebook que outra pessoa vai herdar.

<!-- en:408080a501 -->
---
# Notebook 03 — Dtypes e Precisão Numérica

<!-- en:c337662c0c -->
## Exercício 3.1: preveja o dtype

| Expressão | Resultado | Regra |
| :--- | :-- | :--- |
| `int8 + int16` | `int16` | Os dois são arrays: promove para o menor tipo que comporta ambos. |
| `int32 / 2` | `float64` | **A divisão real sempre produz um float**, seja qual for a entrada. |
| `float32 + 1.0` | `float32` | Um `1.0` simples de Python é um escalar *fraco*: adota o dtype do array. |
| `float32 + np.float64(1.0)` | `float64` | `np.float64(...)` é um valor **tipado**, então vale a promoção normal. |
| `bool + int64` | `int64` | `bool` é promovido para o tipo inteiro. |
| `np.arange(3) ** 2` | `int64` | `arange` usa o inteiro padrão da plataforma; `**` o mantém. |

<!-- en:586dad590c -->
As linhas 3 e 4 são o par que vale guardar na memória, e são as que o NumPy 2.0 mudou (NEP 50). **Um escalar Python nunca amplia um array; um escalar NumPy, sim.** Em um pipeline `float32` construído para reduzir a memória pela metade, um único `np.float64(...)` perdido, ou um resultado de `np.mean()` usado de volta, a dobra novamente sem aviso.

<!-- en:9032b9bce5 -->
## Exercício 3.2: provoque e depois evite um estouro

Este tem uma surpresa. Tente primeiro o óbvio:

<!-- en:1214c462ee -->
**Não estourou**, e a resposta nem é `int16`. `np.sum` amplia seu acumulador para o inteiro padrão da plataforma em qualquer dtype de entrada mais estreito, justamente porque somar em um `int16` seria uma armadilha.

Então, onde o `int16` realmente causa problema? Sempre que o acumulador *não* é ampliado para você: na aritmética elemento a elemento e em qualquer redução em que você fixe o dtype.

<!-- en:c981c9e29d -->
O `values * 400` elemento a elemento é o perigoso: `1000 * 400 = 400 000`, muito além dos 32 767 do `int16`, então dá a volta para `6784`. **Sem aviso, sem exceção, só um número errado.** O NumPy 2 gera `RuntimeWarning` para estouro de escalares, mas fica em silêncio para arrays.

### Duas formas de acertar

<!-- en:7525591647 -->
> **A regra para dtypes inteiros estreitos:** o armazenamento estreito é uma decisão de *armazenamento*. No momento em que fizer contas, pergunte qual é o maior intermediário possível, não o maior valor que está na coluna hoje.

<!-- en:79a7604e3d -->
## Exercício 3.3: quando o float32 não basta?

Este exercício também tem uma surpresa. Some um milhão de cópias `float32` de 0.1, e o NumPy acerta quase exatamente, porque `np.sum` faz **soma em pares**, não uma acumulação ingênua da esquerda para a direita.

<!-- en:e729060ecc -->
Mesmos números, mesmo dtype, **cinco ordens de grandeza** de diferença em exatidão: a única variável é a ordem das somas.

O mecanismo: quando o total acumulado chega a ~2 000, a distância entre valores `float32` vizinhos passa dos 0.1 somados, então cada soma arredonda fora parte do incremento. A soma em pares mantém as somas parciais pequenas e de magnitude parecida, então isso nunca acontece.

<!-- en:a0a5c9d825 -->
**O que levar daqui.** O `float32` reduz a memória pela metade e, em uma GPU, praticamente dobra a vazão, e é por isso que deep learning vive nele. Ele serve para guardar atributos e pesos de modelos. *Não* serve para acumulações longas: somas, médias, variâncias, produtos escalares de vetores longos e qualquer coisa iterativa.

A regra prática é **guardar estreito e acumular largo**: `x32.sum(dtype=np.float64)`. Você mantém a economia de memória e perde o erro.

<!-- en:97ac2e5084 -->
## Exercício 3.4: construa um relatório de memória

<!-- en:19176e4930 -->
`price` se recusa a reduzir, e isso é o relatório funcionando corretamente: valores monetários de duas casas decimais não têm representação exata em `float32`, então o teste de ida e volta falha. `discount` reduz até `float16` porque `0.0`, `0.25` e `0.5` são frações binárias exatas.

A proteção de `_id` vem de uma cicatriz real. Reduzir uma coluna de ID para `int32` porque o máximo de hoje é 2 milhões funciona até a tabela passar de 2,1 bilhões de linhas. Aí o valor dá a volta, e dois clientes diferentes viram o mesmo cliente. **Identificadores são rótulos, não quantidades.** O ideal é nem os guardar como números.

<!-- en:f4949987dd -->
## Exercício 3.5: cancelamento catastrófico na prática

<!-- en:2121fc86fb -->
### O que realmente aconteceu

O estrago foi feito **antes de qualquer conta**. `1 - 1e-17` não pode ser representado em `float64`: perto de 1.0, a distância entre valores representáveis consecutivos é `2.22e-16`, e pedimos um deslocamento 22× menor que isso. O literal é arredondado para exatamente `1.0`, então, quando a fórmula vê `p`, a quantidade que carregava toda a informação, o `1e-17`, simplesmente sumiu. Depois, `1 - p` é `0.0`, e o erro padrão é informado como zero: **um intervalo de confiança de largura zero em torno de uma probabilidade de 1.**

A solução não é mais precisão, e sim uma **mudança de variável**. Carregue diretamente a quantidade pequena `q = 1 - p` e nunca materialize `p`. `q * (1 - q)` é avaliado perto de `1e-17`, onde o float64 tem resolução de sobra.

<!-- en:83ec84db48 -->
> **O sinal a observar:** subtrair dois números quase iguais ou somar um número pequeno a um grande. Os dois destroem os dígitos iniciais da quantidade pequena. Sempre que uma fórmula tiver `1 - p`, `a - b` ou `E[x²] - E[x]²`, pergunte se existe uma forma algebricamente idêntica que mantenha pequena a quantidade pequena. Geralmente existe, e geralmente já está em `numpy` ou `scipy` (`log1p`, `expm1`, `logaddexp`, `np.var`).

<!-- en:ccc0bc9f7a -->
---
# Notebook 04 — Broadcasting

<!-- en:245812229c -->
## Exercício 4.1: compatível ou não?

Trabalhe da direita para a esquerda. Complete o shape mais curto com `1`s **à esquerda**; depois, um par de dimensões é compatível quando elas são iguais ou uma delas é 1.

| Par | Completado | Resultado | Por quê |
| :--- | :--- | :-- | :--- |
| `(5, 4)` e `(4,)` | `(1, 4)` | `(5, 4)` | 4=4 ✔, 5 e 1 → estica ✔ |
| `(5, 4)` e `(5,)` | `(1, 5)` | **erro** | 4 e 5, nenhum é 1 |
| `(5, 4)` e `(5, 1)` | — | `(5, 4)` | 4 e 1 → estica ✔, 5=5 ✔ |
| `(3, 1, 4)` e `(2, 4)` | `(1, 2, 4)` | `(3, 2, 4)` | 4=4, 1 e 2 → estica, 3 e 1 → estica |
| `(2, 3, 4)` e `(3, 1)` | `(1, 3, 1)` | `(2, 3, 4)` | 4 e 1 → estica, 3=3, 2 e 1 → estica |
| `(8, 1, 6)` e `(7, 1)` | `(1, 7, 1)` | `(8, 7, 6)` | 6 e 1, 1 e 7, 8 e 1: todos esticam |

<!-- en:4a59f6a11e -->
A segunda linha é a que pega as pessoas, e pega porque a *intenção* era quase certamente legítima. `(5, 4)` com `(5,)` parece "um valor por linha", mas o broadcasting alinha pela direita, então tenta comparar o 5 com as 4 colunas. Diga o que você quer com `right.reshape(-1, 1)`.

<!-- en:abf6087b3d -->
## Exercício 4.2: conserte o bug silencioso

<!-- en:3663ea5036 -->
`(3,)` é completado para `(1, 3)`; contra `(3, 1)`, as duas dimensões esticam, e você obtém a **diferença externa** completa `(3, 3)`: cada valor real menos cada valor previsto. Só a diagonal tem significado.

O motivo de isso ser perigoso, e não apenas errado: não gera erro. Um `.mean()` seguinte devolve um número, o notebook roda sem problemas, e você informa um MSE calculado sobre nove resíduos em vez de três.

<!-- en:f5980ac0b1 -->
> **O hábito:** depois de qualquer subtração de dois arrays vindos de lugares diferentes, verifique o shape da saída com uma asserção. `assert out.shape == expected` é uma linha e transforma toda uma classe de bugs numéricos silenciosos em um traceback claro na linha que os causou.

<!-- en:88302ab824 -->
## Exercício 4.3: normalize linhas em vez de colunas

<!-- en:e664b53473 -->
**Por que falha.** `X.sum(axis=1)` colapsa o eixo 1, deixando o shape `(100,)`. O broadcasting o completa para `(1, 100)` e tenta alinhá-lo com `(100, 5)`: 100 contra 5, nenhum é 1, erro.

**Por que `keepdims=True` funciona.** Ele mantém o eixo colapsado com tamanho 1, então o shape é `(100, 1)`. Contra `(100, 5)`: 1 estica para 5, 100 coincide com 100. Cada linha é dividida pelo próprio total, que é o que "normalizar linhas" significa.

### A versão que não gera erro

O `ValueError` acima é um presente. Em uma matriz **quadrada**, o mesmo bug produz um array perfeitamente válido de números completamente errados.

<!-- en:1b18f73076 -->
> **A regra:** se você estiver dividindo um array por uma de suas próprias reduções, quase sempre vai querer `keepdims=True`. Isso documenta qual eixo você quis dizer e transforma a falha silenciosa da matriz quadrada em uma falha clara.

<!-- en:8760323457 -->
## Exercício 4.4: distância até os centroides

Uma iteração de k-means. Os índices `None` são o que faz isso funcionar: alinhe os pontos em um novo eixo 1 e os centroides em um novo eixo 0, e a subtração produz todos os pares de ponto e centroide.

<!-- en:01c1c8f552 -->
### O custo de que ninguém fala

O tensor `diff` é `n × k × d`. Aqui são 24 000 floats, quase nada. Em escala realista, ele é o problema inteiro:

<!-- en:4bf83f06fd -->
A solução é a mesma expansão algébrica do Exercício 4.5:

$$\|x - c\|^2 = \|x\|^2 - 2\,x \cdot c + \|c\|^2$$

Cada termo é uma norma de linha ou um produto de matrizes, então o intermediário `(n, k, d)` nunca existe.

<!-- en:3141047be0 -->
A forma expandida é a que `euclidean_distances` do scikit-learn usa, e vale conhecer a contrapartida: ela é **menos exata numericamente**. Quando `x_sq` e `c_sq` são grandes e a distância real é pequena, você está subtraindo dois números grandes quase iguais: é o cancelamento do Exercício 3.5 de novo. Para clustering, isso é irrelevante; se precisar de distâncias pequenas exatas, mantenha a versão com broadcast e processe as linhas em blocos.

<!-- en:8f1354c0de -->
## Exercício 4.5: evite o intermediário

Expanda o quadrado, e a soma dupla se separa:

$$\sum_i \sum_j (a_i - b_j)^2
 = \sum_i \sum_j \left( a_i^2 - 2 a_i b_j + b_j^2 \right)
 = m \sum_i a_i^2 \;-\; 2 \left(\sum_i a_i\right)\!\left(\sum_j b_j\right)
   \;+\; n \sum_j b_j^2$$

em que `n = len(a)` e `m = len(b)`. Três reduções e um produto, sem nada par a par.

<!-- en:cccf388858 -->
20 GB de RAM e minutos de processamento, substituídos por três produtos escalares e um milissegundo. Essa é a habilidade de NumPy de maior impacto que existe: **quando uma expressão constrói um intermediário `(n, m)` ou `(n, m, d)`, expanda a álgebra primeiro.** Distâncias ao quadrado, matrizes de Gram, variâncias e covariâncias se fatoram assim.

<!-- en:5deb85cac9 -->
---
# Notebook 05 — Agregações e Máscaras Booleanas

<!-- en:046151427f -->
## Exercício 5.1: treino de eixos

`axis=` indica a dimensão que **desaparece**. Para `a` de shape `(2, 3, 4)`:

| Expressão | Shape | Raciocínio |
| :--- | :-- | :--- |
| `a.sum(axis=0)` | `(3, 4)` | O eixo 0 (tamanho 2) colapsa. |
| `a.sum(axis=1)` | `(2, 4)` | O eixo 1 (tamanho 3) colapsa. |
| `a.sum(axis=2)` | `(2, 3)` | O eixo 2 (tamanho 4) colapsa. |
| `a.sum(axis=(0, 2))` | `(3,)` | Os dois eixos indicados colapsam; só o eixo 1 sobrevive. |
| `a.mean(axis=1, keepdims=True)` | `(2, 1, 4)` | O eixo 1 fica, com tamanho 1. |
| `a.max(axis=-1)` | `(2, 3)` | `-1` é o último eixo, ou seja, o eixo 2. |

<!-- en:6e46340602 -->
Uma verificação que torna a regra concreta: **o shape do resultado é o shape da entrada com as entradas indicadas apagadas.** `(2, 3, 4)` menos a entrada 0 é `(3, 4)`; menos as entradas 0 e 2 é `(3,)`. Com `keepdims=True`, elas viram `1` em vez de sumir.

<!-- en:9acefa955a -->
## Exercício 5.2: a armadilha do keepdims

<!-- en:eec6f73a58 -->
### O que exatamente o primeiro calculou

`X.sum(axis=1)` tem shape `(5,)`. O broadcasting o completa para `(1, 5)` e o estica pelas linhas, então ele é tratado como **um valor por coluna**, quando guarda um valor por linha.

O resultado é `out[i, j] = X[i, j] / rowsum[j]`: o elemento `(i, j)` dividido pelo total da linha `j`, uma linha que não tem nada a ver com o elemento `(i, j)`, a menos que `i == j`.

<!-- en:70763b0620 -->
Ele só roda porque a matriz é quadrada. Faça-a `(5, 4)`, e o mesmo código gera um erro. É por isso que esse bug sobrevive muito mais em código que por acaso lida com matrizes quadradas (matrizes de correlação, de adjacência, de confusão, pontuações de atenção).

<!-- en:cfea45f676 -->
## Exercício 5.3: prática de máscaras

<!-- en:3148ef4ab5 -->
### Duas observações sobre a substituição

**Winsorizar é melhor que substituir pela média.** Colapsar outliers na média inventa um pico de valores idênticos no centro da distribuição, o que distorce a variância e todos os quantis. Limitar por um percentil mantém a ordenação intacta:

<!-- en:8766c3ef1c -->
**E um aviso sobre a própria regra de ±2σ.** O desvio padrão é calculado *a partir dos dados que incluem os outliers*, então alguns valores extremos inflam `σ` e se escondem atrás do limite mais largo. Alternativas robustas, como mediana ± k·MAD ou uma regra de IQR, não têm esse problema.

<!-- en:363a45424a -->
## Exercício 5.4: group-by do zero

O truque é `np.unique(..., return_inverse=True)`. Ele dá as chaves únicas ordenadas *e* um código inteiro por linha, e, quando toda chave é um inteiro em `0..n_groups-1`, `np.bincount` faz a agregação em C.

<!-- en:7e7c4eb7bd -->
### Por que `np.maximum.at` e não um loop

`maxima[codes] = np.maximum(maxima[codes], values)` parece equivalente e está **errado**. A atribuição com indexação avançada usa buffer: o NumPy lê todos os valores antigos, calcula e grava de volta, então, quando um código aparece duas vezes, a segunda gravação sobrescreve a primeira em vez de se combinar com ela. `ufunc.at` é a versão sem buffer, que existe justamente para isso.

<!-- en:8f4bc65469 -->
`ufunc.at` é correto, mas lento: não consegue vetorizar. Especificamente para o máximo, a alternativa rápida é ordenar por código e usar `np.maximum.reduceat`; use-a quando isso aparecer em uma análise de desempenho, não antes.

<!-- en:05d305d585 -->
## Exercício 5.5: top-k por linha

`argsort` em `(1000, 50)` é `O(n · m log m)` e devolve 50 índices por usuário quando você queria 5. `argpartition` é `O(n · m)`: faz só o trabalho necessário para colocar os `k` maiores nas primeiras `k` posições, sem ordem específica. Ordenar apenas esses cinco não custa nada.

<!-- en:76fe2b9779 -->
### Onde a diferença realmente importa

Com 50 itens, a economia é pequena. Um sistema de recomendação trabalha com catálogos de centenas de milhares de itens, e é aí que a diferença de complexidade pesa:

<!-- en:42a5eb45f5 -->
> **Uma ressalva para produção.** `argpartition` não garante nada sobre empates, então dois itens com pontuações idênticas podem voltar em qualquer ordem, e a ordem pode mudar entre versões do NumPy. Se o seu ranking precisa ser determinístico, desempate explicitamente com uma chave secundária estável (o ID do item) usando `np.lexsort`.

<!-- en:bbf1a63ead -->
---
# Notebook 06 — Álgebra Linear

<!-- en:28b79e56f1 -->
## Exercício 6.1: shapes

`@` tem uma regra para 2-D (as dimensões internas precisam ser iguais), mais dois casos especiais para operandos 1-D e uma regra de lote para dimensões maiores.

| Expressão | Resultado | Regra |
| :--- | :-- | :--- |
| `(3, 4) @ (4, 5)` | `(3, 5)` | Os 4 internos coincidem; as dimensões externas sobrevivem. |
| `(3, 4) @ (3, 4)` | **erro** | 4 interno contra 3. |
| `(4,) @ (4,)` | `()` | Os dois são 1-D → produto interno, um escalar. |
| `(3, 4) @ (4,)` | `(3,)` | 1-D à direita é uma coluna; o eixo é removido depois. |
| `(10, 3, 4) @ (4, 2)` | `(10, 3, 2)` | Os eixos iniciais são dimensões de lote e fazem broadcast. |
| `(3,) @ (3, 4)` | `(4,)` | 1-D à esquerda é uma linha; o eixo é removido depois. |

<!-- en:83a42d6e25 -->
Os casos 1-D pedem cuidado. O NumPy promove um operando 1-D a linha ou coluna, conforme o lado em que está, faz a multiplicação e depois **remove o eixo que adicionou**. É conveniente, e é o motivo de `X @ beta` devolver `(n,)` em vez de `(n, 1)`, que depois faz broadcast com um alvo `(n, 1)` e cai no bug do Exercício 4.2.

<!-- en:2971e9f9ae -->
## Exercício 6.2: matriz de similaridade de cosseno

A similaridade de cosseno é o produto escalar de vetores normalizados. Normalize as linhas uma vez, e um único produto de matrizes gera todos os milhões de pares.

<!-- en:d58d9a60b3 -->
Ela compete com o scikit-learn porque é o *mesmo* algoritmo chamando o *mesmo* BLAS. `E @ E.T` em `(1000, 128)` são 128 milhões de multiplicações e somas, e um `dgemm` otimizado executa isso em poucos milissegundos usando todos os núcleos.

**A memória a observar é a da saída, não a da entrada.** `E` tem 1 MB; a matriz de similaridade tem 8 MB. Com 100 000 documentos, `E` tem 100 MB e a matriz tem **80 GB**: você precisa processar as linhas em blocos ou usar um índice aproximado de vizinhos mais próximos, sem nunca formar a matriz.

<!-- en:4ec3e9db4a -->
## Exercício 6.3: implemente mínimos quadrados

<!-- en:1814f24153 -->
`‖β‖` diminuir à medida que `alpha` cresce é o ridge fazendo exatamente aquilo para que serve. Os dois detalhes de implementação que fazem isso coincidir com o scikit-learn, em vez de apenas se *aproximar*:

1. **Centralize em vez de adicionar uma coluna de uns.** Adicionar colocaria o intercepto dentro da penalidade, e todos os coeficientes sairiam levemente errados: um bug que produz números plausíveis e nenhum erro.
2. **`np.linalg.solve(A, b)`, nunca `np.linalg.inv(A) @ b`.** A resposta é a mesma em aritmética exata; em ponto flutuante, `solve` é mais exato e cerca de duas vezes mais rápido, porque formar uma inversa explícita faz estritamente mais trabalho e acumula mais erro.

<!-- en:ddf0094bb0 -->
## Exercício 6.4: número de condição e colinearidade

<!-- en:67ceb38810 -->
### Como ler os números com honestidade

Padronizar melhora o número de condição em ordens de grandeza, mas **não corrige a colinearidade**, e o número de condição restante (ainda grande) é a colinearidade falando. Dois problemas distintos estavam misturados:

| Problema | Causa | A padronização resolve? |
| :--- | :--- | :-- |
| A coluna 2 tem 50 000× a escala das outras | unidades | **Sim** |
| As colunas 0 e 1 têm correlação de 0.9999 | os dados | **Não** |

Mudar a escala é mudar as unidades; isso não cria informação que não existe. Duas colunas quase idênticas significam que os dados realmente não identificam os coeficientes individuais: a *soma* é bem determinada, a divisão não. As opções reais são remover uma coluna, combiná-las ou aceitar a estimativa enviesada, porém estável, do ridge.

Uma leitura prática de `cond(XᵀX)`: abaixo de `1e3`, tudo bem; de `1e3` a `1e8`, trate os coeficientes com desconfiança; acima de `1e10`, os coeficientes individuais são ruído. Você perde cerca de `log10(cond)` dos seus 16 dígitos significativos.

<!-- en:a1af9c3386 -->
## Exercício 6.5: compressão de imagens com SVD

Toda matriz se fatora como `A = U Σ Vᵀ`, com os valores singulares em `Σ` em ordem decrescente. Manter os `r` maiores dá a **melhor aproximação possível** de posto `r` no sentido de mínimos quadrados (Eckart–Young). O armazenamento cai de `m·n` números para `r·(m + n + 1)`.

<!-- en:2be1de36e8 -->
O posto 20 mantém bem mais de 99% da energia em cerca de um quinto do armazenamento. O resíduo é principalmente o ruído que adicionamos: a parte estruturada da imagem realmente tem posto baixo.

**O motivo de isso importar muito além de imagens:** a SVD truncada *é* o PCA. Mesma fatoração, mesmo truncamento, vocabulário diferente: valores singulares viram variância explicada, e as linhas de `Vᵀ` viram componentes principais. Análise semântica latente, fatoração de matrizes em recomendação e redução de ruído são todas a mesma operação. Você vai revê-la no Módulo 15.

<!-- en:0136c28a4c -->
---
# Notebook 07 — Aleatoriedade e Simulação

<!-- en:74c15ec964 -->
## Exercício 7.1: quebre e conserte a reprodutibilidade

A API antiga mantém **um único fluxo global**. Qualquer coisa que sorteia dele move o cursor compartilhado, inclusive código que você não escreveu e não consegue ver.

<!-- en:25fb552265 -->
Nada na seed mudou. Uma função que não tem nada a ver com a análise alterou os resultados sem aviso, e em um pipeline real essa função está a três importações de distância, dentro de uma biblioteca, adicionada no pull request de outra pessoa.

<!-- en:71e0510799 -->
### E, para trabalho em paralelo, `spawn`

Reutilizar uma seed entre processos dá a todos os *mesmos* números, um bug que parece uma variância suspeitamente pequena. `SeedSequence.spawn` deriva fluxos independentes que continuam totalmente reproduzíveis a partir da seed pai.

<!-- en:defe5f4fe4 -->
## Exercício 7.2: escolha a distribuição

| # | Quantidade | Distribuição | Por quê |
| :-- | :--- | :--- | :--- |
| 1 | Chamados de suporte por hora | **Poisson(λ)** | Contagens de eventos independentes em uma janela fixa: o caso de uso que a define. |
| 2 | Tempo entre dois chamados | **Exponencial(λ)** | O tempo de espera entre eventos de Poisson, sem memória por construção. |
| 3 | Se cada um de 500 e-mails é aberto | **Bernoulli(p)** por e-mail, **Binomial(500, p)** para o total | Um resultado binário repetido de forma independente. |
| 4 | Receita anual por cliente | **Lognormal** (ou Gama) | Positiva, assimétrica à direita, multiplicativa: muitos clientes pequenos, alguns enormes. |
| 5 | Erro de medição do sensor | **Normal(0, σ)** | Uma soma de muitas pequenas perturbações independentes: o território do TCL. |

<!-- en:69f70944d1 -->
O par que vale guardar é **1 e 2**: eles descrevem o mesmo processo por dois ângulos. Se as chegadas são Poisson com taxa λ, os intervalos entre elas são Exponenciais com média 1/λ. Inverter isso, modelando *durações* como Normais porque são "mais ou menos simétricas", produz tempos de espera negativos e intervalos de confiança que cobrem valores impossíveis.

E **4**: a receita quase nunca é Normal. Testar uma diferença de médias de receita com um teste t nos valores brutos é testar a estatística errada; trabalhe com logaritmos, use um teste de postos ou faça bootstrap. O Módulo 08 trata disso direito.

<!-- en:2caa5919a2 -->
## Exercício 7.3: o problema do aniversário

<!-- en:78bf98183e -->
**23 pessoas.** A resposta parece errada porque a intuição troca a pergunta sem avisar: "alguém faz aniversário no *meu* dia?", que precisa de 253 pessoas. A pergunta real é sobre **pares**, e 23 pessoas formam `23 × 22 / 2 = 253` deles.

A simulação coincidir com a fórmula exata com margem de ~0.005 é o objetivo do exercício: quando existe uma forma fechada, a simulação é um jeito de *conferi-la*. Quando não existe, a simulação muitas vezes é o único caminho, e o Exercício 7.5 é esse caso.

<!-- en:af4a17b244 -->
## Exercício 7.4: bootstrap de uma diferença

<!-- en:328a22e788 -->
### O que o intervalo permite e o que não permite dizer

**Se o intervalo excluir o zero**, você pode dizer: os dados são incompatíveis com as duas medianas populacionais serem iguais, no nível de 5%. O intervalo é a faixa plausível para a diferença.

**Você não pode dizer:**

- *"Há 95% de probabilidade de a diferença real estar neste intervalo."* A diferença real é um número fixo, não aleatório. Os 95% são uma propriedade do **procedimento**: intervalos construídos assim cobrem a verdade em 95% das vezes.
- *"B causa um gasto maior."* A menos que a atribuição tenha sido aleatória, isso é uma correlação. O Módulo 09 trata de como conquistar a afirmação causal.
- *"A diferença é grande o bastante para agir."* Significância estatística e significância prática são perguntas diferentes. Uma diferença de £1.20 pode ser inequivocamente real e comercialmente irrelevante.

**Se o intervalo contiver o zero**, isso *não* é evidência de que os grupos são iguais: é uma incapacidade de distingui-los, que também é o que uma amostra pequena demais produz. "Ausência de evidência não é evidência de ausência" é toda a razão de o Exercício 7.5 existir.

Mais uma ressalva específica da mediana: a distribuição bootstrap de uma mediana é **discreta e irregular**, porque uma mediana reamostrada só pode assumir valores presentes nos dados. Funciona, mas os intervalos por percentis para medianas são menos exatos que para médias, especialmente em amostras pequenas.

<!-- en:0ea78df797 -->
## Exercício 7.5: tamanho de amostra por simulação

<!-- en:b97851545f -->
### Por que este é o exercício que muda comportamentos

Cerca de **30 000 usuários** para detectar um aumento de um ponto percentual. Equipes rodam com frequência um teste com 2 000 visitantes, não veem diferença significativa e concluem que a mudança não fez nada. Nesse tamanho, o poder fica em torno de 15%: o teste quase certamente não encontraria nada, fosse qual fosse a verdade.

Vale guardar duas propriedades da curva:

- **O n necessário escala com `1 / efeito²`.** Reduzir pela metade o efeito que você quer detectar quadruplica a amostra. Detectar 10% → 10,5% exige cerca de 4× o tráfego de 10% → 11%.
- **O poder precisa ser calculado antes do teste, não depois.** O "poder observado", calculado a partir do resultado obtido, é uma função determinística do p-valor e não acrescenta informação.

A abordagem por simulação se generaliza onde a fórmula não chega: alocação desigual, olhadas sequenciais nos resultados, métricas de razão, atribuição por clusters, redução de variância com CUPED. Quando a fórmula analítica deixa de existir, simule. O Módulo 09 desenvolve tudo isso.

<!-- en:de46dabad2 -->
---
# Notebook 08 — Desempenho e Memória

<!-- en:0783483e87 -->
## Exercício 8.1: preveja os strides

Um stride é **quantos bytes saltar para andar um passo naquele eixo**. Para `float64` (8 bytes) com shape `(2, 3, 4)` na ordem C, o último eixo anda de 8 em 8, o do meio de `4 × 8 = 32` e o primeiro de `3 × 4 × 8 = 96`.

| Expressão | Strides | Por quê |
| :--- | :--- | :--- |
| `a` | `(96, 32, 8)` | O padrão por linhas. |
| `a.T` | `(8, 32, 96)` | Inverter os eixos inverte os strides. **Nenhum dado se move.** |
| `a[::2]` | `(192, 32, 8)` | Passo 2 no eixo 0 dobra esse stride. |
| `a[:, ::2, :]` | `(96, 64, 8)` | Passo 2 no eixo 1 dobra o stride *dele*. |
| `a.reshape(6, 4)` | `(32, 8)` | Dados contíguos reinterpretados; strides recalculados a partir do novo shape. |

<!-- en:cdd4b1dada -->
Todos eles são **views**. Esta é a coisa mais importante sobre strides: transpor um array de 10 GB, invertê-lo ou pegar uma linha a cada duas não custa *nada*. O NumPy edita duas pequenas tuplas de metadados e devolve um novo objeto que aponta para os mesmos bytes.

A conta chega depois. `a.T` é de graça, mas deixa de ser C-contíguo, então qualquer operação que queira memória contígua (uma chamada BLAS, `.ravel()`, `.tobytes()`) precisa fazer uma cópia nesse momento.

<!-- en:08007fc437 -->
## Exercício 8.2: por linhas ou por colunas

**Primeiro, a previsão.** Uma linha de cache tem 64 bytes: oito valores `float64`. Percorrer a memória em ordem traz os oito com uma busca; saltar uma linha inteira desperdiça sete oitavos de cada busca. Então, as combinações rápidas devem ser aquelas cujo loop interno anda pela memória contígua: `axis=1` na ordem C, `axis=0` na ordem F.

<!-- en:d1bcecf557 -->
### Como ler o resultado com honestidade

A previsão acerta a direção (as combinações desencontradas são as lentas), mas a diferença costuma ser menor do que a história de "8× menos bytes úteis por linha de cache" sugere, e às vezes `axis=0` na ordem C é quase tão rápido quanto `axis=1`. Duas coisas atenuam o efeito:

1. **Pré-busca do hardware.** Um passo constante é um padrão que a CPU reconhece, então ela busca adiantado e esconde boa parte da latência. Um acesso aleatório não teria essa ajuda.
2. **O acumulador cabe no cache.** Somar um array de ordem C ao longo de `axis=0` pode passar uma linha contígua por vez para um acumulador de 3000 elementos que fica residente no L1. As leituras são sequenciais; só as gravações se espalham.

A conclusão confiável não é um multiplicador fixo, e sim: **o layout é uma variável real, e quase não custa controlá-lo.** Quando um array é gravado uma vez e reduzido muitas vezes ao longo de um eixo, combine o layout com o padrão de acesso.

<!-- en:9c6f44396a -->
## Exercício 8.3: elimine os temporários

`np.sqrt(a**2 + b**2) / (c + 1)` aloca cinco temporários de tamanho completo: `a**2`, `b**2`, a soma deles, a raiz quadrada e `c + 1`. Cada um é uma alocação, uma passada completa pela memória e uma liberação.

<!-- en:c977e4bd9f -->
A economia de alocações é grande e inequívoca. A economia de **tempo** é modesta, e vale ser honesto sobre isso: a aritmética ainda precisa acontecer, e os temporários já estavam sendo reaproveitados do cache de blocos pequenos do NumPy.

Onde `out=` se justifica:

- **O array não cabe com folga na RAM.** Cinco temporários de um array de 20 GB não são uma lentidão, são um `OOMKilled`.
- **A expressão está dentro de um loop** (Exercício 1.5), em que as alocações se repetem milhares de vezes.
- **Você está escrevendo uma biblioteca** e não tem como saber o tamanho dos dados de quem chama.

E onde não se justifica: células pontuais de notebook com dados que cabem na memória. A versão com `out=` é bem mais difícil de ler: seis instruções onde havia uma, com o resultado escondido em `buf1`. Não pague esse custo sem uma análise de desempenho que o justifique.

<!-- en:9175ad3e5a -->
## Exercício 8.4: estatísticas em blocos

Uma passada, memória limitada. Acumule a contagem, a soma e a soma dos quadrados, e depois lembre-se do Exercício 3.5, porque `E[x²] − E[x]²` é um cancelamento catastrófico clássico quando a média está longe de zero.

<!-- en:ad6780f88a -->
Com média de 10⁶ e dispersão de 3, `E[x²]` fica em torno de 10¹² e `E[x]²` também; subtraí-los destrói os dígitos iniciais de uma quantidade que vale só 9. A versão deslocada subtrai uma média aproximada *antes* de elevar ao quadrado, então eleva números perto de zero e mantém a precisão completa. O custo é uma subtração extra por bloco.

Em produção, prefira a forma de **fusão paralela de Chan**: ela combina trios `(count, mean, M2)` de blocos independentes, então também funciona quando os blocos são processados fora de ordem ou em máquinas diferentes.

<!-- en:e032ec5a96 -->
## Exercício 8.5: saiba quando parar

Um máximo acumulado é inerentemente sequencial: `out[i]` depende de `out[i-1]`. É exatamente o tipo de problema em que as pessoas supõem que o NumPy não pode ajudar, e mesmo assim `np.maximum.accumulate` o resolve em C.

<!-- en:f014ad529b -->
### A ordem, e por quê

1. **`np.maximum.accumulate`**: uma chamada em C, uma passada pela memória, nenhum objeto Python criado. Esta é a resposta.
2. **Em blocos**: o mesmo loop em C, mais algumas iterações em Python e um `np.maximum` extra por bloco. Um pouco mais lento e sem benefício *aqui*, porque o array já cabe na memória. Ele se justifica quando a entrada é um memmap ou um fluxo maior que a RAM.
3. **O loop Python**: duas ordens de grandeza atrás, porque paga o despacho e o empacotamento do interpretador dez milhões de vezes.

### De onde vem o título do exercício

A versão em blocos tem mais código, mais estado e mais lugares para errar por um, e nesta entrada ela é *mais lenta*. Esta é a lição: **otimizar sem uma restrição medida é um custo, não um benefício.**

Recorra à versão mais complexa quando uma análise de desempenho ou um limite de memória mandar. Pergunte, em ordem: (1) está vetorizado? (2) é o algoritmo certo? (3) cabe na memória? Só depois dessas três a resposta deve envolver blocos, Numba ou Cython.

<!-- en:7550e9ab45 -->
---
## Você concluiu o Módulo 02

Quarenta exercícios. Os fios que vale levar para o Módulo 03:

| Ideia | Onde ela reapareceu |
| :--- | :--- |
| Views criam aliases; cópias, não | 2.2, 2.3, e todo `SettingWithCopyWarning` que você vai encontrar no pandas |
| Shapes se alinham à direita e depois esticam | 4.1, 4.2, 4.3, 5.2: a origem da maioria dos bugs numéricos silenciosos |
| `keepdims=True` ao dividir pela própria redução | 4.3, 5.2 |
| Expanda a álgebra antes de alocar `(n, m, d)` | 4.4, 4.5, 6.2 |
| Guarde estreito, acumule largo | 3.2, 3.3, 8.4 |
| Nunca subtraia dois números grandes quase iguais | 3.5, 4.4, 8.4 |
| `lstsq`/`solve`, nunca `inv` | 6.3, 6.4 |
| Aleatoriedade é um parâmetro, não um global | 7.1, 7.4, 7.5 |
| Meça antes de otimizar | 1.3, 1.5, 8.3, 8.5 |

**Próximo:** Módulo 03 — pandas e Manipulação de Dados. Um DataFrame é um conjunto de arrays NumPy com rótulos, e todas as regras acima continuam valendo por baixo, inclusive e principalmente as sobre views.

<!-- comments -->
EN: # Values above 256 so interning does not flatter the list.
PT: # Valores acima de 256, para o cache de inteiros não favorecer a lista.
EN: # `getsizeof` on an ndarray already includes the buffer, so it is the
PT: # `getsizeof` em um ndarray já inclui o buffer, então é o
EN: # array's honest total — the one case where the shallow measure is right.
PT: # total honesto do array: o único caso em que a medida rasa está certa.
EN: # More repeats at small n, where a single run is too short to time.
PT: # Mais repetições com n pequeno, em que uma única execução é curta demais para medir.
EN: # reshape is NOT always a view — proof:
PT: # reshape NÃO é sempre uma view; a prova:
EN: # `*` allocates; `*=` would not
PT: # `*` aloca; `*=` não alocaria
EN: # Way 1 — strided assignment. Fastest, and the one you will see in the wild.
PT: # Forma 1: atribuição com passo. A mais rápida, e a que você vai ver por aí.
EN: # odd rows,  even columns
PT: # linhas ímpares, colunas pares
EN: # even rows, odd columns
PT: # linhas pares, colunas ímpares
EN: # Way 2 — arithmetic on coordinates. A square is dark when (row + col) is odd.
PT: # Forma 2: aritmética com coordenadas. Uma casa é escura quando (row + col) é ímpar.
EN: # Way 3 — the same idea via broadcasting, without materialising two grids.
PT: # Forma 3: a mesma ideia com broadcasting, sem materializar duas grades.
EN: # Because they are views, writing through one changes `y`:
PT: # Como são views, gravar por meio de uma altera `y`:
EN: # The realistic version of that mistake:
PT: # A versão realista desse erro:
EN: # float64 accumulator by default
PT: # acumulador float64 por padrão
EN: # Fix 1 — widen the accumulator, keep the storage narrow.
PT: # Correção 1: amplia o acumulador e mantém o armazenamento estreito.
EN: #         Costs nothing: the array stays int16 in memory.
PT: #             Não custa nada: o array continua int16 na memória.
EN: # Fix 2 — widen the data before the arithmetic.
PT: # Correção 2: amplia os dados antes da aritmética.
EN: #         Costs 4× the memory, but every downstream operation is then safe.
PT: #             Custa 4× a memória, mas toda operação seguinte fica segura.
EN: # Fix 3 — the defensive habit: make the failure loud instead of silent.
PT: # Correção 3: o hábito defensivo: torna a falha clara em vez de silenciosa.
EN: # Clip at the float64 floor so log scale stays readable when error is 0.
PT: # Limita no piso do float64, para a escala logarítmica continuar legível com erro 0.
EN: # NaNs compare unequal to themselves, so compare them positionally.
PT: # NaNs são diferentes de si mesmos, então são comparados por posição.
EN: # protected
PT: # protegida
EN: # → uint8
PT: # → uint8
EN: # → uint16
PT: # → uint16
EN: # stays float64
PT: # continua float64
EN: # → float16
PT: # → float16
EN: # The direct formula.
PT: # A fórmula direta.
EN: # The rearranged one: parameterise by q = 1 - p and never form `p` at all.
PT: # A rearranjada: parametriza com q = 1 - p e nunca forma `p`.
EN: # The same disease, and the same cure, in three places you will actually meet it.
PT: # A mesma doença e a mesma cura, em três lugares em que você vai realmente encontrá-la.
EN: # noqa: E731
PT: # noqa: E731
EN: # (3, 1) — a column
PT: # (3, 1): uma coluna
EN: # (3,)   — a flat vector
PT: # (3,)  : um vetor plano
EN: # The assertion that would have caught the original bug. `residuals` is still
PT: # A asserção que teria pegado o bug original. `residuals` ainda é o
EN: # the buggy (3, 3) array, so it fires; we catch it here to show the message.
PT: # array errado (3, 3), então ela dispara; capturamos aqui para mostrar a mensagem.
EN: # runs, silently
PT: # roda, em silêncio
EN: # (1000, 1, 3) - (1, 8, 3) → (1000, 8, 3), one difference vector per pair.
PT: # (1000, 1, 3) - (1, 8, 3) → (1000, 8, 3), um vetor de diferença por par.
EN: # (n, 1)
PT: # (n, 1)
EN: # (1, k)
PT: # (1, k)
EN: # (n, k)
PT: # (n, k)
EN: # Rounding can push near-zero distances slightly negative; clip before sqrt.
PT: # O arredondamento pode deixar distâncias quase zero levemente negativas; limita antes da raiz.
EN: # note: rowsum[j], not rowsum[i]
PT: # atenção: rowsum[j], não rowsum[i]
EN: # 2. Replace, without a loop. `np.where` picks element-wise between two arrays.
PT: # 2. Substitui, sem loop. `np.where` escolhe elemento a elemento entre dois arrays.
EN: #    `mean` was computed BEFORE the replacement — using the post-replacement
PT: #    `mean` foi calculada ANTES da substituição; usar a média posterior
EN: #    mean would be circular.
PT: #    seria circular.
EN: # 3. Share in [80, 120]. `&` is element-wise AND; the parentheses are required
PT: # 3. Parcela em [80, 120]. `&` é o E elemento a elemento; os parênteses são obrigatórios
EN: #    because `&` binds tighter than `>=` in Python.
PT: #    porque `&` tem precedência sobre `>=` em Python.
EN: # scale factor makes MAD comparable to σ for normals
PT: # o fator de escala torna a MAD comparável a σ em distribuições normais
EN: # `bincount` cannot do max, so use the ufunc's unbuffered `.at`: for each i,
PT: # `bincount` não calcula máximo, então usa o `.at` sem buffer da ufunc: para cada i,
EN: # maxima[codes[i]] = max(maxima[codes[i]], values[i]).
PT: # maxima[codes[i]] = max(maxima[codes[i]], values[i]).
EN: # Negate so "k largest" becomes "k smallest", which is what kth refers to.
PT: # Inverte o sinal para "k maiores" virar "k menores", que é ao que kth se refere.
EN: # (n, 1), broadcasts
PT: # (n, 1), faz broadcast
EN: # sort just the k
PT: # ordena só os k
EN: # A zero-length vector has no direction; guard rather than divide by 0.
PT: # Um vetor de comprimento zero não tem direção; protege em vez de dividir por 0.
EN: # `lstsq` uses an SVD: it is stable, and it returns the minimum-norm
PT: # `lstsq` usa uma SVD: é estável e devolve a solução de norma mínima
EN: # solution rather than raising when the columns are rank-deficient.
PT: # em vez de gerar um erro quando as colunas têm posto incompleto.
EN: # `solve`, never `inv` — same answer, better conditioned, twice as fast.
PT: # `solve`, nunca `inv`: mesma resposta, melhor condicionamento, duas vezes mais rápido.
EN: # Two nearly identical columns, plus a third on a wildly different scale.
PT: # Duas colunas quase idênticas, mais uma terceira em escala bem diferente.
EN: # ~0.9999 correlated
PT: # correlação de ~0.9999
EN: # Standardise: zero mean, unit variance per column.
PT: # Padroniza: média zero e variância unitária por coluna.
EN: # legacy global API
PT: # API global antiga
EN: # consumes from the same global stream
PT: # consome do mesmo fluxo global
EN: # still touches the global stream...
PT: # ainda toca no fluxo global...
EN: # ...which this generator does not use
PT: # ...que este gerador não usa
EN: # (trials, k) birthdays; sort each row and look for equal neighbours.
PT: # (trials, k) aniversários; ordena cada linha e procura vizinhos iguais.
EN: # Two groups of customer spend. Right-skewed, as spend always is.
PT: # Dois grupos de gastos de clientes. Assimétricos à direita, como gastos sempre são.
EN: # One binomial draw per arm per simulated experiment: fully vectorised.
PT: # Um sorteio binomial por braço por experimento simulado: totalmente vetorizado.
EN: # A pooled rate of exactly 0 or 1 gives SE = 0; those tests cannot reject.
PT: # Uma taxa combinada de exatamente 0 ou 1 dá SE = 0; esses testes não conseguem rejeitar.
EN: # two-sided 5%
PT: # bilateral de 5%
EN: # Interpolate the crossing rather than reporting the nearest grid point.
PT: # Interpola o cruzamento em vez de informar o ponto mais próximo da grade.
EN: # The analytic normal approximation, for comparison.
PT: # A aproximação normal analítica, para comparação.
EN: # scikit-learn, for one, has strong opinions about layout.
PT: # O scikit-learn, por exemplo, tem opiniões fortes sobre layout.
EN: # buf1 = a²
PT: # buf1 = a²
EN: # buf2 = b²
PT: # buf2 = b²
EN: # buf1 = a² + b²
PT: # buf1 = a² + b²
EN: # buf1 = sqrt(a² + b²)
PT: # buf1 = sqrt(a² + b²)
EN: # buf2 = c + 1
PT: # buf2 = c + 1
EN: # buf1 = result
PT: # buf1 = resultado
EN: # Write a file too big to want in memory all at once, with a large offset in the
PT: # Grava um arquivo grande demais para querer na memória de uma vez, com um grande deslocamento
EN: # values — which is exactly the condition that breaks the naive formula.
PT: # nos valores, que é exatamente a condição que quebra a fórmula ingênua.
EN: # Population variance of the shifted data equals that of the original.
PT: # A variância populacional dos dados deslocados é igual à dos originais.
