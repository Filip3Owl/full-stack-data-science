<!-- en:1fa1797549 -->
# 03 · Dtypes e Precisão Numérica

**Módulo 02 — NumPy e Computação Numérica**

---

## O que você vai aprender

1. O zoológico de dtypes, e como escolher deliberadamente em vez de por acidente.
2. **Estouro silencioso de inteiros**: a promessa do Módulo 01, cumprida por inteiro.
3. Regras de promoção de tipos, incluindo o que mudou no NumPy 2.0.
4. `float32` e `float64`: a troca entre memória, velocidade e exatidão, quantificada.
5. Épsilon de máquina e cancelamento catastrófico.
6. `NaN` e `inf`: como se propagam e como controlar os avisos.
7. Como reduzir em 75% a memória de um conjunto de dados com a redução correta de tipos.

**Tempo estimado:** 75 minutos

<!-- en:153719ee3b -->
## 1. O zoológico de dtypes

Ao contrário de Python, o NumPy usa tipos de **largura fixa**. Essa é a origem tanto da sua velocidade quanto do seu lado mais perigoso.

<!-- en:d68810ce31 -->
`precision` é o número de **dígitos decimais em que você pode confiar**. `float32` oferece cerca de 7; `float64`, cerca de 15. Esse número decide se um cálculo é confiável, e vale guardá-lo na memória.

<!-- en:e4e4dcec1f -->
## 2. Estouro silencioso de inteiros

O Módulo 01 avisou que os inteiros do NumPy dão a volta em vez de crescer. Aqui está o quadro completo, e ele é pior que "dá a volta", por causa de *onde* isso acontece em silêncio.

<!-- en:768d1c1cba -->
### A parte perigosa: arrays dão a volta sem aviso

<!-- en:f1f2cb86f1 -->
> [!WARNING]
> **A aritmética de arrays estoura em silêncio.** Sem exceção, sem aviso: só um número errado que flui para todas as agregações seguintes. Em um pipeline de receita, isso aparece como um total negativo; em uma coluna de IDs, como colisões. Nenhum dos dois é pego por um teste que só verifica nulos.

<!-- en:5b0d1528f5 -->
### Onde isso realmente causa problema, e onde não causa

Vale entender esta nuance com exatidão, porque a resposta intuitiva está errada. Considere visualizações diárias de página guardadas como `int32` para economizar memória:

<!-- en:a0fcadc34b -->
**Reduções são mais seguras do que você imaginaria; a aritmética elemento a elemento, não.**

`sum`, `mean`, `prod` e `cumsum` promovem o *acumulador* para o inteiro padrão da plataforma (`int64` aqui) quando a entrada é de um tipo inteiro mais estreito. Assim, `int32_array.sum()` é seguro por padrão.

Operações elemento a elemento (`*`, `+`, `**`) não têm essa proteção. O resultado mantém o dtype da entrada e dá a volta em silêncio. **É aí que o bug mora:** não na agregação que preocupava você, mas no inofensivo `views * price` duas linhas antes.

<!-- en:f3255d0b6b -->
> [!TIP]
> `sum` aceita um argumento `dtype=` que força o acumulador. Use-o ao somar `float32` (veja a seção 4) ou quando tiver alterado o padrão.

<!-- en:ba212bdd9b -->
### Como detectar o estouro em vez de torcer

`np.errstate` controla como o NumPy reporta erros de ponto flutuante e de inteiros. Defini-lo como `"raise"` transforma o silêncio em uma exceção.

<!-- en:cc9b4f8711 -->
## 3. Promoção de tipos

Quando os operandos diferem, o NumPy promove para um tipo que comporte os dois. O NumPy 2.0 mudou isso de forma significativa (NEP 50), então exemplos escritos antes de 2023 podem não corresponder ao que você vê.

<!-- en:77d7099116 -->
As duas últimas linhas são a mudança da NEP 50, e elas importam:

**Um escalar Python não promove mais um array.** `float32_array + 1.0` continua `float32`. Antes do NumPy 2.0, virava `float64`, dobrando a memória sem aviso em pipelines de deep learning. A nova regra é "o array vence", o que é mais previsível e geralmente é o que você queria.

<!-- en:1881190d99 -->
### A divisão de inteiros sempre produz um float

<!-- en:a04e7568b5 -->
## 4. `float32` e `float64`

`float64` é o padrão do NumPy e a escolha certa na maioria das vezes. `float32` reduz a memória pela metade e muitas vezes roda mais rápido, ao custo de cerca de 8 dígitos decimais de precisão. Deep learning usa `float32` (e `float16`) em toda parte; estatística geralmente não deveria usar.

<!-- en:1ed7edfed8 -->
### O custo em exatidão: absorção

O modo de falha mais gritante é a **absorção**: somar um número pequeno a um grande e vê-lo desaparecer por completo, porque o resultado não consegue representar a diferença.

<!-- en:f55e230b20 -->
### O custo em exatidão: acumulação

<!-- en:c4925a00ae -->
> [!NOTE]
> O `float32` erra por apenas 0.125 em dez milhões, muito melhor do que você poderia temer. O motivo é que **o `.sum()` do NumPy usa soma em pares**, que faz o erro crescer como √n em vez de n. Você está se beneficiando de um bom algoritmo, não da exatidão do `float32`.

Escreva você mesmo o total acumulado ingênuo, e a proteção desaparece:

<!-- en:5d767a6ebb -->
**Quase 1% de erro** vindo de um loop que parece totalmente razoável. Esse é o melhor argumento para nunca escrever à mão uma rotina numérica que você poderia chamar da biblioteca.

Quando precisar somar dados `float32`, amplie o acumulador:

<!-- en:e9abdd91bc -->
## 5. Épsilon de máquina e cancelamento catastrófico

O **épsilon** é o menor número que ainda altera 1.0 ao ser somado a ele. Ele define a resolução de todo o tipo.

<!-- en:bef8f531fc -->
### Cancelamento catastrófico

Subtrair dois números quase iguais destrói a precisão: os dígitos iniciais se cancelam, e o que sobra é o erro de arredondamento. Essa é a armadilha numérica clássica, e ela se esconde em fórmulas de aparência inocente.

<!-- en:0cf7e57289 -->
A fórmula ingênua subtrai dois números em torno de 10¹⁶ para chegar a uma resposta em torno de 1. Dezesseis dígitos significativos se cancelam, e o `float64` só tem cerca de quinze, então o resultado é inteiramente ruído de arredondamento.

**É por isso que você usa `np.var` em vez de escrever a fórmula à mão.** A biblioteca implementa a versão estável. A lição vale de modo geral: em operações numéricas padrão, prefira a função da biblioteca à expressão do livro-texto.

<!-- en:191eb0094f -->
## 6. `NaN` e `inf`

<!-- en:255b57d554 -->
### NaN se propaga por tudo

<!-- en:b35c18837a -->
> [!WARNING]
> As funções `nan*` **ignoram** valores ausentes. Isso é conveniente e às vezes errado: `nanmean` de `[1, nan, nan, nan]` é `1.0`, calculado a partir de uma única observação, e ela não vai avisar você. Sempre confira também a contagem.

<!-- en:cad68a0fa8 -->
### Como controlar os avisos

<!-- en:01f3d1d78a -->
## 7. Como reduzir a memória com a redução correta de tipos

Escolher o dtype *seguro* mais estreito é uma das otimizações de maior impacto no trabalho com dados, mas "mais estreito" precisa ser verificado, não adivinhado.

<!-- en:29284606a2 -->
> [!WARNING]
> **Reduza IDs com cuidado.** Um `int32` guarda até ~2,1 bilhões: serve para um ID de usuário hoje e vira uma colisão silenciosa no dia em que a tabela crescer. Quando uma coluna é uma chave, não uma medida, deixe-a larga ou guarde-a como string.

<!-- en:8831a3ec46 -->
## 8. `datetime64` e `timedelta64`

<!-- en:7ec85e5c5b -->
> [!NOTE]
> `datetime64` não tem fuso horário. Ele representa um instante ingênuo. Para qualquer coisa que envolva fusos ou calendários, o `Timestamp` do pandas (Módulo 03) é a ferramenta certa: ele envolve o `datetime64` e adiciona a semântica que falta.

<!-- en:dd905e916e -->
## 9. Exercícios

### Exercício 3.1: preveja o dtype

Diga o dtype do resultado de cada expressão e depois verifique:

```python
np.array([1, 2], dtype=np.int8) + np.array([1, 2], dtype=np.int16)
np.array([1, 2], dtype=np.int32) / 2
np.array([1.0], dtype=np.float32) + 1.0
np.array([1.0], dtype=np.float32) + np.float64(1.0)
np.array([True, False]) + np.array([1, 2])
np.arange(3) ** 2
```

### Exercício 3.2: provoque e depois evite um estouro

Construa um array `int16` cuja soma estoure. Mostre a resposta errada e depois produza a certa de duas formas diferentes.

### Exercício 3.3: quando o float32 não basta?

Encontre o tamanho de array em que a soma de valores `float32` iguais a 0.1 se afasta mais de 0.1% do total real. Faça um gráfico do erro relativo em função do tamanho para `float32` e `float64`.

### Exercício 3.4: construa um relatório de memória

Escreva `memory_report(arrays: dict) -> None`, que imprime por coluna: o dtype atual, o tamanho atual, o dtype seguro mais estreito e a economia possível. Recuse sugerir uma redução para qualquer coluna cujo nome termine em `_id`.

### Exercício 3.5: cancelamento catastrófico na prática

A fórmula do erro padrão de uma proporção é `sqrt(p*(1-p)/n)`. Para `p = 1 - 1e-17` e `n = 1000`, calcule diretamente e depois por uma forma rearranjada. Explique a diferença.

<!-- en:b213486a15 -->
## Principais lições

- Os dtypes do NumPy têm **largura fixa**. Inteiros dão a volta; eles não crescem.
- **O estouro em arrays é silencioso**, sem nenhum aviso. Só o caminho do escalar avisa.
- **Reduções promovem o acumulador** (`int32_arr.sum()` é seguro); **a aritmética elemento a elemento, não** (`int32_arr * 400` dá a volta). O bug quase sempre está na multiplicação, não na soma.
- Use **`np.errstate(over="raise")`** para transformar o silêncio em uma exceção ao depurar.
- Desde o NumPy 2.0 (**NEP 50**), um escalar Python não promove mais um array.
- **`float32` oferece ~7 dígitos decimais; `float64`, ~15.** Perto de 1e8, o `float32` não consegue nem representar +1: valores pequenos são absorvidos por completo.
- O `.sum()` do NumPy usa **soma em pares**; o seu total acumulado escrito à mão, não, e ele se desvia ~1% em um milhão de somas com `float32`.
- Nunca escreva `E[x²] - E[x]²`. Use `np.var`: a biblioteca implementa a forma numericamente estável.
- `NaN` se propaga por todas as agregações. As funções `nan*` o ignoram: **sempre informe a contagem junto**.
- **A redução de tipos precisa ser verificada**, e colunas de ID geralmente devem continuar largas.

---

**Próximo:** [04 · Broadcasting](04_broadcasting.ipynb): o mecanismo que permite ao NumPy combinar arrays de shapes diferentes, e as duas regras que o regem.

<!-- comments -->
EN: # A Python int grows without limit.
PT: # Um int de Python cresce sem limite.
EN: # An int64 array does not. 2**63 is one past the ceiling.
PT: # Um array int64, não. 2**63 é um além do teto.
EN: # The scalar path DOES warn, which makes the inconsistency easy to miss.
PT: # O caminho do escalar AVISA, o que faz a inconsistência passar despercebida.
EN: # The defensive pattern for element-wise work: widen first, deliberately.
PT: # O padrão defensivo para operações elemento a elemento: amplie antes, de propósito.
EN: # Default: silent.
PT: # Padrão: silencioso.
EN: # Opt into loudness. Note this covers ufunc errors; use it around the risky part.
PT: # Opte pelo alerta. Isto cobre erros de ufuncs; use em torno do trecho arriscado.
EN: # You can always ask what two types will produce, without running the operation.
PT: # Você sempre pode perguntar o que dois tipos vão produzir, sem executar a operação.
EN: # The consequence people trip over: `/` silently doubles your memory.
PT: # A consequência que pega as pessoas: `/` dobra a sua memória sem aviso.
EN: # The threshold where it happens is just eps × magnitude.
PT: # O limite em que isso acontece é só eps × magnitude.
EN: # The textbook variance formula, E[x²] - E[x]², versus the two-pass formula.
PT: # A fórmula de variância do livro-texto, E[x²] - E[x]², contra a fórmula de duas passadas.
EN: # algebraically correct
PT: # algebricamente correta
EN: # numerically stable
PT: # numericamente estável
EN: # what NumPy actually does
PT: # o que o NumPy realmente faz
EN: # Division by zero warns by default and produces inf / -inf / nan.
PT: # A divisão por zero avisa por padrão e produz inf / -inf / nan.
EN: # Turn them into errors while debugging — much easier than hunting a NaN
PT: # Transforme em erros ao depurar: é muito mais fácil do que rastrear um NaN
EN: # backwards through ten transformations.
PT: # de trás para a frente por dez transformações.
EN: # The safe pattern for element-wise division with a guard.
PT: # O padrão seguro para divisão elemento a elemento com proteção.
EN: # int64 by default
PT: # int64 por padrão
EN: # float64
PT: # float64
EN: # already 1 byte
PT: # já tem 1 byte
EN: # Try narrowest first; np.can_cast checks the VALUES, not just the type.
PT: # Tenta primeiro o mais estreito; np.can_cast verifica os VALORES, não só o tipo.
EN: # Only accept the demotion if nothing changed. For scores in [0, 100]
PT: # Só aceita a redução se nada mudar. Para notas em [0, 100]
EN: # this holds; for values needing 15 digits it will not.
PT: # isso se mantém; para valores que precisam de 15 dígitos, não.
EN: # Verify nothing was corrupted — a downcast that loses data is not an optimisation.
PT: # Verifica que nada foi corrompido: uma redução que perde dados não é otimização.
EN: # Arithmetic works naturally.
PT: # A aritmética funciona naturalmente.
