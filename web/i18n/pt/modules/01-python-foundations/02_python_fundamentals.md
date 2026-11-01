<!-- en:0ad683b644 -->
# 02 · Fundamentos de Python

**Módulo 01 — Fundamentos de Python para Ciência de Dados**

---

## O que você vai aprender

1. O modelo de objetos do Python, e por que `b = a` não copia nada.
2. Mutabilidade, aliasing e o bug silencioso que eles causam em todo pipeline de dados.
3. Tipos numéricos: inteiros de precisão arbitrária, armadilhas de ponto flutuante e quando recorrer a `Decimal`.
4. Strings, f-strings e codificação de texto, a origem da maioria dos bugs de dados do tipo "na minha máquina funcionou".
5. Valores verdadeiros e falsos, `None` e os operadores de comparação que pegam as pessoas de surpresa.
6. Controle de fluxo, incluindo `for...else` e casamento de padrões estrutural.

**Tempo estimado:** 90 minutos

> Este é o notebook mais importante do módulo. Quase todo bug sutil que você vai escrever em pandas tem origem em algo desta página.

<!-- en:7db6f3bfa7 -->
## 1. Tudo é um objeto

Em Python, não existem "tipos primitivos" que se comportam de forma diferente de todo o resto. Um inteiro, uma função, uma classe, um módulo: tudo é objeto, e todo objeto tem exatamente três coisas:

| Propriedade | Como acessar | Pode mudar? |
| :--- | :--- | :--- |
| **Identidade**: seu endereço na memória | `id(obj)` | Nunca |
| **Tipo**: o que ele é | `type(obj)` | Nunca |
| **Valor**: o que ele contém | o próprio objeto | *Só se o tipo for mutável* |

<!-- en:1573c7f575 -->
## 2. Nomes são etiquetas, não caixas

Este é o conceito mais importante do notebook.

Se você aprendeu C, Java ou C#, provavelmente imagina uma variável como uma **caixa que guarda um valor**. Python não funciona assim. Em Python, um nome é uma **etiqueta presa a um objeto**, e a atribuição move a etiqueta: ela nunca copia o objeto.

![Variáveis em Python: nomes, não caixas](../../../assets/images/names_and_objects.png)

<!-- en:31eba428fc -->
### `==` versus `is`

Eles fazem perguntas realmente diferentes, e confundi-los é um bug clássico:

- `a == b`: "estes têm o mesmo **valor**?"
- `a is b`: "estes são o **mesmo objeto** na memória?"

<!-- en:a957ac133f -->
> [!WARNING]
> **Use `is` apenas para `None`, `True` e `False`.** Para todo o resto, use `==`.
>
> O motivo é que o CPython guarda em cache ("interna") alguns objetos imutáveis pequenos, então `is` pode parecer funcionar e depois quebrar com um valor maior. Esse é um detalhe de implementação em que você nunca deve confiar:

<!-- en:4a2f075a7c -->
## 3. Mutabilidade: onde moram os bugs

Todo tipo em Python é **mutável** (pode ser alterado no lugar) ou **imutável** (não pode). Só essa propriedade decide se `b = a` é inofensivo ou uma bomba-relógio.

| Imutável | Mutável |
| :--- | :--- |
| `int`, `float`, `bool`, `complex` | `list` |
| `str` | `dict` |
| `tuple` | `set` |
| `frozenset` | `bytearray` |
| `bytes` | **`numpy.ndarray`**, **`pandas.DataFrame`** |

![Mutável versus imutável](../../../assets/images/mutable_vs_immutable.png)

<!-- en:e6fc8d6d6a -->
### Por que isso importa no trabalho real com dados

O exemplo abstrato acima parece fácil de evitar. Não é, porque o aliasing costuma ficar escondido atrás de uma chamada de função.

<!-- en:1c3f5b5f5c -->
### A solução: copie antes de modificar

Python oferece dois tipos de cópia, e a diferença importa assim que os seus dados ficam aninhados, o que, com dados reais, sempre acontece.

| Cópia | Sintaxe | Copia o contêiner? | Copia o que está dentro? |
| :--- | :--- | :-: | :-: |
| **Rasa** | `list(x)`, `x[:]`, `copy.copy(x)` | ✔ | ✘ compartilha os objetos internos |
| **Profunda** | `copy.deepcopy(x)` | ✔ | ✔ copia tudo recursivamente |

<!-- en:77c4c024b7 -->
### O padrão profissional

Em vez de copiar defensivamente em todo lugar, prefira funções que **não modificam as próprias entradas**. Construa e devolva algo novo:

<!-- en:c251c63ed6 -->
> [!TIP]
> É exatamente por isso que a maioria dos métodos do pandas devolve um novo objeto em vez de editar no lugar, e por isso `inplace=True` é desaconselhado no pandas moderno. Você vai rever isso no Módulo 03.

<!-- en:e7577ebda0 -->
## 4. Números

### Inteiros não têm valor máximo

Ao contrário da maioria das linguagens, os inteiros de Python crescem até o tamanho que precisarem. Não há estouro nem teto de `int64`.

<!-- en:7ed3334f50 -->
> [!WARNING]
> **Os inteiros do NumPy NÃO se comportam assim.** O NumPy usa tipos de largura fixa para ganhar velocidade, então eles estouram sem aviso. Essa é uma fonte real de números errados em pipelines de produção, e o Módulo 02 trata disso em detalhes.

<!-- en:fd698c4073 -->
### Floats são aproximações

Números de ponto flutuante são guardados em binário (IEEE 754, 64 bits). Assim como 1/3 não pode ser escrito exatamente em decimal, **0.1 não pode ser escrito exatamente em binário**. Por isso, toda operação com float carrega um pequeno erro de representação.

<!-- en:2d335dcd7c -->
#### Como comparar floats corretamente

Nunca use `==` em floats calculados. Compare dentro de uma tolerância.

<!-- en:37531bd82e -->
#### Quando a exatidão é obrigatória: dinheiro

Em cálculos financeiros, o erro acumulado de float é inaceitável: reguladores e auditores não aceitam "errou por 1e-16 por linha, vezes 40 milhões de linhas". Use `decimal.Decimal`, que faz aritmética exata em base 10.

<!-- en:f4253dff14 -->
### Divisão de inteiros, divisão inteira e o módulo de números negativos

<!-- en:a684164051 -->
## 5. Strings

Strings são **sequências imutáveis de caracteres Unicode**. Toda "modificação" na verdade produz uma nova string.

<!-- en:725b8b351a -->
### Os métodos de string que você vai usar de verdade

Limpar colunas de texto é uma tarefa diária. Estes métodos resolvem a maior parte:

<!-- en:0c3017d275 -->
### f-strings: a formatação que você deve usar em todo lugar

As f-strings (Python 3.6+) são o jeito mais rápido e legível de construir strings. Vale aprender a minilinguagem de formatação depois do `:`, que elimina muito código manual de arredondamento.

<!-- en:656939a337 -->
### Codificação: a origem de metade dos seus erros ao carregar dados

Uma `str` é texto abstrato. Para gravá-la em disco ou enviá-la pela rede, ela precisa ser **codificada** em `bytes`. UTF-8 é o padrão correto; arquivos antigos costumam estar em Latin-1 ou Windows-1252, e errar o palpite corrompe os seus dados.

<!-- en:87d1ae1f94 -->
> [!TIP]
> Quando `pd.read_csv` gera um `UnicodeDecodeError`, o arquivo não está quebrado: ele apenas não está em UTF-8. Tente `encoding="latin-1"` ou `encoding="cp1252"`. O notebook 06 trata disso direito.

<!-- en:146c3ab125 -->
## 6. Booleanos, `None` e valores verdadeiros ou falsos

Todo objeto Python pode ser avaliado em um contexto booleano. As regras são curtas e vale memorizá-las, porque deixam o Python idiomático muito mais conciso.

<!-- en:82d665758b -->
### Verificações idiomáticas de vazio

<!-- en:8ca88cb6ae -->
> [!WARNING]
> Esta expressão é um dos poucos lugares em que **pandas e NumPy rompem deliberadamente com o Python puro**. Chamar `bool()` em um array com mais de um elemento é ambíguo (deveria significar "algum" ou "todos"?), então eles geram um erro.

<!-- en:9a121f89be -->
### `None` não é zero, nem dado ausente

`None` é o objeto nulo do Python. Em ciência de dados, você vai encontrar **três tipos diferentes de "ausente"**, e eles não são intercambiáveis:

<!-- en:4efff9bf84 -->
> [!WARNING]
> `nan != nan` é o motivo de `df[df["col"] == np.nan]` devolver, sem aviso, **zero linhas** em vez dos seus valores ausentes. A chamada correta é `df[df["col"].isna()]`. Esse erro chega à produção com frequência.

<!-- en:c621383945 -->
## 7. Controle de fluxo

### Condicionais e comparações encadeadas

<!-- en:3a5503495c -->
### Loops: percorra objetos, não índices

O sinal mais comum de quem vem de outra linguagem é um loop sobre `range(len(x))`.

<!-- en:17497327d5 -->
### `break`, `continue` e a cláusula `for...else`

`for...else` é um recurso realmente útil de Python que quase ninguém conhece: o bloco `else` roda **apenas se o loop terminar sem passar por um `break`**.

<!-- en:e2ca825953 -->
### Casamento de padrões estrutural (`match`)

Adicionado no Python 3.10, o `match` é muito mais que um `switch`: ele desestrutura os dados enquanto faz o casamento. É excelente para tratar respostas heterogêneas de APIs.

<!-- en:d79d525d24 -->
### O operador morsa `:=`

Atribui *e* devolve um valor em uma única expressão. Use-o quando, sem ele, você calcularia algo duas vezes ou adicionaria uma linha descartável.

<!-- en:6d121ffe7f -->
## 8. Exercícios

### Exercício 2.1: preveja antes de executar

Para cada trecho, escreva a saída **antes** de executá-lo. Depois, confira.

```python
# (a)
a = [1, 2, 3]
b = a
b = b + [4]          # atenção: NÃO é b += [4]
print(a)

# (b)
a = [1, 2, 3]
b = a
b += [4]
print(a)

# (c)
t = ([1, 2], 3)
t[0].append(99)
print(t)
```

Explique *por que* (a) e (b) diferem, e por que (c) funciona mesmo com tuplas sendo imutáveis.

### Exercício 2.2: um bug de comparação de floats

Esta função está errada. Encontre o bug e corrija.

```python
def is_fully_allocated(weights: list[float]) -> bool:
    """Devolve True quando os pesos da carteira somam 1."""
    return sum(weights) == 1.0

is_fully_allocated([0.1, 0.2, 0.3, 0.4])   # devolve False!
```

### Exercício 2.3: enriquecimento seguro

Escreva `add_discount(orders, pct)`, que devolve novos dicionários de pedido com a chave `discounted_price`. Prove com uma asserção que a lista de entrada não foi modificada.

### Exercício 2.4: limpeza de texto

Escreva `clean_city(raw)`, que transforma `"  sÃO   paulo\n"` em `"Sao Paulo"`: remova os espaços das pontas, reduza sequências internas de espaços e deixe o resultado com iniciais maiúsculas.

*Dica:* `" ".join(raw.split())` reduz os espaços em um só passo.

### Exercício 2.5: auditoria de valores ausentes

Dado `values = [1.0, np.nan, 3.0, None, 5.0]`, conte quantas entradas estão "ausentes". Depois, explique por que `values.count(np.nan)` dá a resposta errada.

<!-- en:e26839f194 -->
## Principais lições

- **Nomes são etiquetas presas a objetos.** A atribuição nunca copia.
- **A mutabilidade decide tudo.** Funções que modificam seus argumentos causam bugs invisíveis no ponto da chamada; devolva novos objetos.
- `copy.copy` é rasa; dados aninhados precisam de `copy.deepcopy`.
- **Floats são aproximados.** Compare com `math.isclose` / `np.isclose`; use `Decimal` para dinheiro.
- Inteiros de Python não têm limite; **inteiros do NumPy estouram sem aviso**.
- Texto precisa ser codificado em bytes. UTF-8 é o padrão; um `UnicodeDecodeError` significa "codec errado", não "arquivo quebrado".
- **`nan != nan`**: sempre teste valores ausentes com `.isna()` / `np.isnan`.
- Percorra objetos, não índices: `enumerate`, `zip(..., strict=True)`.

---

**Próximo:** [03 · Estruturas de Dados](03_data_structures.ipynb), e por que escolher um `set` em vez de uma `list` pode transformar 40 minutos em 2 segundos.

<!-- comments -->
EN: # a memory address; yours will differ
PT: # um endereço de memória; o seu será diferente
EN: # Even a function is an object with the same three properties.
PT: # Até uma função é um objeto, com as mesmas três propriedades.
EN: # Assigning a function to a new name does not call it — it just binds
PT: # Atribuir uma função a um novo nome não a chama: apenas liga
EN: # another name to the same function object.
PT: # outro nome ao mesmo objeto de função.
EN: # does NOT copy the list — binds a second label to the SAME list
PT: # NÃO copia a lista: liga uma segunda etiqueta à MESMA lista
EN: # `is` compares identity
PT: # `is` compara a identidade
EN: # a separate list that happens to hold equal values
PT: # uma lista separada que por acaso tem valores iguais
EN: # another label for list_1
PT: # outra etiqueta para list_1
EN: # Small integers (-5 to 256) are pre-allocated and shared by CPython…
PT: # Inteiros pequenos (-5 a 256) são pré-alocados e compartilhados pelo CPython…
EN: # …but larger ones are not.
PT: # …mas os maiores não são.
EN: # Both are equal, which is what you actually care about.
PT: # Os dois são iguais, e é isso que realmente importa.
EN: # --- Immutable: safe -------------------------------------------------------
PT: # --- Imutável: seguro -------------------------------------------------------
EN: # cannot modify 10 in place → creates a NEW int, rebinds b
PT: # não dá para modificar 10 no lugar → cria um NOVO int e religa b
EN: # --- Mutable: dangerous ----------------------------------------------------
PT: # --- Mutável: perigoso ------------------------------------------------------
EN: # modifies the list IN PLACE → both names see the change
PT: # modifica a lista NO LUGAR → os dois nomes veem a mudança
EN: # --- Shallow copy: a new outer list, but the SAME inner dicts --------------
PT: # --- Cópia rasa: uma nova lista externa, mas os MESMOS dicts internos -------
EN: # --- Deep copy: everything duplicated, all the way down -------------------
PT: # --- Cópia profunda: tudo duplicado, até o fundo ----------------------------
EN: # reset
PT: # reinicia
EN: # `{**record, "total": ...}` builds a new dict from the old one plus a key.
PT: # `{**record, "total": ...}` constrói um novo dict a partir do antigo mais uma chave.
EN: # Handy for readability: underscores are ignored in numeric literals.
PT: # Útil para a leitura: sublinhados são ignorados em literais numéricos.
EN: # Doubling 2**62 lands on 2**63, one past the int64 ceiling. Instead of raising,
PT: # Dobrar 2**62 chega a 2**63, um além do teto do int64. Em vez de gerar um erro,
EN: # the value wraps around to the most negative representable number.
PT: # o valor dá a volta para o número mais negativo representável.
EN: # Standard library: math.isclose (relative + absolute tolerance).
PT: # Biblioteca padrão: math.isclose (tolerância relativa + absoluta).
EN: # NumPy equivalent, and its array-wide version — this is what you will use daily.
PT: # O equivalente em NumPy e sua versão para arrays: é isso que você vai usar todo dia.
EN: # Note: build Decimals from STRINGS. Decimal(0.1) would faithfully copy the
PT: # Atenção: construa Decimals a partir de STRINGS. Decimal(0.1) copiaria fielmente o
EN: # float's error, defeating the point entirely.
PT: # erro do float, o que anula todo o propósito.
EN: # Python floors toward negative infinity, unlike C/Java which truncate toward 0.
PT: # Python arredonda para o infinito negativo, ao contrário de C/Java, que truncam em direção a 0.
EN: # This surprises people, but it makes the modulo result's sign predictable —
PT: # Isso surpreende, mas torna previsível o sinal do resultado do módulo,
EN: # handy for cyclical features such as hour-of-day or day-of-week.
PT: # útil para atributos cíclicos como hora do dia ou dia da semana.
EN: # Indexing and slicing work as on any sequence: [start:stop:step], stop excluded.
PT: # Indexação e fatiamento funcionam como em qualquer sequência: [start:stop:step], sem incluir stop.
EN: # `join` is the efficient way to concatenate many strings. Building a string with
PT: # `join` é o jeito eficiente de concatenar muitas strings. Construir uma string com
EN: # `+=` in a loop is O(n²) because each step allocates a whole new string.
PT: # `+=` em um loop é O(n²), porque cada passo aloca uma string inteira nova.
EN: # `=` prints the expression AND its value — superb for debugging.
PT: # `=` imprime a expressão E o seu valor: ótimo para depurar.
EN: # Decoding with the wrong codec produces "mojibake" — or an outright error.
PT: # Decodificar com o codec errado produz "mojibake" ou um erro direto.
EN: # These are all "falsy":
PT: # Todos estes são "falsos":
EN: # Two that surprise people:
PT: # Dois que surpreendem:
EN: # Idiomatic — reads naturally and works for any container.
PT: # Idiomático: lê-se com naturalidade e funciona para qualquer contêiner.
EN: # Works, but noisier.
PT: # Funciona, mas é mais poluído.
EN: # NaN's defining, deeply surprising property: it is not equal to itself.
PT: # A propriedade definidora e muito surpreendente de NaN: ele não é igual a si mesmo.
EN: # Python allows `a <= x < b`, which reads like maths and evaluates
PT: # Python permite `a <= x < b`, que se lê como matemática e avalia
EN: # `x` only once. Most languages require `x >= a and x < b`.
PT: # `x` só uma vez. A maioria das linguagens exige `x >= a and x < b`.
EN: # ✘ C-style — works, but noisy and error-prone.
PT: # ✘ Estilo C: funciona, mas é poluído e propenso a erros.
EN: # ✔ Idiomatic — iterate the object directly.
PT: # ✔ Idiomático: percorra o objeto diretamente.
EN: # ✔ When you genuinely need the index, use enumerate.
PT: # ✔ Quando realmente precisar do índice, use enumerate.
EN: # ✔ To walk two sequences together, use zip.
PT: # ✔ Para percorrer duas sequências juntas, use zip.
EN: # `strict=True` (Python 3.10+) raises if the lengths differ, rather than
PT: # `strict=True` (Python 3.10+) gera um erro se os tamanhos forem diferentes, em vez de
EN: # silently truncating — which has hidden many data-alignment bugs.
PT: # truncar sem aviso, algo que já escondeu muitos bugs de alinhamento de dados.
EN: # Reached only when `break` never fired.
PT: # Só é alcançado quando o `break` nunca é executado.
EN: # Match a specific key/value AND bind another key in one step.
PT: # Casa uma chave/valor específica E liga outra chave em um único passo.
EN: # Without the walrus: you must either call the function twice, or add a line.
PT: # Sem o morsa: você precisa chamar a função duas vezes ou adicionar uma linha.
