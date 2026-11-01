<!-- en:93eb6a8c15 -->
# 05 · Programação Orientada a Objetos

**Módulo 01 — Fundamentos de Python para Ciência de Dados**

---

## O que você vai aprender

1. Quando uma classe é a ferramenta certa, e quando uma função é melhor.
2. Classes, instâncias e a diferença entre atributos de classe e de instância.
3. Métodos dunder: como fazer seus objetos funcionarem com `print`, `len`, `==` e `[]`.
4. `@dataclass`: a maior parte do código repetitivo, eliminada.
5. `@property` para atributos validados e calculados.
6. Herança e composição, e por que o mercado prefere composição.
7. **O padrão de estimadores do scikit-learn**, a interface que você vai implementar de verdade.

**Tempo estimado:** 75 minutos

<!-- en:3174b640af -->
## 1. Quando quem trabalha com ciência de dados deve escrever uma classe?

Resposta honesta: **menos vezes do que os tutoriais sugerem.** A maior parte do código de análise deve ser feita de funções que operam sobre DataFrames. Recorra a uma classe quando houver **estado e comportamento que andam juntos**:

| Situação | Use |
| :--- | :--- |
| Transformar dados uma vez e devolver um resultado | uma **função** |
| Agrupar constantes relacionadas | um **módulo** ou `Enum` |
| Carregar um conjunto fixo de campos | uma **`@dataclass`** |
| Algo que *aprende* parâmetros e depois os *aplica* | uma **classe** |
| Várias implementações intercambiáveis atrás de uma interface | uma **hierarquia de classes** ou `Protocol` |

A quarta linha é a mais importante em ciência de dados: um scaler aprende uma média e um desvio padrão a partir dos dados de treino e depois os aplica a dados novos. Esse estado ajustado é exatamente a razão de ser de um objeto.

<!-- en:e97b03d2ce -->
## 2. Classes e instâncias

<!-- en:f74d524198 -->
### Atributos de classe e de instância: uma armadilha

Atributos de classe são compartilhados. Se um atributo **mutável** for modificado por qualquer instância, todas as instâncias veem a mudança. É a lição do modelo de objetos do notebook 02 com uma roupa nova.

<!-- en:febea8a761 -->
## 3. Métodos dunder

Os métodos "dunder" (de *double underscore*, sublinhado duplo) permitem que seus objetos participem da sintaxe embutida de Python. Implementar alguns deles faz uma classe parecer nativa.

<!-- en:69db710de4 -->
### Métodos dunder de contêiner

Implemente `__len__`, `__getitem__` e `__iter__`, e a sua classe passa a funcionar com `len()`, `[]`, `for`, `in` e desempacotamento, sem custo extra.

<!-- en:e604ba6965 -->
## 4. `@dataclass`: elimine o código repetitivo

A maioria das classes no trabalho com dados apenas guarda campos. `@dataclass` gera `__init__`, `__repr__` e `__eq__` a partir de atributos com anotação de tipo.

<!-- en:918b589406 -->
### Dataclasses congeladas: configuração imutável

`frozen=True` torna as instâncias imutáveis e hasheáveis. Essa é a escolha certa para objetos de configuração: garante que nenhum trecho do código reescreva um hiperparâmetro sem aviso no meio de uma execução.

<!-- en:14a2b17c40 -->
## 5. `@property`: atributos calculados e validados

Uma property parece um atributo no ponto da chamada, mas executa código. Use-a para validar na atribuição ou para expor um valor derivado que deve estar sempre consistente com suas entradas.

<!-- en:e6ef65568a -->
## 6. Herança e composição

A herança expressa uma relação **"é um"**. A composição expressa **"tem um"**. A prática moderna favorece fortemente a composição, porque hierarquias profundas de herança ficam impossíveis de entender.

<!-- en:8ae112a325 -->
### Composição: o padrão que você vai usar com mais frequência

<!-- en:8007f27a36 -->
## 7. O padrão de estimadores do scikit-learn

Todo objeto do scikit-learn segue um único contrato. Implemente-o, e o seu código personalizado se encaixa direto em um `Pipeline`, um `GridSearchCV` ou um `cross_val_score`. Essa é a habilidade mais valiosa ao escrever classes em ciência de dados aplicada.

| Método | Contrato |
| :--- | :--- |
| `fit(X, y=None)` | **Aprende** parâmetros a partir dos dados de treino. Guarda-os em `self` com um sublinhado no final (`self.mean_`). Devolve `self`. |
| `transform(X)` | **Aplica** os parâmetros aprendidos. Nunca aprende aqui. |
| `fit_transform(X)` | Atalho, herdado sem custo de `TransformerMixin`. |

> [!WARNING]
> O sublinhado no final não é enfeite: é a convenção que marca um atributo como *aprendido a partir dos dados*. `self.strategy` é um hiperparâmetro escolhido pelo usuário; `self.median_` é algo que `fit` calculou.

<!-- en:97af505136 -->
### A recompensa: ela se combina com o resto do scikit-learn

<!-- en:0466e7713b -->
## 8. Exercícios

### Exercício 5.1: classe ou função?

Para cada tarefa, decida se você escreveria uma função, uma dataclass ou uma classe completa, e justifique em uma frase:

1. Converter uma temperatura de Celsius para Fahrenheit.
2. Guardar as configurações de conexão de um banco de dados.
3. Um tokenizador que constrói um vocabulário a partir de um corpus de treino e depois codifica textos novos.
4. Calcular a mediana de uma lista.

### Exercício 5.2: prática de métodos dunder

Estenda `Measurement` com `__sub__`, `__mul__` (por um número simples) e `__format__`, para que `f"{m:.2f}"` funcione.

### Exercício 5.3: conversão para dataclass

Reescreva a classe `Dataset` da seção 2 como uma `@dataclass`. Mantenha `mean()` e `summary()`. Quantas linhas você eliminou?

### Exercício 5.4: construa um estimador

Escreva `RareCategoryGrouper(BaseEstimator, TransformerMixin)`, que:

- em `fit`, aprende quais categorias aparecem em menos de `min_frequency` linhas;
- em `transform`, substitui essas categorias por `"__other__"`.

Confirme que funciona dentro de um `SklearnPipeline`. Esta é uma etapa real de pré-processamento, usada para impedir que colunas categóricas de alta cardinalidade explodam.

### Exercício 5.5: encontre o bug

```python
class Model:
    history = []
    def fit(self, X, y):
        self.history.append("fitted")
        return self
```

Crie duas instâncias de `Model` e ajuste cada uma uma vez. Explique `len(m1.history)`.

<!-- en:37ba385522 -->
## Principais lições

- Use uma classe quando **estado e comportamento andam juntos**, geralmente quando algo é *ajustado* e depois *aplicado*.
- **Atributos de classe são compartilhados.** Os mutáveis são um bug esperando para acontecer; inicialize o estado de cada instância em `__init__`.
- Implemente `__repr__` em toda classe que escrever: o seu eu do futuro vai lê-lo em um depurador.
- `@dataclass` elimina o código repetitivo; `frozen=True` cria objetos de configuração imutáveis e hasheáveis.
- `@property` valida na atribuição e expõe valores derivados com segurança.
- **Prefira composição a herança.**
- Siga o **contrato de estimadores do scikit-learn** (`fit` aprende e guarda `attribute_`, `transform` aplica, `__init__` só guarda os argumentos), e o seu código se combina com todo o ecossistema.

---

**Próximo:** [06 · Arquivos e Formatos de Dados](06_files_and_data_formats.ipynb): como ler e gravar dados sem corrompê-los.

<!-- comments -->
EN: # A CLASS attribute — shared by every instance. Use these for constants.
PT: # Um atributo de CLASSE, compartilhado por todas as instâncias. Use para constantes.
EN: # instance attribute
PT: # atributo de instância
EN: # copy, so the caller's list stays independent
PT: # copia, para a lista de quem chama continuar independente
EN: # `self` is the instance the method was called on. It is passed
PT: # `self` é a instância em que o método foi chamado. Ele é passado
EN: # automatically: `d.mean()` is really `Dataset.mean(d)`.
PT: # automaticamente: `d.mean()` é na verdade `Dataset.mean(d)`.
EN: # BUG: one list, shared by every instance ever created
PT: # BUG: uma única lista, compartilhada por todas as instâncias já criadas
EN: # a fresh list per instance
PT: # uma lista nova por instância
EN: # let Python try the reflected operation
PT: # deixa Python tentar a operação refletida
EN: # Supporting slices as well as integers costs one line and makes the
PT: # Aceitar fatias além de inteiros custa uma linha e faz a
EN: # class behave the way users expect a sequence to behave.
PT: # classe se comportar como os usuários esperam de uma sequência.
EN: # Mutable defaults need `field(default_factory=...)` — the dataclass
PT: # Padrões mutáveis exigem `field(default_factory=...)`: o mecanismo da dataclass
EN: # machinery enforces this, catching the notebook-04 trap at class-definition
PT: # impõe isso, detectando a armadilha do notebook 04 na definição da classe,
EN: # time rather than letting it bite you at runtime.
PT: # em vez de deixar que ela cause problemas durante a execução.
EN: # Proof that the mutable-default trap is handled:
PT: # Prova de que a armadilha do padrão mutável foi tratada:
EN: # goes through the setter below
PT: # passa pelo setter abaixo
EN: # float-safe comparison, per notebook 02
PT: # comparação segura para floats, como no notebook 02
EN: # log1p computes log(1 + x) accurately for small x, and tolerates zeros.
PT: # log1p calcula log(1 + x) com precisão para x pequeno e tolera zeros.
EN: # scikit-learn requires __init__ to do nothing but store its arguments
PT: # O scikit-learn exige que __init__ apenas guarde os argumentos
EN: # unchanged, under the same names. `get_params`/`clone` depend on it.
PT: # sem alteração, com os mesmos nomes. `get_params`/`clone` dependem disso.
EN: # returning self is what allows `.fit(X).transform(X)`
PT: # devolver self é o que permite `.fit(X).transform(X)`
EN: # Guard against being used before fitting — a clear error beats a
PT: # Protege contra o uso antes do ajuste: um erro claro é melhor que um
EN: # confusing AttributeError deep inside numpy.
PT: # AttributeError confuso nas profundezas do numpy.
EN: # --- Try it out -------------------------------------------------------------
PT: # --- Experimente ------------------------------------------------------------
EN: # inject extreme outliers
PT: # injeta outliers extremos
EN: # free, from TransformerMixin
PT: # sem custo, vindo de TransformerMixin
EN: # Our custom class sits alongside built-in ones as an equal citizen.
PT: # A nossa classe personalizada fica ao lado das embutidas, em pé de igualdade.
EN: # get_params works because __init__ only stored its arguments — this is what
PT: # get_params funciona porque __init__ só guardou os argumentos: é isso que
EN: # lets GridSearchCV tune our custom step (Module 13).
PT: # permite ao GridSearchCV ajustar a nossa etapa personalizada (Módulo 13).
