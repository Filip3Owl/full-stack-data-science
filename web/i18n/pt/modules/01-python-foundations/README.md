<!-- en:0077af316c -->
# Módulo 01 — Fundamentos de Python para Ciência de Dados

> **Parte I — Fundamentos** · Tempo estimado: **8–10 horas** · Pré-requisitos: nenhum

Este módulo constrói a camada de Python sobre a qual todo o resto do curso se apoia. Ele deliberadamente *não* é um tutorial genérico de "aprenda Python": cada tópico foi escolhido porque aparece em trabalhos reais de ciência de dados, e os exemplos vêm de problemas com dados, não de exercícios de brinquedo.

<p align="center">
  <img src="../../assets/images/course_stack_map.png" alt="O mapa de habilidades de ciência de dados full-stack, com o Módulo 01 na camada de fundamentos" width="720">
</p>

---

## Por que este módulo existe

A maioria das pessoas que chega à ciência de dados já sabe escrever *algum* Python. O que as separa de um profissional raramente é a sintaxe, e sim o conjunto de hábitos abaixo. Pular este módulo é o motivo mais comum para alguém escrever, mais tarde, notebooks que produzem números errados sem nenhum aviso.

| O hábito | O que ele evita |
| :--- | :--- |
| Entender nomes e objetos | Modificar um DataFrame que você achava ter copiado |
| Conhecer a complexidade de cada coleção | Um loop `for` que leva 40 minutos em vez de 2 segundos |
| Comprehensions e geradores | Carregar um arquivo de 20 GB na memória de uma só vez |
| Escrever funções, não scripts | Lógica de análise copiada e colada que acaba divergindo |
| Tratar erros deliberadamente | Um pipeline que engole falhas e reporta lixo |
| Testar suas transformações | Levar um bug de engenharia de atributos para produção |

---

## Objetivos de aprendizagem

Ao final deste módulo, você será capaz de:

1. **Configurar e reproduzir** um ambiente Python profissional (ambiente virtual, kernel, versões fixadas).
2. **Explicar o modelo de objetos do Python** (nomes, referências, mutabilidade) e prever o comportamento de aliasing.
3. **Escolher a coleção embutida certa** para cada tarefa com base na complexidade de tempo.
4. **Escrever transformações idiomáticas** com comprehensions, geradores e as funções funcionais embutidas.
5. **Projetar funções reutilizáveis** com assinaturas sensatas, type hints e docstrings.
6. **Modelar dados com classes**, incluindo `dataclass` e o padrão de estimadores do scikit-learn.
7. **Ler e gravar** CSV, JSON e Parquet com segurança usando `pathlib` e gerenciadores de contexto.
8. **Tratar erros e registrar logs** de um jeito que torne diagnosticável um pipeline com falha.
9. **Testar transformações de dados** com `pytest`, para que as refatorações continuem seguras.

---

## Notebooks

Siga-os em ordem: cada um se apoia no anterior.

| # | Notebook | O que você vai aprender | Tempo |
| :-: | :--- | :--- | :-: |
| 01 | [Introdução ao Curso e Configuração](notebooks/01_course_introduction_and_setup.ipynb) | O ciclo de vida de ciência de dados, a função, as ferramentas, a reprodutibilidade do ambiente, o fluxo de trabalho em notebooks | 60 min |
| 02 | [Fundamentos de Python](notebooks/02_python_fundamentals.ipynb) | O modelo de objetos, tipos, armadilhas numéricas, strings, controle de fluxo, valores verdadeiros e falsos | 90 min |
| 03 | [Estruturas de Dados](notebooks/03_data_structures.ipynb) | list / tuple / dict / set, complexidade, `collections`, comprehensions | 90 min |
| 04 | [Funções e Programação Funcional](notebooks/04_functions_and_functional_programming.ipynb) | Assinaturas, escopo, closures, decoradores, geradores, `map`/`filter` | 90 min |
| 05 | [Programação Orientada a Objetos](notebooks/05_object_oriented_programming.ipynb) | Classes, métodos dunder, `dataclass`, herança e composição, o padrão de estimadores | 75 min |
| 06 | [Arquivos e Formatos de Dados](notebooks/06_files_and_data_formats.ipynb) | `pathlib`, gerenciadores de contexto, CSV / JSON / Parquet, codificações, comparação de formatos | 75 min |
| 07 | [Erros, Logging e Testes](notebooks/07_errors_logging_and_testing.ipynb) | Exceções, erros personalizados, logging, asserções, `pytest`, validação de dados | 90 min |
| 08 | [Projeto do Módulo](notebooks/08_module_project.ipynb) | Construa um pequeno pipeline de ETL e relatórios, com testes, usando tudo o que foi visto | 120 min |
| 09 | [Soluções dos Exercícios](notebooks/09_exercise_solutions.ipynb) | Respostas comentadas com o raciocínio — **abra só depois de tentar os exercícios** | — |

---

## Figuras principais

Os diagramas abaixo são usados nos notebooks. Eles são gerados por [`scripts/generate_figures.py`](../../scripts/generate_figures.py), e você pode gerá-los de novo a qualquer momento com `.venv/bin/python scripts/generate_figures.py`.

| Figura | Notebook |
| :--- | :--- |
| ![Ciclo de vida de ciência de dados](../../assets/images/ds_lifecycle.png) | 01 |
| ![Nomes e objetos](../../assets/images/names_and_objects.png) | 02 |
| ![Mutável e imutável](../../assets/images/mutable_vs_immutable.png) | 02 |
| ![Estruturas de dados](../../assets/images/data_structures_comparison.png) | 03 |
| ![Escopo LEGB](../../assets/images/legb_scope.png) | 04 |
| ![Hierarquia de exceções](../../assets/images/exception_hierarchy.png) | 07 |

---

## Exercícios

Cada notebook termina com exercícios. A [página de exercícios](exercises/README.md) reúne todos os 29, e as soluções comentadas estão no [notebook 09](notebooks/09_exercise_solutions.ipynb).

> **Faça os exercícios antes de ler as soluções.** Ler uma solução dá a sensação de aprender, mas não é aprender.

---

## Checklist de autoavaliação

Você está pronto para o Módulo 02 quando conseguir responder a tudo isto *sem consultar nada*:

- [ ] Por que `b = a` seguido de `b.append(1)` às vezes altera `a` e às vezes não?
- [ ] Por que `0.1 + 0.2 == 0.3` é `False`, e o que você deve escrever no lugar?
- [ ] Por que `x in my_set` é mais rápido que `x in my_list`, e por quanto?
- [ ] Qual é a diferença entre uma list comprehension e uma expressão geradora?
- [ ] O que um argumento padrão mutável (`def f(x=[])`) faz, e por que é um bug?
- [ ] Quando você usaria `@dataclass` em vez de escrever `__init__` à mão?
- [ ] Por que `except Exception:` costuma estar errado?
- [ ] Por que um pipeline de dados deve usar `logging` em vez de `print`?
- [ ] O que torna o Parquet uma escolha melhor que o CSV para um conjunto de dados de 5 GB?

---

## Leitura complementar

| Recurso | Por que vale o seu tempo |
| :--- | :--- |
| [Fluent Python, 2ª ed.](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056348/) (Ramalho) | O livro definitivo sobre o modelo de objetos do Python; os capítulos 1 a 3 correspondem diretamente aos notebooks 02 e 03. |
| [PEP 8](https://peps.python.org/pep-0008/) | O guia de estilo que todo código Python presume que você leu. |
| [PEP 484](https://peps.python.org/pep-0484/) | Type hints, usados em todo o `src/dstoolkit/`. |
| [Documentação de `collections`](https://docs.python.org/3/library/collections.html) | Só `Counter` e `defaultdict` já vão poupar centenas de linhas. |
| [Wiki TimeComplexity](https://wiki.python.org/moin/TimeComplexity) | A tabela de complexidade de cada operação embutida. |

---

**Próximo:** Módulo 02 — NumPy e Computação Numérica.
