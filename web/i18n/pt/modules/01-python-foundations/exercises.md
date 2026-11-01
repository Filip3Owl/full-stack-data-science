<!-- en:eb24e8a6ef -->
# Módulo 01 — Exercícios

Todo notebook deste módulo termina com uma seção de exercícios. Esta página reúne todos em um só lugar, para você trabalhar neles sem precisar rolar a tela, e indica onde estão as soluções comentadas.

> [!WARNING]
> **Tente cada exercício antes de abrir a solução.** Ler uma resposta pronta produz uma sensação convincente de entendimento, que desaparece no momento em que você encara uma célula em branco. Dez minutos empacado ensinam mais do que uma hora de leitura.

---

## Onde estão as soluções

Todas as soluções estão em um único notebook executado:

📓 **[09 · Soluções dos Exercícios](../notebooks/09_exercise_solutions.ipynb)**

Cada solução inclui o raciocínio e, quando mais de uma resposta é defensável, as alternativas com seus prós e contras.

---

## A lista de exercícios

### Notebook 01 — Introdução ao Curso e Configuração

| # | Exercício | Habilidade |
| :-- | :--- | :--- |
| 1.1 | Auditoria do ambiente | Inspeção do interpretador e dos pacotes |
| 1.2 | Reproduza a análise com outra seed | Variabilidade amostral |
| 1.3 | Quebre o notebook de propósito | Estado oculto, execução fora de ordem |
| 1.4 | Mapeamento do ciclo de vida | Enquadrar um produto de dados real |

### Notebook 02 — Fundamentos de Python

| # | Exercício | Habilidade |
| :-- | :--- | :--- |
| 2.1 | Preveja antes de executar | `+` e `+=`, imutabilidade de tuplas |
| 2.2 | Um bug de comparação de floats | IEEE 754, comparação com tolerância |
| 2.3 | Enriquecimento seguro | Funções que não modificam a entrada |
| 2.4 | Limpeza de texto | Expressões idiomáticas de normalização de strings |
| 2.5 | Auditoria de valores ausentes | `nan != nan` |

### Notebook 03 — Estruturas de Dados

| # | Exercício | Habilidade |
| :-- | :--- | :--- |
| 3.1 | Escolha a estrutura | Escolha guiada pela complexidade |
| 3.2 | Conserte a função lenta | De O(n×m) para O(n+m) com um set |
| 3.3 | Prática de comprehensions | Comprehensions de list / dict / set |
| 3.4 | Agrupe sem pandas | `defaultdict` |
| 3.5 | Ordenação por várias chaves | Tuplas como chave, direções de ordenação mistas |

### Notebook 04 — Funções e Programação Funcional

| # | Exercício | Habilidade |
| :-- | :--- | :--- |
| 4.1 | Conserte a assinatura | Argumentos somente por palavra-chave, type hints, docstrings |
| 4.2 | Encontre a armadilha | Argumentos padrão mutáveis |
| 4.3 | Escreva um decorador | `functools.wraps`, validação de argumentos |
| 4.4 | Pipeline com geradores | Avaliação preguiçosa, memória constante |
| 4.5 | Contador com closure | `nonlocal` |

### Notebook 05 — Programação Orientada a Objetos

| # | Exercício | Habilidade |
| :-- | :--- | :--- |
| 5.1 | Classe ou função? | Critério de design |
| 5.2 | Prática de métodos dunder | `__sub__`, `__mul__`, `__format__` |
| 5.3 | Conversão para dataclass | `@dataclass`, `__post_init__` |
| 5.4 | Construa um estimador | O contrato `fit`/`transform` do scikit-learn |
| 5.5 | Encontre o bug | Atributos de classe mutáveis compartilhados |

### Notebook 06 — Arquivos e Formatos de Dados

| # | Exercício | Habilidade |
| :-- | :--- | :--- |
| 6.1 | Manipulação de caminhos | `pathlib` |
| 6.2 | Diagnostique um CSV quebrado | Separadores, codificações, decimais, datas |
| 6.3 | JSON aninhado | `pd.json_normalize` |
| 6.4 | Comprove a seleção de colunas | Armazenamento colunar e por linhas |
| 6.5 | Gravação segura | Gerenciadores de contexto, gravações atômicas |

### Notebook 07 — Erros, Logging e Testes

| # | Exercício | Habilidade |
| :-- | :--- | :--- |
| 7.1 | Restrinja o except | Tratamento de exceções específicas |
| 7.2 | Hierarquia de exceções personalizada | Erros de domínio, possibilidade de nova tentativa |
| 7.3 | Instrumente uma função | Níveis de `logging` |
| 7.4 | Escreva os testes primeiro | Design guiado por testes, casos extremos |
| 7.5 | Capture um bug real | Explosão de linhas no merge, `validate=` |

### Notebook 08 — Projeto do Módulo

O projeto termina com seis extensões graduadas (de ⭐ a ⭐⭐⭐). Elas são abertas de propósito e não têm uma única resposta certa: são o mais perto que este módulo chega do trabalho real.

| # | Extensão | Dificuldade |
| :-- | :--- | :-- |
| 1 | Promova código para o `dstoolkit` | ⭐ |
| 2 | Processe a entrada em blocos | ⭐⭐ |
| 3 | Um relatório de linhas rejeitadas | ⭐⭐ |
| 4 | Idempotência e gravações atômicas | ⭐⭐ |
| 5 | Um decorador `@monitored` | ⭐⭐⭐ |
| 6 | Validação declarativa de esquema | ⭐⭐⭐ |

---

## Como conferir o seu próprio trabalho

Além de comparar com as soluções, dois hábitos valem para qualquer caso:

1. **Escreva um teste.** Se você não consegue dizer qual é a saída correta, ainda não entendeu o problema.
2. **Restart & Run All.** Se a sua solução só funciona quando as células rodam na ordem em que você clicou nelas, ela não funciona.
