<!-- en:6d88c558b9 -->
# 01 · Introdução ao Curso e Configuração

**Módulo 01 — Fundamentos de Python para Ciência de Dados**

---

## O que você vai aprender

1. O que uma pessoa cientista de dados realmente faz no dia a dia, e quais partes do trabalho este curso cobre.
2. O ciclo de vida da ciência de dados, e por que ele é um ciclo, não uma linha reta.
3. Como configurar um ambiente Python **reproduzível**, e por que isso é um requisito profissional, não um luxo.
4. Como trabalhar no Jupyter sem cair nas suas duas armadilhas clássicas.
5. Uma primeira amostra de ponta a ponta: entram dados, sai uma conclusão.

**Tempo estimado:** 60 minutos

<!-- en:10a08e47a5 -->
## 1. O que é, de verdade, ser cientista de dados?

O título cobre muita coisa, mas em uma empresa de médio ou grande porte o trabalho quase sempre se divide em quatro atividades recorrentes:

| Atividade | Como ela é na prática | Parcela do tempo |
| :--- | :--- | :-- |
| **Enquadramento** | Transformar um pedido vago do negócio ("dá para reduzir o churn?") em uma pergunta mensurável, com uma métrica de sucesso definida | ~15% |
| **Trabalho com dados** | Encontrar, juntar, limpar e validar dados; descobrir que uma coluna mudou de significado em março | ~50% |
| **Modelagem e análise** | Estatística, experimentos, machine learning: a parte que todo mundo ensina, e a menor fatia | ~20% |
| **Entrega** | Colocar o modelo em produção, escrever o relatório, convencer um stakeholder, monitorar o que foi entregue | ~15% |

> [!NOTE]
> Esses 50% não indicam uma equipe mal administrada: esse é o trabalho. Um curso que só ensina modelagem prepara você para 20% da sua rotina. Este cobre as quatro atividades.

<!-- en:a560578dee -->
### As três funções de que você vai ouvir falar

Os títulos variam de empresa para empresa, mas as responsabilidades se agrupam em três formatos:

- **Analista de Dados**: descreve o que *aconteceu*. SQL, dashboards, estatística descritiva.
- **Cientista de Dados**: explica *por que* aconteceu e prevê o que *vai* acontecer. Estatística, experimentação, ML.
- **Engenheiro de ML**: faz a previsão *rodar de forma confiável* para milhões de usuários. Engenharia, pipelines, serving, monitoramento.

As fronteiras são nebulosas e estão mudando. O mercado espera cada vez mais que quem trabalha com ciência de dados tenha credibilidade nas três colunas, e é exatamente isso que "full stack" significa no título deste curso.

<!-- en:d3801d1768 -->
## 2. O ciclo de vida da ciência de dados

Quase todo projeto segue o mesmo ciclo. O mais importante a guardar: **as setas também apontam para trás.**

Durante a modelagem (etapa 6), você vai descobrir que um atributo está vazando informação e vai voltar à etapa 5. Vai apresentar resultados (etapa 8) e descobrir que a pergunta do negócio era sutilmente diferente, o que leva você de volta à etapa 1. Isso é normal e esperado: não significa que você planejou mal.

![O ciclo de vida da ciência de dados](../../../assets/images/ds_lifecycle.png)

<!-- en:8746e3cd1c -->
### As etapas, concretamente

| Etapa | Pergunta típica | Onde você vai aprender |
| :--- | :--- | :--- |
| 1. Entendimento do negócio | "Que decisão isto vai mudar?" | Módulos 09, 24 |
| 2. Coleta de dados | "Onde isto está, e posso confiar?" | Módulos 05, 20 |
| 3. Limpeza e preparação | "O que os valores ausentes *significam*?" | Módulos 03, 10 |
| 4. Análise exploratória de dados | "Como os dados realmente são?" | Módulos 04, 06 |
| 5. Engenharia de atributos | "Que representação torna o sinal aprendível?" | Módulo 10 |
| 6. Modelagem | "Qual modelo, e com quais hiperparâmetros?" | Módulos 11–19 |
| 7. Avaliação | "Isto é melhor que a abordagem atual? Por quanto, e com quanta certeza?" | Módulos 08, 13 |
| 8. Implantação e monitoramento | "Ainda está funcionando três meses depois?" | Módulos 21–23 |

<!-- en:632ea57468 -->
## 3. Reprodutibilidade: o ambiente

### Por que ambientes virtuais são inegociáveis

Imagine dois projetos no seu notebook. O projeto A foi construído com `pandas 1.5`; o projeto B precisa de `pandas 3.0`, em que vários comportamentos mudaram. Com uma única instalação de Python no sistema, você só consegue atender a um deles.

Um **ambiente virtual** é um diretório isolado com seu próprio interpretador Python e seus próprios pacotes instalados. Um por projeto. Eles são baratos, descartáveis e eliminam uma categoria inteira de bugs do tipo "na minha máquina funciona".

```text
full-stack-data-science/
└── .venv/                     ← o ambiente isolado deste projeto
    ├── bin/python             ← seu próprio interpretador
    └── lib/python3.12/site-packages/
                               ← seus próprios pacotes, invisíveis para outros projetos
```

> [!WARNING]
> Nunca use `pip install` no Python do sistema. No macOS e no Linux, você pode quebrar ferramentas do sistema operacional que dependem dele.

<!-- en:701ab6e7e1 -->
### Os três níveis de reprodutibilidade

Equipes profissionais fazem essa distinção, e você também deveria:

| Nível | Mecanismo | O que garante |
| :--- | :--- | :--- |
| **1. Solto** | `pandas` | "Alguma versão do pandas." Não reproduz quase nada. |
| **2. Restrito** | `pandas>=2.2,<3` | Reproduz a *API*. É o que o `pyproject.toml` declara. |
| **3. Fixado** | `pandas==3.0.5` | Reproduz *exatamente os mesmos bits*. É o que o `requirements-lock.txt` registra. |

A convenção: pessoas editam o nível 2, máquinas geram o nível 3, e a produção é implantada a partir do nível 3.

<!-- en:b4abd8c5d4 -->
### Verifique o seu próprio ambiente

Execute a célula abaixo. Ela deve mostrar o interpretador dentro de `.venv`. Se mostrar um caminho do sistema, como `/usr/bin/python3`, o seu notebook está conectado ao kernel errado. Corrija isso no seletor de kernel, no canto superior direito, antes de continuar, ou nada mais neste curso será importado corretamente.

<!-- en:f51500d196 -->
### Como tornar importável o pacote de utilitários do curso

Este repositório traz um pequeno pacote, `dstoolkit`, em `src/`. Ele foi instalado no ambiente em modo *editável* (`pip install -e .`), o que significa que a importação abaixo reflete suas edições locais na hora, sem precisar reinstalar.

Instalações editáveis são a forma padrão de projetos profissionais permitirem que notebooks e testes importem código compartilhado sem gambiarras frágeis com `sys.path`.

<!-- en:621fbcb0cb -->
## 4. Como trabalhar no Jupyter sem dar um tiro no próprio pé

### O modelo mental

Um notebook **não** é um script. Ele é um processo Python de longa duração (o *kernel*) mais um documento de células que você pode enviar a esse processo **em qualquer ordem**. Essa flexibilidade é o que torna os notebooks ótimos para exploração, e também é a origem dos dois bugs clássicos de notebook.

<!-- en:7116d01c61 -->
### Armadilha 1: estado oculto por execução fora de ordem

O contador de execução `[n]`, à esquerda de cada célula, indica a ordem em que as células *realmente rodaram*, que pode não ter nada a ver com a ordem delas na página.

```python
# Célula A, executada em 1º lugar
df = load_data()

# Célula B, executada em 3º lugar
df = df[df["revenue"] > 0]      # rodou duas vezes → filtrou duas vezes

# Célula C, executada em 2º lugar
print(len(df))                  # imprimiu um número que já não é válido
```

**A solução**, um hábito que vale criar desde já: antes de confiar em um resultado ou compartilhar um notebook, execute **Kernel → Restart Kernel and Run All Cells**. Se falhar, o seu notebook estava mentindo para você.

<!-- en:937e635d54 -->
### Armadilha 2: notebooks não substituem funções

Uma célula de 200 linhas não pode ser testada, reutilizada nem revisada. O padrão profissional é um **ciclo de promoção**:

1. Explore de forma interativa em uma célula.
2. Quando funcionar, coloque o código em uma função.
3. Quando precisar dela duas vezes, mova a função para `src/dstoolkit/`.
4. Escreva um teste para ela.
5. Importe-a de volta no notebook.

Você vai praticar esse ciclo no notebook 08 deste módulo.

<!-- en:10bfb4f562 -->
### Comandos mágicos que vale conhecer

O Jupyter adiciona as "mágicas": comandos com o prefixo `%` (valem para uma linha) ou `%%` (valem para a célula inteira). Estes concentram a maior parte do valor.

**`%timeit`** executa uma instrução muitas vezes e informa uma média robusta. Use-o sempre que estiver escolhendo entre duas implementações. O Módulo 03 o usa para mostrar por que sets ganham de listas em testes de pertencimento.

<!-- en:82957bd694 -->
**`%%time`** mede a célula inteira exatamente uma vez. Use-o em operações lentas e pontuais, em que executar mil vezes seria absurdo. Ele precisa ser a primeira linha da célula.

<!-- en:1f94f28034 -->
Outras mágicas que você vai usar o tempo todo:

| Mágica | O que faz |
| :--- | :--- |
| `%who` / `%whos` | Lista as variáveis definidas no kernel, um jeito rápido de encontrar estado oculto |
| `%debug` | Abre um depurador post-mortem depois de uma exceção |
| `%load_ext autoreload` + `%autoreload 2` | Reimporta automaticamente seus módulos de `src/` depois que você os edita |
| `!command` | Executa um comando do shell, por exemplo `!ls data/raw` |

> [!TIP]
> Coloque `%load_ext autoreload` e `%autoreload 2` no início de qualquer notebook em que você esteja editando código em `src/`. Sem isso, você precisa reiniciar o kernel para ver as mudanças.

<!-- en:05bdef7d37 -->
## 5. Uma primeira amostra de ponta a ponta

Chega de configuração. Vamos fazer uma versão em miniatura de todo o ciclo de vida em uma só tela: gerar dados, explorá-los e responder a uma pergunta com um gráfico.

O cenário: uma equipe de e-commerce quer saber se **clientes conquistados por anúncios pagos gastam menos que clientes conquistados de forma orgânica**, uma pergunta que aparece em praticamente toda empresa voltada ao consumidor.

Tudo aqui será explicado direito mais tarde. Por enquanto, só observe o formato do fluxo de trabalho.

<!-- en:592d866e69 -->
### Etapa 4: análise exploratória de dados

A primeira pergunta sobre qualquer conjunto de dados é sempre a mesma: *o que tem nele, e alguma coisa está obviamente quebrada?*

<!-- en:3a4fe53581 -->
> [!WARNING]
> Repare que a **média fica bem acima da mediana** nos dois grupos. Essa é a marca de uma distribuição assimétrica à direita, e significa que a média sozinha daria uma imagem enganosa de um cliente "típico". Relatar apenas uma média aqui é um dos erros analíticos mais comuns no mercado. O Módulo 06 mostra quando usar cada medida de tendência central.

<!-- en:a1f58e64ef -->
### Etapa 7: o que podemos realmente concluir?

O gráfico mostra uma diferença. A pergunta seguinte de um profissional vem de imediato: *"essa diferença é real, ou pode ser só ruído?"*

Essa pergunta, a diferença entre uma observação e uma **afirmação defensável**, é o assunto inteiro dos Módulos 07 e 08. Aqui vai uma prévia da resposta:

<!-- en:1b26ebac9a -->
> [!NOTE]
> Um p-valor pequeno diz que a diferença observada é improvável supondo que os dois canais gastem exatamente igual. Ele **não** diz que os anúncios pagos *causam* um gasto menor: os canais diferem em muitos aspectos ao mesmo tempo. Separar correlação de causalidade é o assunto do Módulo 09.

Esse é o ciclo inteiro em miniatura: uma pergunta, alguns dados, uma olhada no formato deles, uma comparação quantificada e uma declaração honesta do que isso prova e do que não prova.

<!-- en:a27ec53bb8 -->
## 6. Como este repositório está organizado

```text
full-stack-data-science/
├── pyproject.toml              ← declaração de dependências (nível 2)
├── requirements-lock.txt       ← versões exatas (nível 3)
├── assets/images/              ← diagramas gerados
├── data/
│   ├── raw/                    ← entradas imutáveis; NUNCA edite à mão
│   └── processed/              ← saídas dos seus notebooks
├── modules/01-python-foundations/
│   ├── notebooks/              ← as lições
│   ├── scripts/                ← versões .py em texto puro (diffs legíveis)
│   └── exercises/
├── src/dstoolkit/              ← código compartilhado, testado e importável
└── tests/                      ← testes com pytest para src/
```

### A única regra sobre `data/raw/`

**Dados brutos são imutáveis.** Você nunca os edita, nunca os sobrescreve, nunca "só conserta aquela linha ruim" no Excel. Toda transformação é escrita como código que lê de `raw/` e grava em `processed/`.

O motivo é a reprodutibilidade: se a limpeza está no código, qualquer pessoa pode executá-la de novo e chegar aos seus números. Se está em uma edição manual não documentada, os seus resultados nunca poderão ser reproduzidos, nem por você, daqui a seis meses.

<!-- en:b25593c30d -->
## 7. Exercícios

> Faça estes exercícios antes de passar para o notebook 02. As soluções estão em [`exercises/`](../exercises/).

### Exercício 1.1: auditoria do ambiente

Escreva uma célula que imprima:
- a versão do Python, como uma tupla `(major, minor, micro)`;
- se você está dentro de um ambiente virtual;
- as versões instaladas de `numpy`, `pandas` e `matplotlib`.

*Dica:* todo pacote expõe um atributo `__version__`.

### Exercício 1.2: reproduza a análise com outra seed

Execute de novo a análise de clientes com `seed=7` em vez de `seed=42`.

- As médias dos grupos mudam? Por quanto?
- A *conclusão* muda?
- O que a sua resposta diz sobre o quanto confiar em uma única amostra?

### Exercício 1.3: quebre o notebook de propósito

1. Execute todas as células em ordem.
2. Volte para cima e execute de novo a célula que cria `customers`, mudando `N_CUSTOMERS` para `50`.
3. Sem executar mais nada, olhe a variável `summary`.

Explique em uma frase o que acabou de acontecer e diga qual comando de menu evita esse tipo de bug.

### Exercício 1.4: mapeamento do ciclo de vida

Escolha qualquer produto de dados que você usa (um feed de recomendações, um alerta de fraude, uma previsão de entrega). Escreva uma frase para cada uma das oito etapas do ciclo de vida, descrevendo o que a equipe por trás dele provavelmente precisou fazer.

<!-- en:2a0597d2ad -->
## Principais lições

- Ciência de dados é **50% trabalho com dados**. A modelagem é a ponta pequena e visível.
- O ciclo de vida é um **ciclo**; voltar atrás é o processo funcionando, não falhando.
- **Ambientes virtuais são por projeto e inegociáveis.** Declare restrições soltas para as pessoas e gere versões exatas para as máquinas.
- Um notebook é um *processo de longa duração*, não um script. Use **Restart & Run All** antes de confiar em um resultado ou compartilhá-lo.
- Promova o código que funciona para fora dos notebooks, para um pacote testado.
- **Dados brutos são imutáveis.** Toda transformação é código.

---

**Próximo:** [02 · Fundamentos de Python](02_python_fundamentals.ipynb): o modelo de objetos, e por que `b = a` é uma das linhas mais mal compreendidas de Python.

<!-- comments -->
EN: # `sys.executable` is the absolute path of the interpreter running this notebook.
PT: # `sys.executable` é o caminho absoluto do interpretador que roda este notebook.
EN: # This is the ground truth — far more reliable than checking `which python` in a
PT: # Esta é a fonte da verdade, muito mais confiável que conferir `which python` em um
EN: # terminal, which tells you about your *shell*, not about this kernel.
PT: # terminal, que informa sobre o seu *shell*, não sobre este kernel.
EN: # A virtual environment sets `sys.prefix` to its own root while leaving
PT: # Um ambiente virtual aponta `sys.prefix` para a própria raiz e deixa
EN: # `sys.base_prefix` pointing at the original interpreter. When they differ,
PT: # `sys.base_prefix` apontando para o interpretador original. Quando são diferentes,
EN: # we are inside a venv.
PT: # estamos dentro de um venv.
EN: # Apply the course-wide matplotlib style. One call, and every figure in every
PT: # Aplica o estilo de matplotlib do curso. Com uma chamada, todas as figuras de todos
EN: # notebook shares the same fonts, colours, and spacing.
PT: # os notebooks compartilham as mesmas fontes, cores e espaçamentos.
EN: # ---------------------------------------------------------------------------
PT: # ---------------------------------------------------------------------------
EN: # Stage 2: get some data. We simulate it so the notebook is fully self-contained
PT: # Etapa 2: obter dados. Simulamos para o notebook ser totalmente autossuficiente
EN: # and reproducible. A fixed seed means you see exactly the numbers we see.
PT: # e reproduzível. Uma seed fixa faz você ver exatamente os números que nós vemos.
EN: # Two acquisition channels with genuinely different spending behaviour.
PT: # Dois canais de aquisição com comportamentos de gasto realmente diferentes.
EN: # We use a lognormal distribution because customer spend is right-skewed:
PT: # Usamos uma distribuição lognormal porque o gasto de clientes é assimétrico à direita:
EN: # most people spend a little, a few spend a lot. Module 07 covers why.
PT: # a maioria gasta pouco, e poucos gastam muito. O Módulo 07 explica por quê.
EN: # higher centre
PT: # centro mais alto
EN: # lower centre
PT: # centro mais baixo
EN: # `groupby` splits the data by channel, then computes several statistics at once.
PT: # `groupby` separa os dados por canal e calcula várias estatísticas de uma vez.
EN: # Module 03 covers this pattern in depth — it is the workhorse of pandas.
PT: # O Módulo 03 aprofunda esse padrão, que é o burro de carga do pandas.
EN: # --- Left panel: the full distribution ---------------------------------------
PT: # --- Painel esquerdo: a distribuição completa ---------------------------------
EN: # Overlapping histograms show the *shape* of spending, not just its centre.
PT: # Histogramas sobrepostos mostram o *formato* do gasto, não só o seu centro.
EN: # --- Right panel: a like-for-like comparison ---------------------------------
PT: # --- Painel direito: uma comparação equivalente -------------------------------
EN: # A box plot compresses each distribution to its median and quartiles, which
PT: # Um box plot resume cada distribuição à sua mediana e aos seus quartis, o que
EN: # makes the comparison between groups much easier to read.
PT: # torna a comparação entre os grupos muito mais fácil de ler.
EN: # required for the boxes to accept a fill colour
PT: # necessário para que as caixas aceitem uma cor de preenchimento
EN: # Welch's t-test: compares two means without assuming the groups share a
PT: # Teste t de Welch: compara duas médias sem supor que os grupos tenham a mesma
EN: # variance. Module 08 explains the assumptions, and what to do when they fail.
PT: # variância. O Módulo 08 explica os pressupostos e o que fazer quando eles falham.
