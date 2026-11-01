<!-- en:32d3974171 -->
# Full-Stack Data Science com Python

Um curso completo e prático que leva você da própria linguagem Python até **modelos de machine learning rodando em produção**. Cada conceito é ensinado com código real e executável em notebooks Jupyter: nada de pseudocódigo nem de exemplos de brinquedo que quebram quando você os executa.

> **Para quem é:** qualquer pessoa que queira atuar como Cientista de Dados ou Engenheira de ML em uma empresa de médio ou grande porte. O currículo espelha o stack que essas equipes realmente usam no dia a dia.

---

## Filosofia

Três regras definem cada notebook deste repositório:

| Regra | O que significa na prática |
| :--- | :--- |
| **Execute, não só leia** | Cada notebook é publicado *com as saídas já executadas*, então dá para ler como um livro, mas você aprende executando e quebrando o código. |
| **Explique o "porquê"** | Os comentários do código explicam a intenção, não a sintaxe. As células de texto explicam o raciocínio estatístico ou de engenharia por trás de cada escolha. |
| **Feito para produção** | Preferimos os padrões que equipes reais usam (pipelines, type hints, testes, seeds reproduzíveis) aos atalhos que só funcionam em uma demonstração. |

---

## O stack de ciência de dados

Espera-se que quem trabalha com ciência de dados em uma grande empresa domine cinco camadas. Este curso cobre todas elas:

```
┌───────────────────────────────────────────────────────────────────┐
│  5. COMUNICAÇÃO        storytelling · dashboards · apresentações   │
│                        a stakeholders · relatórios de experimentos │
├───────────────────────────────────────────────────────────────────┤
│  4. PRODUÇÃO           MLflow · FastAPI · Docker · CI/CD ·         │
│     (MLOps)            monitoramento · drift · model registry      │
├───────────────────────────────────────────────────────────────────┤
│  3. MODELAGEM          scikit-learn · XGBoost · PyTorch ·          │
│                        statsmodels · inferência causal             │
├───────────────────────────────────────────────────────────────────┤
│  2. ANÁLISE            NumPy · pandas · matplotlib · seaborn ·     │
│                        probabilidade · testes de hipótese          │
├───────────────────────────────────────────────────────────────────┤
│  1. FUNDAMENTOS        Python · estruturas de dados · POO · SQL ·  │
│                        Git · testes · ambientes virtuais           │
└───────────────────────────────────────────────────────────────────┘
```

---

## Currículo

### Parte I: Fundamentos

| # | Módulo | Principais tópicos | Status |
| :-: | :--- | :--- | :-: |
| **01** | [**Fundamentos de Python para Ciência de Dados**](modules/01-python-foundations/) | Ambiente e ferramentas, tipos, controle de fluxo, estruturas de dados, comprehensions, funções, geradores, decoradores, POO, arquivos, erros, logging, testes, type hints | ✅ **Pronto** |
| **02** | [**NumPy e Computação Numérica**](modules/02-numpy-numerical-computing/) | ndarrays, views e cópias, dtypes e precisão, broadcasting, agregações e máscaras, álgebra linear, geradores aleatórios, strides e memória | ✅ **Pronto** |
| 03 | pandas e Manipulação de Dados | Series/DataFrame, indexação, joins, group-by, remodelagem, índices de tempo, encadeamento de métodos | 🔜 |
| 04 | Visualização de Dados | Gramática do matplotlib, gráficos estatísticos com seaborn, interatividade com plotly, escolha de gráficos, acessibilidade | 🔜 |
| 05 | SQL e Bancos de Dados | De SELECT a funções de janela, CTEs, planejamento de consultas, SQLAlchemy, DuckDB, padrões de data warehouse | 🔜 |

### Parte II: Estatística e Inferência

| # | Módulo | Principais tópicos | Status |
| :-: | :--- | :--- | :-: |
| 06 | Estatística Descritiva e EDA | Tendência central, dispersão, assimetria e curtose, correlação, outliers, fluxo de análise exploratória | 🔜 |
| 07 | Probabilidade e Distribuições | Variáveis aleatórias, distribuições discretas e contínuas, teorema central do limite, lei dos grandes números, simulação | 🔜 |
| 08 | Estatística Inferencial | Estimação, intervalos de confiança, testes de hipótese, poder, comparações múltiplas, bootstrap | 🔜 |
| 09 | Experimentação e Testes A/B | Desenho de experimentos, tamanho de amostra, testes sequenciais, CUPED, noções de inferência causal | 🔜 |

### Parte III: Machine Learning

| # | Módulo | Principais tópicos | Status |
| :-: | :--- | :--- | :-: |
| 10 | Limpeza de Dados e Engenharia de Atributos | Dados ausentes, codificação, escalonamento, vazamento de dados, pipelines, ColumnTransformer | 🔜 |
| 11 | Aprendizado Supervisionado: Regressão | Modelos lineares e regularizados, pressupostos, análise de resíduos, árvores de regressão | 🔜 |
| 12 | Aprendizado Supervisionado: Classificação | Regressão logística, SVM, kNN, naive Bayes, desbalanceamento de classes, limiares | 🔜 |
| 13 | Avaliação e Ajuste de Modelos | Validação cruzada, métricas, ROC/PR, calibração, busca em grade, aleatória e bayesiana | 🔜 |
| 14 | Ensembles e Gradient Boosting | Bagging, random forests, XGBoost, LightGBM, CatBoost, stacking | 🔜 |
| 15 | Aprendizado Não Supervisionado | k-means, clustering hierárquico, DBSCAN, PCA, t-SNE/UMAP, detecção de anomalias | 🔜 |
| 16 | Séries Temporais e Previsão | Decomposição, estacionariedade, ARIMA, SARIMAX, Prophet, backtesting | 🔜 |

### Parte IV: Deep Learning e Dados Não Estruturados

| # | Módulo | Principais tópicos | Status |
| :-: | :--- | :--- | :-: |
| 17 | Fundamentos de Deep Learning | Tensores, autograd, MLPs, otimizadores, regularização, loops de treino em PyTorch | 🔜 |
| 18 | Processamento de Linguagem Natural | Pré-processamento de texto, TF-IDF, embeddings, transformers, fine-tuning, padrões de uso de LLMs | 🔜 |
| 19 | Visão Computacional | CNNs, aumento de dados, transfer learning, noções de segmentação | 🔜 |

### Parte V: Escala e Produção

| # | Módulo | Principais tópicos | Status |
| :-: | :--- | :--- | :-: |
| 20 | Processamento de Big Data | DataFrames em PySpark, avaliação preguiçosa, particionamento, Dask, DuckDB em escala | 🔜 |
| 21 | MLOps I: Reprodutibilidade | Estrutura de projeto, configuração, rastreamento de experimentos com MLflow, model registry, conceitos de DVC | 🔜 |
| 22 | MLOps II: Implantação | Servir modelos com FastAPI, Docker, inferência em lote e online, CI/CD, implantação na nuvem | 🔜 |
| 23 | Monitoramento e Explicabilidade | Drift de dados e de conceito, monitoramento de desempenho, SHAP, equidade, governança de modelos | 🔜 |
| 24 | Projeto Final e Comunicação | Projeto de ponta a ponta, storytelling, resumos executivos, portfólio e preparação para entrevistas | 🔜 |

---

## Versão web

Todo o curso pode ser lido no navegador: lições com as saídas já executadas, busca, modo escuro e progresso salvo no próprio navegador. O site é gerado a partir dos notebooks e dos READMEs. Os notebooks continuam em inglês; as traduções para o português existem apenas no site.

```bash
uv pip install --python .venv/bin/python -e ".[web]"
.venv/bin/python scripts/build_site.py --serve     # http://127.0.0.1:8000
```

As células de código são somente leitura no site. Para executá-las, use o JupyterLab, como descrito abaixo.

---

## Primeiros passos

### Pré-requisitos

- **Python 3.12** (o curso fixa essa versão para garantir a reprodutibilidade)
- **git**
- Cerca de 5 GB de espaço livre em disco, se você instalar os extras de deep learning

### 1. Clone e entre no projeto

```bash
git clone <url-do-seu-fork> full-stack-data-science
cd full-stack-data-science
```

### 2. Crie o ambiente virtual

Escolha a ferramenta que você tiver à mão:

<details>
<summary><b>Opção A: uv (recomendada, a mais rápida)</b></summary>

[uv](https://docs.astral.sh/uv/) é um gerenciador de pacotes Python muito rápido.

```bash
# Instale o uv uma única vez (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Crie o ambiente e instale o stack principal
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```
</details>

<details>
<summary><b>Opção B: venv da biblioteca padrão</b></summary>

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[dev]"
```
</details>

### 3. Registre o kernel do Jupyter

Isso permite selecionar o ambiente de dentro do JupyterLab:

```bash
.venv/bin/python -m ipykernel install --user \
    --name dsc-py312 \
    --display-name "Python 3.12 (Data Science Course)"
```

### 4. Abra o JupyterLab

```bash
.venv/bin/jupyter lab
```

Depois, abra `modules/01-python-foundations/notebooks/` e comece pelo notebook `01`.

> **Selecione o kernel certo.** No canto superior direito de cada notebook, confira se o kernel é *Python 3.12 (Data Science Course)*. Se for outro, clique nele e troque; caso contrário, as importações vão falhar.

### 5. (Opcional) Instale os extras dos próximos módulos

```bash
uv pip install --python .venv/bin/python -e ".[advanced-ml]"     # Módulos 14-16
uv pip install --python .venv/bin/python -e ".[deep-learning]"   # Módulos 17-19
uv pip install --python .venv/bin/python -e ".[big-data]"        # Módulo 20
uv pip install --python .venv/bin/python -e ".[mlops]"           # Módulos 05, 21-23
```

### Verifique sua instalação

```bash
.venv/bin/python scripts/check_environment.py
```

Cada pacote principal deve aparecer com uma marca de confirmação verde.

---

## Estrutura do repositório

```
full-stack-data-science/
├── README.md                   ← descrição do curso
├── pyproject.toml              ← dependências e configuração das ferramentas
├── requirements-lock.txt       ← versões exatas fixadas (reprodutibilidade)
├── .gitignore
│
├── assets/
│   └── images/                 ← diagramas gerados, usados nos notebooks
│
├── data/
│   ├── raw/                    ← entradas imutáveis (nunca editadas à mão)
│   ├── processed/              ← saídas limpas produzidas pelos notebooks
│   └── external/               ← downloads de terceiros
│
├── modules/
│   ├── 01-python-foundations/
│   │   ├── README.md           ← programa, objetivos e checklist do módulo
│   │   ├── notebooks/          ← as lições (.ipynb, executados)
│   │   ├── scripts/            ← versões .py em texto puro de cada notebook
│   │   └── exercises/          ← exercícios práticos e soluções
│   └── 02-numpy-numerical-computing/
│       └── …                   ← mesma estrutura, um diretório por módulo
│
├── web/                        ← modelos, estilos e traduções da versão web
├── src/dstoolkit/              ← pacote de utilitários importável em todo o curso
├── scripts/                    ← utilitários do projeto
└── tests/                      ← testes com pytest para src/
```

**Por que `src/dstoolkit/`?** Projetos reais não deixam a lógica presa dentro de notebooks. À medida que o curso avança, o código que se mostra útil é promovido de um notebook para esse pacote, ganha um teste e depois é importado de volta. Você vai praticar esse ciclo de promoção desde o Módulo 01.

---

## Como estudar

Todos os módulos seguem o mesmo ritmo:

1. **Leia a visão geral do módulo**: objetivos de aprendizagem e tempo estimado.
2. **Siga os notebooks em ordem**: cada um se apoia no anterior.
3. **Execute cada célula você mesmo.** Depois, mude um valor e preveja a saída *antes* de executar de novo. É aí que o aprendizado realmente acontece.
4. **Faça os exercícios** antes de abrir as soluções.
5. **Conclua o projeto do módulo**: um problema integrado que usa tudo o que foi visto.

**Ritmo sugerido:** um módulo por semana, com cerca de 6 a 8 horas semanais. O curso completo leva aproximadamente 6 meses em tempo parcial.

---

## Convenções

Estas convenções valem para todos os notebooks, para que você sempre saiba o que está vendo.

### Blocos de destaque

> [!NOTE]
> Contexto extra que vale conhecer, mas que não é necessário para seguir em frente.

> [!TIP]
> Um atalho, uma expressão idiomática ou um hábito usado por profissionais.

> [!WARNING]
> Um erro comum que produz resultados errados sem nenhum aviso.

### Estilo de código

- **PEP 8** verificada pelo `ruff` e formatação com `black` (linhas de 88 caracteres).
- **Type hints** em todas as funções de `src/`.
- **Docstrings** no estilo NumPy, o padrão no ecossistema científico de Python.
- **Aleatoriedade com seed**: toda célula estocástica define uma seed para que a sua saída seja igual à nossa.

### Verificações de qualidade

```bash
.venv/bin/ruff check .        # lint
.venv/bin/black --check .     # formatação
.venv/bin/pytest              # testes de src/
```

---

## Licença

MIT: use, faça fork, ensine com ele.
