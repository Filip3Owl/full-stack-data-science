<!-- en:14aa149345 -->
# 07 · Aleatoriedade e Simulação

**Módulo 02 — NumPy e Computação Numérica**

---

## O que você vai aprender

1. A API moderna de `Generator`, e por que o antigo `np.random.seed` é um risco.
2. Uma reprodutibilidade que resiste ao paralelismo, com `SeedSequence` e `spawn`.
3. As distribuições das quais você vai realmente sortear, e quando cada uma se aplica.
4. Amostragem: com e sem reposição, ponderada, permutações.
5. **Monte Carlo**: como responder a perguntas que não têm fórmula fechada.
6. **O bootstrap**: intervalos de confiança sem nenhum pressuposto sobre a distribuição.

**Tempo estimado:** 75 minutos

> Este notebook é a ponte para a metade estatística do curso. Tudo nos Módulos 07 a 09 usa estas ferramentas ou é validado com elas.

<!-- en:743a77fa93 -->
## 1. A API de `Generator`

O NumPy tem dois sistemas de números aleatórios. Use o mais novo.

<!-- en:7ecab7e202 -->
### Por que o estado global é um problema

`np.random.seed` define um estado que *todas* as bibliotecas compartilham. Se o scikit-learn, uma chamada de gráfico ou uma função auxiliar de um colega sortear um número entre a sua seed e o seu sorteio, o seu resultado "reproduzível" muda sem aviso.

<!-- en:8514f75631 -->
> [!TIP]
> **Crie um `Generator` no início do script ou do notebook e passe-o adiante.** Funções que precisam de aleatoriedade devem receber `rng` como parâmetro em vez de recorrer a um global. É o mesmo princípio de "nenhum estado oculto" do conselho do Módulo 01 sobre funções puras.

<!-- en:3b2c64f261 -->
### Fluxos independentes para trabalho em paralelo

Dividir um único gerador entre processos dando a cada um a seed `seed + i` é um erro real: seeds próximas podem produzir fluxos correlacionados. O `spawn` de `SeedSequence` cria filhos comprovadamente independentes.

<!-- en:cf96fcd5e2 -->
## 2. As distribuições que você vai usar

Cada uma modela um tipo diferente de processo. Escolher a certa é uma decisão de modelagem, não uma decisão técnica.

<!-- en:e06d6bafac -->
### Qual usar, e quando

| Distribuição | Modela | Onde você vai encontrá-la |
| :--- | :--- | :--- |
| **Uniforme** | "qualquer valor em um intervalo, com a mesma probabilidade" | inicialização aleatória, embaralhamento |
| **Normal** | somas de muitos efeitos pequenos (TCL) | erro de medição, resíduos |
| **Lognormal** | quantidades que são *produtos* de efeitos | renda, gasto, duração de sessão |
| **Exponencial** | tempo de espera entre eventos | churn, tempo até a falha |
| **Binomial** | sucessos em *n* tentativas independentes | conversões entre visitantes |
| **Poisson** | contagem de eventos em uma janela fixa | chegadas por hora, defeitos por lote |

> [!NOTE]
> O gasto de clientes é **lognormal**, não normal, e por isso a média ficou bem acima da mediana lá na primeira análise do Módulo 01. Reconhecer o processo gerador diz qual estatística-resumo é honesta.

<!-- en:ba98debbd3 -->
## 3. Amostragem

<!-- en:411bb7ba87 -->
## 4. Monte Carlo

Quando uma quantidade é difícil de deduzir analiticamente, **simule**. Sorteie muitos resultados aleatórios e tire a média. A estimativa converge a uma taxa de `1/√n`, o que é lento, mas funciona em problemas para os quais não existe fórmula.

<!-- en:2879eb116a -->
### Uma pergunta sem fórmula fácil

*"Uma página de checkout converte 12%. Se 40 pessoas a visitarem, qual é a probabilidade de pelo menos 8 converterem?"* Esta pergunta tem, sim, uma forma fechada, o que a torna um bom jeito de conferir se a simulação dá a resposta certa.

<!-- en:78721e40ef -->
## 5. O bootstrap

O bootstrap estima a incerteza de *qualquer* estatística reamostrando, com reposição, os dados que você já tem. Ele não precisa de **nenhum pressuposto** sobre a distribuição subjacente, e é por isso que funciona para medianas, razões e correlações, em que as fórmulas clássicas são difíceis ou nem existem.

<!-- en:b6babab490 -->
> [!NOTE]
> Repare que a distribuição bootstrap da **média** é próxima da normal, mesmo com dados fortemente assimétricos. É o Teorema Central do Limite aparecendo por conta própria, e o Módulo 07 o desenvolve direito.

<!-- en:4ad9841006 -->
## 6. Exercícios

### Exercício 7.1: quebre e conserte a reprodutibilidade

Escreva uma função que sorteie números aleatórios com a API **antiga** e mostre que chamar uma função não relacionada entre a seed e o sorteio altera o resultado. Depois, reescreva com um `Generator` injetado e mostre que isso não pode acontecer.

### Exercício 7.2: escolha a distribuição

Para cada caso, diga a distribuição e justifique em uma linha:

1. Número de chamados de suporte que chegam por hora.
2. Tempo entre dois chamados consecutivos.
3. Se cada um de 500 e-mails é aberto.
4. Receita total por cliente ao longo de um ano.
5. Erro de medição de um sensor bem calibrado.

### Exercício 7.3: o problema do aniversário

Estime por simulação a probabilidade de pelo menos duas pessoas em um grupo de `k` fazerem aniversário no mesmo dia, para `k` de 2 a 60. Faça um gráfico e encontre o menor `k` em que a probabilidade passa de 50%. Compare com a fórmula exata.

### Exercício 7.4: bootstrap de uma diferença

Dois grupos, `a` e `b`, de gastos de clientes. Calcule por bootstrap um intervalo de confiança de 95% para a **diferença de medianas**. Ele contém o zero? Explique o que isso permite e o que não permite afirmar.

### Exercício 7.5: tamanho de amostra por simulação

Uma página converte 10%. Você quer detectar um aumento para 11% com 80% de poder e α = 0.05. Estime o tamanho de amostra necessário por braço **por simulação**: para uma grade de tamanhos de amostra, simule 2000 testes A/B e registre com que frequência o teste atinge significância. Esse é o núcleo do Módulo 09.

<!-- en:87925db4f5 -->
## Principais lições

- Use **`np.random.default_rng(seed)`**. O antigo `np.random.seed` compartilha um único estado global, e sua reprodutibilidade é frágil.
- **Injete o gerador** nas funções em vez de recorrer a um global: é o mesmo princípio das funções puras do Módulo 01.
- Em trabalho paralelo, use **`SeedSequence.spawn`**, nunca `seed + i`.
- Associe a **distribuição ao processo gerador**: contagens → Poisson, tempos de espera → Exponencial, quantidades multiplicativas → Lognormal.
- **Monte Carlo** responde a perguntas sem forma fechada; seu erro diminui como **1/√n**, então 100× mais esforço rendem 10× mais exatidão.
- **O bootstrap** dá um intervalo de confiança para qualquer estatística sem pressuposto de distribuição: reamostre com reposição e tome os percentis.
- Vetorize a reamostragem (`rng.integers(0, n, size=(B, n))`) em vez de usar um loop.

---

**Próximo:** [08 · Desempenho e Memória](08_performance_and_memory.ipynb): strides, contiguidade, operações in-place e como saber quando o NumPy chegou ao limite.

<!-- comments -->
EN: # ✘ Legacy: one hidden global state shared by every library in the process.
PT: # ✘ Antigo: um estado global oculto, compartilhado por todas as bibliotecas do processo.
EN: # ✔ Modern: an explicit object you own and pass around.
PT: # ✔ Moderno: um objeto explícito que é seu e que você passa adiante.
EN: # someone else drew numbers in between
PT: # outra parte do código sorteou números no meio
EN: # The Generator has no such problem: its state is private to the object.
PT: # O Generator não tem esse problema: seu estado é privado do objeto.
EN: # cannot touch our generator
PT: # não consegue tocar no nosso gerador
EN: # Re-spawning from the same parent reproduces them exactly.
PT: # Criar de novo a partir do mesmo pai os reproduz exatamente.
EN: # Permutation vs shuffle: one returns a new array, the other works in place.
PT: # Permutation e shuffle: um devolve um novo array, o outro trabalha no lugar.
EN: # in place, returns None
PT: # no lugar, devolve None
EN: # A train/test split, built from a permutation. This is what scikit-learn does.
PT: # Uma divisão treino/teste construída a partir de uma permutação. É o que o scikit-learn faz.
EN: # --- Left: what the simulation is doing -----------------------------------
PT: # --- Esquerda: o que a simulação está fazendo -------------------------------
EN: # --- Right: convergence, with the 1/√n envelope ---------------------------
PT: # --- Direita: convergência, com o envelope de 1/√n --------------------------
EN: # The theoretical ±1/√n envelope, scaled by the standard deviation of the estimator.
PT: # O envelope teórico de ±1/√n, escalado pelo desvio padrão do estimador.
EN: # Draw all resamples at once: shape (n_resamples, n). Vectorised, not looped.
PT: # Sorteia todas as reamostras de uma vez: shape (n_resamples, n). Vetorizado, sem loop.
EN: # Right-skewed data — the case where the normal-theory interval is questionable.
PT: # Dados assimétricos à direita: o caso em que o intervalo da teoria normal é questionável.
EN: # The median has no simple closed-form standard error — the bootstrap does not care.
PT: # A mediana não tem um erro padrão simples em forma fechada, e o bootstrap não se importa.
EN: # The bootstrap distribution of the mean.
PT: # A distribuição bootstrap da média.
EN: # Validate against SciPy's implementation.
PT: # Valida com a implementação do SciPy.
