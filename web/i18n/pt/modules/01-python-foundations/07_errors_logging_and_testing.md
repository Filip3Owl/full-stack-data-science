<!-- en:3bac7e86c7 -->
# 07 · Erros, Logging e Testes

**Módulo 01 — Fundamentos de Python para Ciência de Dados**

---

## O que você vai aprender

1. Exceções: como capturar a certa, e por que `except Exception:` esconde bugs.
2. `try / except / else / finally` e o encadeamento de exceções.
3. Exceções personalizadas que fazem as falhas de um pipeline se explicarem sozinhas.
4. **Logging**, e por que `print` não sobrevive ao contato com a produção.
5. Validação defensiva de dados: falhe rápido, falhe de forma clara.
6. **`pytest`**: como testar transformações de dados, incluindo os testes que pegam os bugs que testes unitários normalmente deixam passar.

**Tempo estimado:** 90 minutos

> Este notebook é a diferença entre um script que *parece* funcionar e um pipeline que você deixaria rodar sem supervisão.

<!-- en:111a1e40b3 -->
## 1. A hierarquia de exceções

Exceções formam uma árvore de classes. Capturar uma classe captura todas as suas subclasses, e é exatamente por isso que capturar em um nível alto demais é perigoso.

![Exceções de Python que você vai encontrar toda semana](../../../assets/images/exception_hierarchy.png)

<!-- en:3149e23526 -->
Essa última linha importa. `KeyboardInterrupt` e `SystemExit` herdam de `BaseException`, *não* de `Exception`. Isso é proposital, para que `except Exception:` não engula o seu Ctrl-C. Um `except:` sem classe captura `BaseException` e, com isso, torna impossível interromper um processo descontrolado.

<!-- en:fb0a75d556 -->
## 2. Capture de forma restrita

A regra: **capture a exceção mais específica com a qual você consegue fazer algo de útil.** Todo o resto deve se propagar.

<!-- en:bc7b833dfa -->
> [!WARNING]
> `parse_amount_bad` transforma `"N/A"` em `0.0`. Some uma coluna disso e você terá um número de receita errado, com toda a confiança e sem nenhum aviso. **Devolver um padrão plausível em caso de falha é como números ruins chegam aos dashboards.**

<!-- en:2e79be54d0 -->
## 3. `try / except / else / finally`

As quatro cláusulas têm funções distintas:

| Cláusula | Roda quando | Use para |
| :--- | :--- | :--- |
| `try` | sempre | a operação que pode falhar |
| `except` | uma exceção correspondeu | tratar aquela falha específica |
| `else` | **nenhuma** exceção foi gerada | o trabalho seguinte, mantido fora do `try` |
| `finally` | **sempre**, mesmo em `return` ou ao gerar o erro de novo | limpeza |

<!-- en:a2d489a394 -->
### Encadeamento de exceções: `raise ... from`

Ao converter um erro de baixo nível em um erro de domínio, preserve o original. `from exc` mantém a cadeia completa de causas no traceback.

<!-- en:1de9df8994 -->
## 4. Exceções personalizadas

Definir seus próprios tipos de exceção permite que quem chama trate especificamente as *suas* falhas. Em um pipeline de dados, é assim que você distingue "os dados estão ruins" de "o código está quebrado", dois problemas com responsáveis bem diferentes.

<!-- en:74d7f04257 -->
## 5. Logging

`print` serve em um notebook. Ele deixa de ser adequado no momento em que o código roda sem supervisão, porque não traz gravidade, data e hora nem origem, e não pode ser filtrado nem redirecionado.

| | `print` | `logging` |
| :--- | :-: | :-: |
| Níveis de gravidade | ✘ | ✔ |
| Data e hora | ✘ | ✔ |
| Qual módulo emitiu | ✘ | ✔ |
| Silenciar sem editar o código | ✘ | ✔ |
| Enviar para um arquivo ou serviço | ✘ | ✔ |

<!-- en:eef8c6d79a -->
### Como escolher um nível

| Nível | Use para | Exemplo |
| :--- | :--- | :--- |
| `DEBUG` | Diagnósticos que você só quer ao investigar | shapes, contagens intermediárias |
| `INFO` | Confirmação de que as coisas estão andando | "200 mil linhas carregadas" |
| `WARNING` | Algo inesperado, mas recuperável | "12 nulos imputados" |
| `ERROR` | Uma operação falhou; a execução pode continuar | "uma partição falhou" |
| `CRITICAL` | O programa não pode continuar | "banco de dados inacessível" |

<!-- en:2cd51d9269 -->
### Como registrar uma exceção com o traceback

`logger.exception` só é válido dentro de um bloco `except` e anexa o traceback completo automaticamente. É isso que você quer em um processo agendado: sem isso, você recebe "algo falhou" e não tem como descobrir o quê.

<!-- en:c56f8abd33 -->
> [!TIP]
> Use a formatação preguiçosa com `%s` (`log.info("got %d", n)`), não f-strings. No nível `INFO`, uma mensagem `DEBUG` suprimida não custa nada; com uma f-string, você paga o custo da formatação em toda chamada.

<!-- en:94ef1e63c9 -->
## 6. Validação defensiva: falhe rápido

O bug mais barato é o que é pego na fronteira. Valide o formato dos dados *quando eles entram* no pipeline, não três transformações depois.

<!-- en:99426caf08 -->
> [!WARNING]
> **`assert` serve para bugs, não para validar dados.** Python remove toda instrução `assert` quando roda com `-O` (modo otimizado), então uma asserção não é uma garantia. Use `raise` para tudo o que precisa ser sempre verificado, incluindo toda validação de dados de entrada.

<!-- en:dc5f71e84e -->
## 7. Testes com `pytest`

Quem trabalha com ciência de dados testa menos que quem trabalha com engenharia de software, e esse é o principal motivo de bugs de análise chegarem à produção. Uma transformação é uma *função*: recebe uma entrada e devolve uma saída. Nada é mais fácil de testar.

### Anatomia de um teste

```python
def test_<what>_<condition>_<expected>():
    # Arrange: monte as entradas
    # Act: chame o que está sendo testado
    # Assert: verifique o resultado
```

<!-- en:9054778e9d -->
### Como escrever os testes

Normalmente, eles ficam em `tests/`. Aqui os definimos no próprio notebook para você vê-los rodar e depois olhar o conjunto real no repositório.

<!-- en:59e87ed3ba -->
### Como executá-los

Em um projeto real, você simplesmente digitaria `pytest` no terminal, e ele encontraria todos os arquivos `test_*.py`. Para mostrar o executor funcionando de dentro de um notebook, gravamos o mesmo código em um arquivo temporário e chamamos o pytest sobre ele.

<!-- en:c3a56cf626 -->
### O conjunto real de testes do projeto

O repositório tem um conjunto funcional em `tests/`. Execute-o a qualquer momento com:

```bash
.venv/bin/pytest
```

<!-- en:1a1de43670 -->
### O que testar em código de ciência de dados

| Teste isto | Porque |
| :--- | :--- |
| **Shape**: contagem de linhas e colunas depois de um join | A explosão silenciosa de linhas em um join muitos para muitos é extremamente comum |
| **Nenhuma modificação** das entradas | O bug de aliasing do notebook 02, detectado automaticamente |
| **Casos extremos**: vazio, uma só linha, tudo nulo, tudo zero | Divisão por zero e agregações vazias |
| **Limites**: negativo, zero, muito grande | Reembolsos, itens gratuitos, outliers |
| **Invariantes**: "as parcelas somam 1", "nenhum nulo em uma chave" | Registra o pressuposto do qual a sua matemática depende |
| **Casos de resposta conhecida** | Um exemplo pequeno calculado à mão fixa a definição |

> [!TIP]
> Não persiga 100% de cobertura. Teste as transformações cuja falha seria **silenciosa**: são elas que produzem um número errado com aparência plausível.

<!-- en:47b7848224 -->
## 8. Exercícios

### Exercício 7.1: restrinja o except

```python
def load_and_clean(path):
    try:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df["amount"] = df["amount"].astype(float)
        return df
    except:
        return pd.DataFrame()
```

Liste três falhas distintas que isso esconde. Reescreva de modo que cada uma seja tratada ou propagada da forma adequada.

### Exercício 7.2: hierarquia de exceções personalizada

Projete exceções para um pipeline de ETL que cubram: origem inacessível, esquema incompatível, falha de validação e falha ao gravar no destino. Elas devem compartilhar uma classe base, para que quem chama possa capturar todas de uma vez.

### Exercício 7.3: instrumente uma função

Pegue `add_revenue_features` e adicione logging: `INFO` na entrada, com a contagem de linhas; `WARNING` se mais de 5% dos valores forem nulos; `DEBUG` com o shape resultante.

### Exercício 7.4: escreva os testes primeiro

Antes de escrever qualquer implementação, escreva testes `pytest` para uma função `pct_change(series)`, que devolve a variação percentual entre períodos. Cubra: uma série normal, um zero no denominador, um único elemento, uma série vazia e uma série com nulos. *Depois*, implemente a função para que os testes passem.

### Exercício 7.5: capture um bug real

```python
def merge_customer_orders(customers, orders):
    return customers.merge(orders, on="customer_id", how="left")
```

Escreva um teste que falhe quando `orders` tiver valores duplicados de `customer_id` e o merge multiplicar as linhas sem aviso. Esse bug exato é um dos mais comuns em análises em produção.

<!-- en:bb29edbae7 -->
## Principais lições

- **Capture de forma restrita.** `except Exception:` esconde bugs; um `except:` sem classe também engole o Ctrl-C.
- **Nunca devolva um padrão plausível em caso de falha**: é assim que números errados chegam aos dashboards.
- Use `raise ... from exc` para manter intacta a cadeia de causas.
- **Exceções personalizadas** permitem distinguir "dados ruins" de "código quebrado".
- Use **`logging`**, não `print`, com formatação preguiçosa `%s` e um logger por módulo. `logger.exception` dentro de `except` captura o traceback.
- **`assert` é para bugs; `raise` é para dados.** Asserções desaparecem com `-O`.
- Teste **shape, modificação, casos extremos e invariantes**: as falhas silenciosas são as que valem o teste.

---

**Próximo:** [08 · Projeto do Módulo](08_module_project.ipynb): junte tudo em um pipeline de ponta a ponta, testado e com logs.

<!-- comments -->
EN: # The relationships are real inheritance, and you can verify them:
PT: # As relações são herança de verdade, e você pode verificá-las:
EN: # a typo and a null are now indistinguishable
PT: # um erro de digitação e um nulo agora são indistinguíveis
EN: # genuinely missing — a legitimate, expected state
PT: # realmente ausente: um estado legítimo e esperado
EN: # A non-numeric string is a data-quality problem worth surfacing,
PT: # Uma string não numérica é um problema de qualidade de dados que vale expor,
EN: # not something to paper over with a zero.
PT: # e não algo para disfarçar com um zero.
EN: # Only reached on success. Keeping this OUT of the `try` block means we
PT: # Só é alcançado em caso de sucesso. Manter isto FORA do bloco `try` garante que
EN: # never accidentally catch an exception raised by the follow-on code.
PT: # nunca capturamos por acidente uma exceção gerada pelo código seguinte.
EN: # Always runs — including after the `return` statements above.
PT: # Sempre roda, inclusive depois dos `return` acima.
EN: # `from exc` records the underlying cause, so a reader of the traceback
PT: # `from exc` registra a causa original, então quem lê o traceback
EN: # sees both "config missing" AND which path failed.
PT: # vê tanto "configuração ausente" QUANTO qual caminho falhou.
EN: # A good message states what was wrong AND what was found instead.
PT: # Uma boa mensagem diz o que estava errado E o que foi encontrado no lugar.
EN: # One `except` clause catches all three subtypes.
PT: # Uma única cláusula `except` captura os três subtipos.
EN: # Configure once, at the entry point of your program — never inside a library.
PT: # Configure uma vez, no ponto de entrada do programa, nunca dentro de uma biblioteca.
EN: # so the output appears in the notebook
PT: # para a saída aparecer no notebook
EN: # override any configuration Jupyter already applied
PT: # sobrescreve qualquer configuração que o Jupyter já tenha aplicado
EN: # Convention: one logger per module, named after it. `__name__` gives you the
PT: # Convenção: um logger por módulo, com o nome dele. `__name__` fornece o
EN: # module path for free, which is what makes log filtering possible later.
PT: # caminho do módulo sem esforço, e é isso que permite filtrar os logs depois.
EN: # Lazy %-formatting: the string is only built if the level is enabled.
PT: # Formatação % preguiçosa: a string só é construída se o nível estiver ativo.
EN: # With f-strings, formatting happens even for suppressed messages.
PT: # Com f-strings, a formatação acontece mesmo para mensagens suprimidas.
EN: # 1. Structural checks — cheap, and catch the majority of real mistakes.
PT: # 1. Verificações de estrutura: baratas, pegam a maioria dos erros reais.
EN: # 2. Semantic checks — assumptions the maths depends on.
PT: # 2. Verificações de significado: pressupostos dos quais a matemática depende.
EN: # 3. Postcondition — verify our own output before handing it back.
PT: # 3. Pós-condição: verifica a própria saída antes de devolvê-la.
EN: # An assertion is right here: this fires only if THIS function has a bug,
PT: # Aqui uma asserção está certa: ela só dispara se ESTA função tiver um bug,
EN: # which is exactly what assertions are for.
PT: # que é exatamente para o que servem as asserções.
EN: # The function under test — a realistic feature-engineering step.
PT: # A função em teste: uma etapa realista de engenharia de atributos.
EN: # never mutate the caller's frame
PT: # nunca modifica o DataFrame de quem chama
EN: # Guard the division: an all-zero frame is valid input, not an error.
PT: # Protege a divisão: um DataFrame todo zerado é uma entrada válida, não um erro.
EN: # Float-safe comparison — the lesson from notebook 02, applied.
PT: # Comparação segura para floats: a lição do notebook 02, aplicada.
EN: # THE test data scientists most often forget, and the bug that hurts most.
PT: # O teste que mais se esquece em ciência de dados, e o bug que mais dói.
EN: # --- Edge cases: where the real bugs hide -----------------------------
PT: # --- Casos extremos: onde os bugs de verdade se escondem ---------------
EN: # refunds are legitimately negative
PT: # reembolsos são legitimamente negativos
EN: # Print the per-test result lines, trimming pytest's boilerplate.
PT: # Imprime as linhas de resultado de cada teste, sem o texto padrão do pytest.
