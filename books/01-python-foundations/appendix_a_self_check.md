# Before You Move On
@lead You're ready for Module 02 when you can answer every question below *without looking anything up*. If one makes you hesitate, the chapter in brackets is where to go back to.

## Self-assessment

1. Why does `b = a` followed by `b.append(1)` sometimes change `a`, and sometimes not? *(Chapter 2)*
2. Why is `0.1 + 0.2 == 0.3` `False`, and what should you write instead? *(Chapter 2)*
3. Why is `x in my_set` faster than `x in my_list`, and roughly by how much? *(Chapter 3)*
4. What's the difference between a list comprehension and a generator expression? *(Chapters 3–4)*
5. What does `def f(x=[])` do, and why is it a bug? *(Chapter 4)*
6. When would you use `@dataclass` rather than writing `__init__` yourself? *(Chapter 5)*
7. Why is `except Exception:` usually wrong? *(Chapter 7)*
8. Why should a pipeline use `logging` rather than `print`? *(Chapter 7)*
9. What makes Parquet a better choice than CSV for a 5 GB dataset? *(Chapter 6)*
10. Why must `assert` never be used to validate incoming data? *(Chapter 7)*

## Where the exercises and solutions live

Every chapter ends with the exercises from its notebook. The full list, 29 exercises plus the project's six extensions, is in `modules/01-python-foundations/exercises/README.md`, and the worked solutions with their reasoning are in notebook `09_exercise_solutions.ipynb`.

> [!WARNING]
> **Attempt each exercise before opening the solution.** Reading a worked
> answer produces a convincing feeling of understanding that vanishes the
> moment you face a blank cell. Ten minutes of being stuck teaches more than an
> hour of reading.

Two habits will tell you whether your own answer is right, solution or not. **Write a test**: if you can't state what the correct output is, you don't understand the problem yet. And **Restart & Run All**: if your solution only works when cells run in the order you happened to click them, it doesn't work.

## Further reading

| Resource | Why it's worth your time |
| :--- | :--- |
| *Fluent Python*, 2nd ed. (Luciano Ramalho) | The definitive book on Python's object model. Its first chapters map onto Chapters 2 and 3. |
| [PEP 8](https://peps.python.org/pep-0008/) | The style guide every Python codebase assumes you've read. |
| [PEP 484](https://peps.python.org/pep-0484/) | Type hints, used throughout `src/dstoolkit/`. |
| [The `collections` docs](https://docs.python.org/3/library/collections.html) | `Counter` and `defaultdict` alone will save you hundreds of lines. |
| [TimeComplexity wiki](https://wiki.python.org/moin/TimeComplexity) | The cost of every built-in operation, in one table. |
