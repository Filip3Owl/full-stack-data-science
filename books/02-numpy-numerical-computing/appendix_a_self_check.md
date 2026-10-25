# Before You Move On
@lead You're ready for Module 03 when you can answer every question below *without looking anything up*. If one makes you hesitate, the chapter in brackets is where to go back to.

## Self-assessment

1. Why is a Python list of numbers both slower *and* larger than an ndarray? *(Chapter 1)*
2. Why does `arr[1:3]` behave differently from `lst[1:3]`? *(Chapter 2)*
3. What does `axis=0` actually collapse? *(Chapter 5)*
4. When does `keepdims=True` matter, and what breaks without it? *(Chapter 5)*
5. What are the two broadcasting rules? *(Chapter 4)*
6. Why does `int32_arr.sum()` not overflow while `int32_arr * 400` does? *(Chapter 3)*
7. Why should you never call `np.linalg.inv` to solve a system? *(Chapter 6)*
8. What does a condition number of 10¹⁶ tell you? *(Chapter 6)*
9. Why is `np.random.seed` a poor choice for reproducibility? *(Chapter 7)*
10. What are strides, and why is `arr.T` free? *(Chapter 8)*

## The three traps this module exists to prevent

If you remember nothing else, remember these. Each produces **wrong numbers rather than an exception**.

1. **A slice is a view.** `arr[1:3] = 0` writes into the original array. Lists copy on slice; arrays don't. *(Chapter 2)*
2. **Integers wrap silently.** `int32` arithmetic past 2,147,483,647 doesn't raise; it goes negative. *(Chapter 3)*
3. **`(n,)` and `(n, 1)` broadcast to `(n, n)`.** No error, an answer of the wrong shape, and a mean over the wrong thing. *(Chapter 4)*

## Where the exercises and solutions live

Each of notebooks 01–08 ends with five exercises, 40 in all, indexed in `modules/02-numpy-numerical-computing/exercises/README.md`. The worked solutions are in notebook `10_exercise_solutions.ipynb`, and the project adds six open-ended extensions.

> [!WARNING]
> **Do the exercises before reading the solutions.** Reading a solution feels
> like learning, and isn't.

## Further reading

| Resource | Why it's worth your time |
| :--- | :--- |
| [NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html) | The official tour: a fast second pass over Chapters 1–2. |
| "Array programming with NumPy" (Harris et al., *Nature*, 2020) | The paper by the maintainers: the clearest account of *why* the design is what it is. |
| [NumPy broadcasting docs](https://numpy.org/doc/stable/user/basics.broadcasting.html) | The rules stated formally, with a shape table worth memorising. |
| "What Every Computer Scientist Should Know About Floating-Point Arithmetic" (Goldberg, 1991) | The reference behind Chapter 3. Dense, and worth it. |
| [NumPy random sampling guide](https://numpy.org/doc/stable/reference/random/index.html) | Why `Generator` replaced `RandomState`, and how `SeedSequence` works. |
| *Introduction to Linear Algebra* (Gilbert Strang) | If Chapter 6 felt thin, this is the book to read alongside Modules 11–19. |
