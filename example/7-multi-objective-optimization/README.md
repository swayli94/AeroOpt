# RVEA ZDT example — notes on results vs. other algorithms

This folder runs **RVEA** (Reference Vector Guided Evolutionary Algorithm) on the same ZDT benchmarks and plotting style as the **DE** and **NSGA-III** examples under `example/`.

If the figure `rvea_zdt_subplots.png` looks **much worse** than e.g. `example/8-de/de_zdt_subplots.png`, that does **not** mean RVEA is “weaker in general” or that the implementation is necessarily wrong. The following points explain why.

## 1. “Newer” ≠ “better on every problem with the same budget”

RVEA (Cheng et al., IEEE TEC 2016) is **not** designed to dominate all older algorithms on **every** multi-objective problem and **every** computational budget.

Its main motivation is **many-objective** optimization (typically **three or more** objectives), where Pareto-based ranking becomes less discriminative. RVEA uses **reference vectors** and **angle-penalized distance (APD)** to balance convergence and diversity in that regime.

The ZDT suite here is **bi-objective** (\(f_1, f_2\)). For two objectives, **NSGA-II/III, DE, and even simple scalarization** can be very competitive. There is **no** theoretical or empirical guarantee that RVEA will look best on **bi-objective ZDT** with a **short** run compared to DE.

## 2. Same computational budget as the DE example

`example_rvea.py` and `example/8-de/example_de.py` both use shared constants from `example/examples_common.py`:

- `POPULATION_SIZE = 32`
- `MAX_ITERATIONS = 20`

**Twenty generations** is **very small** for ZDT problems, especially for difficult instances (e.g. **ZDT4**, **ZDT6**). Any algorithm may still be far from the true Pareto front. Differences in the plots then reflect **which search mechanism fits the landscape under this tight budget**, not a universal ranking of algorithm “age” or prestige.

## 3. Why RVEA can look weaker on these particular plots

- **Reference vectors + APD**: With \(M = 2\), the interaction between the Das–Dennis-style reference set, population size, and the **generation-dependent** APD penalty differs from dominance + crowding (NSGA-II) or **differential mutation** (DE). Under **few generations**, RVEA is not guaranteed to approach the front faster than DE.
- **ZDT4 / ZDT6**: These are **multimodal / harder** landscapes. **Differential evolution** sometimes finds promising regions earlier than reference-vector nicheing; that is a **problem–operator match**, not proof that RVEA is inferior in general.
- **Elite markers (red triangles)**: The script plots the **first non-dominated front** from the valid archive (`db_elite`). If feasible points are sparse or the run is short, that front can look **sparse** and **far** from the analytical ZDT front — also affected by the **constraint** \(x_1^2 + x_2^2 \le 0.64\) used in these examples.

## 4. How to get “nicer” RVEA plots (if you want)

- Increase **`max_iterations`** in the generated `settings.json` block (e.g. hundreds of generations) and re-run.
- Check that **population size** and **reference directions** are consistent (for bi-objective problems, the number of reference points and `population_size` should be chosen coherently).
- Compare RVEA with other methods on **three or more objectives**, where RVEA’s design target is more representative.

## 5. One-sentence summary

**RVEA is not a universal upgrade over DE (or others) on every bi-objective ZDT instance with a 20-generation budget;** seeing better DE figures under this setup is **expected** and reflects **objective count, operator mechanics, and budget** — not necessarily a bug in RVEA.

## References

- R. Cheng, Y. Jin, M. Olhofer, and B. Sendhoff, “A reference vector guided evolutionary algorithm for many-objective optimization,” *IEEE Trans. Evol. Comput.*, 2016.
- Run the example: `python example_rvea.py` (from this directory, with project root on `PYTHONPATH` as in other examples).
