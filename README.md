[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/rIbFx4ql)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=22105229)
# Optimization Mini-Project (Pick 1 of 4 Topics)

This repository provides **four implementation-focused mini-project topics**. Student teams (**2–5 students**) must select **one** topic, implement optimization algorithms, benchmark against baselines, and submit code + a concise Markdown report.

---

## 1) Key Rules

### 1.1 Team, deliverables, and report format

* **Team size:** 2–5 students
* **Submission:** GitHub repository link (one repo per team)
* **Report:** `report.md` in Markdown, **single-column**, **no more than 5 pages** (figures/tables count toward pages)
* **Required artifacts:**

  * `report.md`
  * `README.md` (this file, updated by your team with exact run instructions)
  * Source code (`.py` recommended; notebooks allowed—see §1.4)
  * `results/` (logs in `.json` or `.csv`)
  * `figures/` (plots exported as `.png` or `.pdf`)
  * `requirements.txt` or `environment.yml`

### 1.2 Baselines (explicitly encouraged)

You are **encouraged** to benchmark against library solvers/optimizers:

* `torch.optim` (e.g., `SGD`, `Adam`, `LBFGS`) as baselines
* `cvxpy` (and its solvers) as reference solutions/baselines when applicable

**Important:** at least **two** of your compared methods must be **your own implementations** (i.e., you implement the update rule, not just call a library optimizer).

### 1.3 Reproducibility requirement

Your repo must include **one** reproducible entry point:

* **Preferred:** a script command (e.g., `python main.py`) that regenerates the main figures/tables
  **or**
* a notebook `final_run.ipynb` that works via **Restart Kernel → Run All** and regenerates outputs in `figures/` and `results/`.

Your `README.md` must state:

* Python version
* environment setup commands
* one command / one notebook to reproduce results
* random seed(s)

### 1.4 Notebook vs. Python source files

* **Recommended:** Python scripts (`main.py`) for experiments + optional notebooks for exploration.
* **Allowed:** notebook-only submissions **if** `final_run.ipynb` runs end-to-end from a fresh kernel and regenerates outputs.

---

## 2) Performance Metrics (what you must compare)

Each team must report metrics in three categories (appropriate to your chosen topic):

### 2.1 Solution quality / accuracy (choose at least one)

* **Objective value** (F(x))
* **Objective gap** to best-known (F^\star):
  [
  \mathrm{gap}(x)=\frac{|F(x)-F^\star|}{\max(1,|F^\star|)}
  ]
* **Feasibility violation** (constrained problems)
* **Recovery error** (matrix completion): (|UV^\top-M^\star|_F/|M^\star|_F)
* **Prediction accuracy** (optional, if you evaluate classification)

### 2.2 Speed / efficiency (required)

* **Wall-clock time** (seconds)
* Iterations and/or gradient evaluations (recommended)

### 2.3 Convergence / learning behavior (required)

* Curves: metric vs **iterations** and metric vs **time**
* Stationarity proxy (optional): (|\nabla F(x)|), KKT residual, or ADMM residuals
* For online/bandit: **regret** curves

**Minimum plotting requirement (all topics):**

* At least **2 plots** (vs iteration and/or vs time)
* At least **1 summary table** comparing methods

---

## 3) Grading (suggested rubric)

* **70% Implementation quality**

  * correctness of update rules and objective computations
  * numerical stability and reasonable stopping rules
  * reproducible runs and clean code organization
* **20% Experimental quality**

  * fair comparisons (same data, clear hyperparameters, sensible tuning)
  * meaningful ablations (e.g., stepsize, regularization, batch size)
  * appropriate metrics and clear plots
* **10% Report quality**

  * concise and clear, correct conclusions, limitations discussed

---
## Pick 1 of 4 Topics (Implementation-Focused)

See [Project.md](./Project.md) for details


## 5) Repository Layout (recommended, not enforced)

You may organize your code freely, but the following is recommended:

```
.
├── README.md
├── report.md
├── requirements.txt  (or environment.yml)
├── src/
├── notebooks/        (optional)
├── results/
└── figures/
```

---

## 6) What to put in your team’s README (required)

Include:

* Topic chosen (1/2/3/4)
* Team members and contributions
* Setup commands
* Exact reproduction instructions (`mail.py` or `final_run.ipynb`)
* Brief summary of implemented methods (“ours”) and baseline methods (“library”)

---

If you want, paste your course’s exact due date/time and submission channel (e.g., LMS link), and I will insert a final “Submission” section with precise wording.
