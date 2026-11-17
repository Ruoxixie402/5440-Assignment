# 5440 Assignment 4 — Logistic Regression

Completed solution for Problem 1, parts (a)–(d).

## Main files

- [Corrected, executed Jupyter notebook](5440%20Assignment4_corrected.ipynb)
- [Complete PDF: explanations, results, plots, and source code](5440_Assignment4_Complete.pdf)
- [Documented Python source](assignment4_solution.py)
- [Input data](marks.txt)
- [Numerical results and figures](results/)

## Running the notebook

1. Download this repository using **Code → Download ZIP**, and extract it.
2. Keep `marks.txt` in the same folder as the notebook.
3. Install dependencies if needed:
   ```bash
   python -m pip install -r requirements.txt
   ```
4. Open `5440 Assignment4_corrected.ipynb` in Jupyter.
5. Select **Restart Kernel and Run All Cells**, then save.

The notebook already includes executed outputs. Alternatively, run `python assignment4_solution.py` from the repository folder.

## Methods and checks

- Custom gradient descent with Armijo backtracking.
- Exact-Hessian Newton, with and without backtracking.
- Built-in `fmin_tnc` used only as a reference.
- Analytic gradient and Hessian verified against finite differences.
- Full parameter histories, projected contour plots, loss convergence, and intercept trajectories.
- Median optimizer timing over 51 repeated runs and a separate harder-start comparison.
- Full function documentation and mathematical derivations.

## Data-loading convention

The supplied file contains 100 numerical rows with no header. The instructor-provided reference notebook uses `pd.read_csv("marks.txt")`, which treats the first numerical row as a header and leaves **99 observations**. This solution deliberately reproduces that convention. Reading all 100 rows with `header=None` gives a slightly different estimate.

The two scores are standardized during optimization. This changes the gradient-descent trajectory but not the unregularized model's optimum. Reported coefficients and histories are converted back to the original units. All custom methods use the same stopping rule in standardized coordinates.

## Verified results

| Method | Intercept | w1 | w2 | Accepted updates |
|---|---:|---:|---:|---:|
| Gradient descent + backtracking | -24.869938 | 0.203382 | 0.199882 | 101 |
| Newton, full steps | -24.869933 | 0.203382 | 0.199882 | 7 |
| Newton + backtracking | -24.869933 | 0.203382 | 0.199882 | 7 |

Final summed negative log-likelihood is approximately **20.25118226**, and training accuracy is **88.8889%**. Numerical details and timings may vary across machines and library versions.

The contour plots hold the intercept fixed only when evaluating contours; plotted paths project iterates whose intercept also changes.

The PDF and notebook were executed and checked for consistency. The final grade remains the instructor's decision.

## Reference filename

The supplied reference is `logistic-regression.ipynb`, the notebook corresponding to `logistic-regression.pdf` named in the question. Its reported coefficients and accuracy were checked directly. The instructor reference file itself is not republished here.
