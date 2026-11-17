"""Assignment 4, Problem 1: logistic regression.
Reference example: instructor-provided logistic-regression.ipynb.
Run from the directory containing marks.txt. Requires numpy, pandas,
scipy, matplotlib, and ipython. The reference notebook's header-inference
convention is preserved, leaving 99 observations.
"""
from IPython.display import display


# Import libraries
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import expit
from scipy.optimize import fmin_tnc


np.set_printoptions(precision=8, suppress=True)

OUTPUT = Path("results")
OUTPUT.mkdir(exist_ok=True)


# Load and prepare the data
data_path = Path("marks.txt")


if not data_path.exists():
    raise FileNotFoundError(
        "marks.txt must be in the notebook working directory."
    )

# Deliberately matches the reference notebook.
data = pd.read_csv(data_path).to_numpy(dtype=float)

features = data[:, :2]
y = data[:, 2]

if not np.isfinite(data).all():
    raise ValueError("The data contain missing or nonfinite values.")

if not np.isin(y, [0, 1]).all():
    raise ValueError("The response must contain only 0 and 1.")

# Original feature units.
X = np.column_stack([
    np.ones(len(y)),
    features
])

# Standardization improves gradient-descent conditioning.
feature_mean = features.mean(axis=0)
feature_std = features.std(axis=0)

if np.any(feature_std == 0):
    raise ValueError("A feature has zero standard deviation.")

Z = np.column_stack([
    np.ones(len(y)),
    (features - feature_mean) / feature_std
])

print("Number of observations:", len(y))
print("Original design matrix shape:", X.shape)
print("Standardized design matrix shape:", Z.shape)

display(
    pd.DataFrame(
        data,
        columns=["Exam 1", "Exam 2", "Admitted"]
    ).head()
)


def to_original(parameters):
    """Convert standardized logistic coefficients to original exam-score units.
    
    Parameters
    ----------
    parameters : array_like, shape (3,) or (k, 3)
        Finite coefficients ordered as intercept, first slope, second slope.
    
    Returns
    -------
    numpy.ndarray, shape (k, 3)
        Original-unit coefficients; a single input vector returns one row.
    
    Assumptions
    -----------
    The global feature_mean and feature_std contain the two training-feature
    means and positive population standard deviations used to construct Z.
    """
    parameters = np.atleast_2d(parameters)

    weights = parameters[:, 1:] / feature_std
    intercept = parameters[:, 0] - weights @ feature_mean

    return np.column_stack([intercept, weights])


# Plot the observations
admitted = y == 1
not_admitted = y == 0

plt.figure(figsize=(7, 5))

plt.scatter(
    features[admitted, 0],
    features[admitted, 1],
    label="Admitted",
    marker="+",
    s=70
)

plt.scatter(
    features[not_admitted, 0],
    features[not_admitted, 1],
    label="Not admitted",
    marker="o",
    s=35
)

plt.xlabel("First exam score")
plt.ylabel("Second exam score")
plt.title("Admissions data")
plt.legend()
plt.show()


# Define the loss, gradient, and exact Hessian
def loss(theta, A, y):
    """Evaluate the summed, unregularized logistic negative log-likelihood.
    
    Parameters
    ----------
    theta : numpy.ndarray, shape (d,)
        Finite model coefficients, including the intercept coefficient.
    A : numpy.ndarray, shape (n, d)
        Finite design matrix, including a column of ones for the intercept.
    y : numpy.ndarray, shape (n,)
        Binary labels in {0, 1}; dimensions must agree with A and theta.
    
    Returns
    -------
    float
        Sum of binary cross-entropy losses, not their average.
    
    Notes
    -----
    logaddexp avoids exponent overflow. No regularization is included.
    """
    z = A @ theta
    return float(np.logaddexp(0, z).sum() - y @ z)


def gradient(theta, A, y):
    """Compute the exact analytic gradient of the summed logistic loss.
    
    Parameters
    ----------
    theta : numpy.ndarray, shape (d,)
        Finite model coefficients.
    A : numpy.ndarray, shape (n, d)
        Finite design matrix with an intercept column.
    y : numpy.ndarray, shape (n,)
        Binary labels; dimensions must match A and theta.
    
    Returns
    -------
    numpy.ndarray, shape (d,)
        A.T @ (sigmoid(A @ theta) - y), with no regularization term.
    """
    probabilities = expit(A @ theta)
    return A.T @ (probabilities - y)


def hessian(theta, A, y):
    """Compute the exact analytic Hessian H = A.T @ diag(p*(1-p)) @ A.
    
    Parameters
    ----------
    theta : numpy.ndarray, shape (d,)
        Finite model coefficients.
    A : numpy.ndarray, shape (n, d)
        Finite design matrix with an intercept column.
    y : numpy.ndarray, shape (n,)
        Binary labels. Retained for a common interface; H does not use y.
    
    Returns
    -------
    numpy.ndarray, shape (d, d)
        Exact symmetric Hessian, evaluated without an n-by-n diagonal array.
    
    Assumptions
    -----------
    Dimensions agree. Full column rank and nonsaturated probabilities are
    needed for a nonsingular numerical Newton system. No penalty is added.
    """
    z = A @ theta

    # Equivalent to p * (1 - p).
    variance = expit(z) * expit(-z)

    return A.T @ (variance[:, None] * A)


# (a) Backtracking line search
def backtracking(
    theta,
    direction,
    A,
    y,
    current_loss,
    g,
    c=1e-4,
    shrink=0.5,
    max_backtracks=80
):
    """Select a step satisfying Armijo sufficient decrease.
    
    Parameters
    ----------
    theta, direction, g : numpy.ndarray, shape (d,)
        Current coefficients, descent direction, and current gradient.
    A : numpy.ndarray, shape (n, d)
        Finite design matrix with an intercept column.
    y : numpy.ndarray, shape (n,)
        Binary labels with compatible dimensions.
    current_loss : float
        loss(theta, A, y), using the same summed objective.
    c : float, default 1e-4
        Armijo fraction, satisfying 0 < c < 0.5.
    shrink : float, default 0.5
        Step contraction factor, satisfying 0 < shrink < 1.
    max_backtracks : int, default 80
        Positive maximum number of trial steps; the first trial is alpha=1.
    
    Returns
    -------
    (alpha, candidate_loss, reductions) : tuple
        Accepted positive step, its objective, and number of contractions.
    
    Raises
    ------
    ValueError
        If g @ direction is nonnegative.
    RuntimeError
        If none of the allowed trials satisfies Armijo.
    
    Assumptions
    -----------
    Inputs have matching dimensions; g and current_loss refer to theta.
    """
    directional_derivative = float(g @ direction)

    if directional_derivative >= 0:
        raise ValueError("The direction is not a descent direction.")

    alpha = 1.0

    for reductions in range(max_backtracks):
        candidate = theta + alpha * direction
        candidate_loss = loss(candidate, A, y)

        armijo_bound = (
            current_loss
            + c * alpha * directional_derivative
        )

        if (
            np.isfinite(candidate_loss)
            and candidate_loss <= armijo_bound
        ):
            return alpha, candidate_loss, reductions

        alpha *= shrink

    raise RuntimeError(
        "Backtracking could not find an acceptable step."
    )


# Part (a) and (d): Custom optimizers
# This function implements: Gradient descent with backtracking/ Newton’s method with full steps/Newton’s method with backtracking
def optimize(
    A,
    y,
    method="gd",
    line_search=True,
    initial=None,
    tol=1e-6,
    maxiter=20000
):
    """Minimize summed logistic loss by custom gradient descent or Newton.
    
    Parameters
    ----------
    A : numpy.ndarray, shape (n, d)
        Finite design matrix, optionally standardized; includes intercept.
    y : numpy.ndarray, shape (n,)
        Binary labels with compatible dimensions.
    method : {'gd', 'newton'}
        Negative-gradient direction or exact-Hessian Newton direction.
    line_search : bool
        Use Armijo backtracking if True; otherwise take alpha=1.
    initial : array_like, shape (d,), or None
        Starting coefficients in A's coordinates; None means all zeros.
    tol : positive float
        Stop when the gradient infinity norm in A's coordinates is <= tol.
    maxiter : positive int
        Maximum number of accepted parameter updates.
    
    Returns
    -------
    dict
        theta: final coefficients; path: initial and all accepted vectors;
        losses: objective for every vector; step_sizes and backtracks: one
        value per accepted update; status: termination reason; steps: updates.
    
    Assumptions
    -----------
    A and y have the stated dimensions. Newton requires a solvable Hessian.
    Numerical failures return a status rather than a fabricated solution.
    No built-in optimizer or regularization is used in this routine.
    """
    theta = (
        np.zeros(A.shape[1])
        if initial is None
        else np.array(initial, dtype=float).copy()
    )

    path = [theta.copy()]
    losses = [loss(theta, A, y)]
    step_sizes = []
    backtracks = []

    status = "maximum iterations"

    for _ in range(maxiter):
        g = gradient(theta, A, y)

        if np.linalg.norm(g, ord=np.inf) <= tol:
            status = "converged"
            break

        if method == "gd":
            direction = -g

        elif method == "newton":
            H = hessian(theta, A, y)

            try:
                # Solve H @ direction = -g.
                direction = np.linalg.solve(H, -g)

            except np.linalg.LinAlgError:
                status = "singular Hessian"
                break

        else:
            raise ValueError("Use method='gd' or method='newton'.")

        if (
            not np.all(np.isfinite(direction))
            or g @ direction >= 0
        ):
            status = "invalid descent direction"
            break

        if line_search:
            try:
                alpha, new_loss, reductions = backtracking(
                    theta=theta,
                    direction=direction,
                    A=A,
                    y=y,
                    current_loss=losses[-1],
                    g=g
                )

            except RuntimeError:
                status = "line search failed"
                break

        else:
            alpha = 1.0
            reductions = 0
            new_loss = loss(
                theta + direction,
                A,
                y
            )

        theta = theta + alpha * direction

        path.append(theta.copy())
        losses.append(new_loss)
        step_sizes.append(alpha)
        backtracks.append(reductions)

        if not np.isfinite(new_loss):
            status = "nonfinite loss"
            break

    # Handle convergence occurring exactly on the final allowed step.
    if (
        status == "maximum iterations"
        and np.linalg.norm(
            gradient(theta, A, y),
            ord=np.inf
        ) <= tol
    ):
        status = "converged"

    return {
        "theta": theta,
        "path": np.array(path),
        "losses": np.array(losses),
        "step_sizes": np.array(step_sizes),
        "backtracks": np.array(backtracks),
        "status": status,
        "steps": len(path) - 1
    }


# Check the analytic derivatives: Check the gradient and Hessian against central finite differences
probe = np.array([0.2, -0.3, 0.4])
epsilon = 1e-5
basis = np.eye(3)

numerical_gradient = np.array([
    (
        loss(probe + epsilon * direction, Z, y)
        - loss(probe - epsilon * direction, Z, y)
    ) / (2 * epsilon)
    for direction in basis
])

numerical_hessian = np.column_stack([
    (
        gradient(probe + epsilon * direction, Z, y)
        - gradient(probe - epsilon * direction, Z, y)
    ) / (2 * epsilon)
    for direction in basis
])

assert np.allclose(
    gradient(probe, Z, y),
    numerical_gradient,
    rtol=1e-6,
    atol=1e-6
)

assert np.allclose(
    hessian(probe, Z, y),
    numerical_hessian,
    rtol=1e-6,
    atol=1e-6
)

print("Gradient and Hessian checks passed.")


# (b) Compute the TNC reference
tnc_theta, tnc_evaluations, tnc_return_code = fmin_tnc(
    func=loss,
    x0=np.zeros(3),
    fprime=gradient,
    args=(X, y),
    messages=0,
    maxfun=10000,
    pgtol=1e-10,
    ftol=1e-14,
    xtol=1e-12
)

reference_theta = np.array([
    -24.86874868,
    0.20337210,
    0.19987258
])

print("Coefficients reported in logistic-regression.ipynb:")
print(reference_theta)

print("\nTighter-tolerance TNC coefficients:")
print(tnc_theta)

print("\nTNC loss:", loss(tnc_theta, X, y))


# Run all methods and compare performance
methods = {
    "GD + backtracking": ("gd", True),
    "Newton, full steps": ("newton", False),
    "Newton + backtracking": ("newton", True)
}

runs = {}
comparison_rows = []

for name, (method, use_backtracking) in methods.items():

    result = optimize(
        Z,
        y,
        method=method,
        line_search=use_backtracking
    )

    runs[name] = result
    theta_original = to_original(result["theta"])[0]

    # Warm-up run.
    optimize(
        Z,
        y,
        method=method,
        line_search=use_backtracking
    )

    timings = []

    for _ in range(51):
        start = perf_counter()

        optimize(
            Z,
            y,
            method=method,
            line_search=use_backtracking
        )

        timings.append(perf_counter() - start)

    comparison_rows.append({
        "Method": name,
        "Intercept": theta_original[0],
        "w1": theta_original[1],
        "w2": theta_original[2],
        "Loss": loss(theta_original, X, y),
        "Steps": result["steps"],
        "Accuracy (%)": (
            100 * np.mean((X @ theta_original >= 0) == y)
        ),
        "Distance to reference": np.linalg.norm(
            theta_original - reference_theta
        ),
        "Distance to TNC": np.linalg.norm(
            theta_original - tnc_theta
        ),
        "Median time (ms)": 1000 * np.median(timings),
        "Backtrack reductions": int(
            result["backtracks"].sum()
        ),
        "Status": result["status"]
    })

comparison = pd.DataFrame(comparison_rows)

display(comparison.round(8))


# (c) Inspect the recorded parameter vectors
histories = {}

for name, result in runs.items():

    history = pd.DataFrame(
        to_original(result["path"]),
        columns=["Intercept", "w1", "w2"]
    )

    history.index.name = "Iteration"

    history["Loss"] = result["losses"]
    history["Step size"] = np.r_[
        np.nan,
        result["step_sizes"]
    ]
    history["Backtrack reductions"] = np.r_[
        0,
        result["backtracks"]
    ]

    histories[name] = history

print("First five gradient-descent iterates:")
display(histories["GD + backtracking"].head())

print("Last five gradient-descent iterates:")
display(histories["GD + backtracking"].tail())


# Part (c) and (d) : Plot paths over loss contours
optimum = to_original(
    runs["Newton + backtracking"]["theta"]
)[0]

minimum_loss = loss(optimum, X, y)

all_paths = np.vstack([
    to_original(result["path"])
    for result in runs.values()
])

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5),
    constrained_layout=True
)

for ax, zoom in zip(axes, [False, True]):

    if zoom:
        limits = (
            optimum[1] - 0.035,
            optimum[1] + 0.035,
            optimum[2] - 0.035,
            optimum[2] + 0.035
        )

        offsets = np.array([
            0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16
        ])

    else:
        limits = (
            0,
            max(0.24, 1.05 * all_paths[:, 1].max()),
            0,
            max(0.24, 1.05 * all_paths[:, 2].max())
        )

        offsets = np.array([
            0.1, 1, 5, 15, 50, 150, 400, 800, 1400, 2000
        ])

    w1_grid, w2_grid = np.meshgrid(
        np.linspace(limits[0], limits[1], 180),
        np.linspace(limits[2], limits[3], 180)
    )

    grid_parameters = np.column_stack([
        np.full(w1_grid.size, optimum[0]),
        w1_grid.ravel(),
        w2_grid.ravel()
    ])

    contour_values = np.empty(len(grid_parameters))

    # Evaluate in batches to avoid large temporary arrays.
    for start in range(0, len(grid_parameters), 1000):

        logits = X @ grid_parameters[start:start + 1000].T

        contour_values[start:start + 1000] = (
            np.logaddexp(0, logits)
            - y[:, None] * logits
        ).sum(axis=0)

    contours = ax.contour(
        w1_grid,
        w2_grid,
        contour_values.reshape(w1_grid.shape),
        levels=minimum_loss + offsets,
        colors="0.7",
        linewidths=0.8
    )

    ax.clabel(
        contours,
        fontsize=7,
        fmt="%.1f"
    )

    for name, result in runs.items():

        path = to_original(result["path"])

        ax.plot(
            path[:, 1],
            path[:, 2],
            "-o",
            markersize=2,
            linewidth=1.2,
            label=name
        )

    ax.plot(
        optimum[1],
        optimum[2],
        "k*",
        markersize=12,
        label="Optimum"
    )

    ax.set(
        xlabel="$w_1$",
        ylabel="$w_2$",
        xlim=limits[:2],
        ylim=limits[2:],
        title=(
            "Near the optimum"
            if zoom else "Full projected paths"
        )
    )

axes[0].legend(fontsize=8)

fig.suptitle(
    "Fixed-intercept loss contours and projected optimization paths",
    fontsize=12
)

fig.savefig(
    OUTPUT / "optimization_paths.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# Plot loss convergence and the intercept
fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4),
    constrained_layout=True
)

for name, result in runs.items():

    objective_gap = np.maximum(
        result["losses"] - minimum_loss,
        1e-14
    )

    axes[0].semilogy(
        objective_gap,
        label=name
    )

    original_path = to_original(result["path"])

    axes[1].plot(
        original_path[:, 0],
        label=name
    )

axes[0].set(
    xlabel="Accepted step",
    ylabel="Loss minus optimum (floored at 1e-14)",
    title="Objective convergence"
)

axes[1].set(
    xlabel="Accepted step",
    ylabel="Intercept",
    title="Intercept trajectory"
)

axes[0].legend(fontsize=8)

fig.savefig(
    OUTPUT / "convergence.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# (d): Test Newton from a harder starting point
hard_start = np.array([-5.0, 5.0, 5.0])

stress_rows = []

for use_backtracking in [False, True]:

    result = optimize(
        Z,
        y,
        method="newton",
        line_search=use_backtracking,
        initial=hard_start,
        maxiter=100
    )

    stress_rows.append({
        "Backtracking": use_backtracking,
        "Status": result["status"],
        "Accepted steps": result["steps"],
        "Final loss": result["losses"][-1],
        "Total backtrack reductions": int(
            result["backtracks"].sum()
        )
    })

stress_results = pd.DataFrame(stress_rows)

display(stress_results)


# Plot the fitted decision boundary
theta = to_original(
    runs["Newton + backtracking"]["theta"]
)[0]

exam1_values = np.linspace(
    features[:, 0].min() - 5,
    features[:, 0].max() + 5,
    200
)

exam2_boundary = -(
    theta[0] + theta[1] * exam1_values
) / theta[2]

plt.figure(figsize=(7, 5))

plt.scatter(
    features[admitted, 0],
    features[admitted, 1],
    marker="+",
    s=70,
    label="Admitted"
)

plt.scatter(
    features[not_admitted, 0],
    features[not_admitted, 1],
    marker="o",
    s=35,
    label="Not admitted"
)

plt.plot(
    exam1_values,
    exam2_boundary,
    color="black",
    label="Decision boundary: p = 0.5"
)

plt.xlabel("First exam score")
plt.ylabel("Second exam score")
plt.title("Fitted logistic-regression classifier")

plt.xlim(
    features[:, 0].min() - 5,
    features[:, 0].max() + 5
)

plt.ylim(
    features[:, 1].min() - 5,
    features[:, 1].max() + 5
)

plt.legend()

plt.savefig(
    OUTPUT / "decision_boundary.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# Export results to CSV
comparison.to_csv(
    OUTPUT / "method_comparison.csv",
    index=False
)

stress_results.to_csv(
    OUTPUT / "newton_stress_test.csv",
    index=False
)

filenames = {
    "GD + backtracking": "gd_backtracking_trajectory.csv",
    "Newton, full steps": "newton_full_trajectory.csv",
    "Newton + backtracking": "newton_backtracking_trajectory.csv"
}

for name, history in histories.items():
    history.to_csv(
        OUTPUT / filenames[name]
    )

print("Comparison tables, parameter histories, and figures saved to results/.")
