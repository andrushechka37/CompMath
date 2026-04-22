
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class SolveResult:
    x: np.ndarray
    y: np.ndarray
    iterations: int
    converged: bool
    history: list


def solve_tridiagonal(lower, diag, upper, rhs):
    n = len(diag)
    a = lower.astype(float).copy()
    b = diag.astype(float).copy()
    c = upper.astype(float).copy()
    d = rhs.astype(float).copy()

    for i in range(1, n):
        m = a[i - 1] / b[i - 1]
        b[i] -= m * c[i - 1]
        d[i] -= m * d[i - 1]

    x = np.empty(n, dtype=float)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x


class QuasilinearizedBVP:
    def __init__(self, eps, A, B, n=2001):
        self.eps = float(eps)
        self.A = float(A)
        self.B = float(B)
        self.n = int(n)
        self.x = np.linspace(0.0, 1.0, self.n)
        self.h = self.x[1] - self.x[0]
        self.last_result = None

    def linear_guess(self):
        return self.A + (self.B - self.A) * self.x

    def internal_layer_guess(self, center=0.5):
        s = np.sqrt(max(2.0 * self.eps, 1e-14))
        g = np.tanh((self.x - center) / s) # догадка как тангенс
        return g + (self.A - g[0]) * (1.0 - self.x) + (self.B - g[-1]) * self.x # лин добавка чтоб удовл ну

    def positive_guess(self, amplitude=1.0): # позитив/негатив гессы
        return amplitude * np.sin(np.pi * self.x)

    def negative_guess(self, amplitude=1.0):
        return -amplitude * np.sin(np.pi * self.x)

    def _linearized_step(self, yk): # f(y)≈f(yk)+f'(yk)(y - yk).
        h2 = self.h * self.h
        yi = yk[1:-1]

        lower = np.full(self.n - 3, self.eps / h2, dtype=float)
        upper = np.full(self.n - 3, self.eps / h2, dtype=float)
        diag = -2.0 * self.eps / h2 - (3.0 * yi * yi - 1.0)
        rhs = -2.0 * yi * yi * yi

        rhs[0] -= (self.eps / h2) * self.A
        rhs[-1] -= (self.eps / h2) * self.B

        y_inner = solve_tridiagonal(lower, diag, upper, rhs)

        y_new = np.empty_like(yk)
        y_new[0] = self.A
        y_new[-1] = self.B
        y_new[1:-1] = y_inner
        return y_new

    def solve(self, y0=None, tol=1e-8, max_iter=100, damping=1.0):
        if y0 is None:
            if self.A < 0.0 < self.B:
                yk = self.internal_layer_guess()
            elif self.A == 0.0 and self.B == 0.0:
                yk = np.zeros_like(self.x)
            else:
                yk = self.linear_guess()
        else:
            yk = np.asarray(y0, dtype=float).copy()

        history = []
        converged = False

        for it in range(1, max_iter + 1):
            y_new = self._linearized_step(yk)

            if damping != 1.0:
                y_new = damping * y_new + (1.0 - damping) * yk

            err = float(np.max(np.abs(y_new - yk)))
            history.append(err)
            yk = y_new

            if err < tol:
                converged = True
                break

        result = SolveResult(
            x=self.x.copy(),
            y=yk.copy(),
            iterations=it,
            converged=converged,
            history=history,
        )
        self.last_result = result
        return result

    def compute_layer_thickness(self, result=None, level=0.9):
        if result is None:
            result = self.last_result
        if result is None:
            raise RuntimeError("Сначала вызовите solve().")

        x = result.x
        y = result.y

        if not np.all(np.diff(y) >= -1e-10):
            raise RuntimeError("Толщина внутреннего слоя определяется для монотонного решения.")
        if y.min() > -level or y.max() < level:
            raise RuntimeError("Решение не достигает уровней ±level.")

        x_left = np.interp(-level, y, x)
        x_right = np.interp(level, y, x)

        return {
            "level": level,
            "x_left": float(x_left),
            "x_right": float(x_right),
            "thickness": float(x_right - x_left),
        }

    def summary(self, result=None, level=0.9):
        if result is None:
            result = self.last_result
        if result is None:
            raise RuntimeError("Сначала вызовите solve().")
        layer = self.compute_layer_thickness(result=result, level=level)
        return {
            "eps": self.eps,
            "A": self.A,
            "B": self.B,
            "n": self.n,
            "iterations": result.iterations,
            "converged": result.converged,
            "max_y": float(result.y.max()),
            "min_y": float(result.y.min()),
            "layer_thickness": layer["thickness"],
        }

    def plot_solution(self, result=None, ax=None, label=None):
        if result is None:
            result = self.last_result
        if result is None:
            raise RuntimeError("Сначала вызовите solve().")
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))
        ax.plot(result.x, result.y, label=label or fr"$\varepsilon={self.eps:g}$")
        ax.set_xlabel("x")
        ax.set_ylabel("y(x)")
        ax.grid(True, alpha=0.3)
        if label is not None:
            ax.legend()
        return ax

    def plot_iterations(self, result=None, ax=None):
        if result is None:
            result = self.last_result
        if result is None:
            raise RuntimeError("Сначала вызовите solve().")
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        ax.semilogy(range(1, len(result.history) + 1), result.history, marker="o")
        ax.set_xlabel("Итерация")
        ax.set_ylabel(r"$\max |y^{(k+1)} - y^{(k)}|$")
        ax.grid(True, alpha=0.3)
        return ax


def solve_problem(eps, A, B, n=2001, y0=None, tol=1e-8, max_iter=100, damping=1.0):
    solver = QuasilinearizedBVP(eps=eps, A=A, B=B, n=n)
    result = solver.solve(y0=y0, tol=tol, max_iter=max_iter, damping=damping)
    return solver, result


def study_layer_thickness(eps_values, A=-2.0, B=2.0, n=2001, level=0.9, tol=1e-8, max_iter=300, damping=0.7):
    rows = []
    solutions = {}
    prev = None

    for eps in sorted(list(eps_values), reverse=True):
        solver = QuasilinearizedBVP(eps=eps, A=A, B=B, n=n)
        if prev is None:
            y0 = solver.internal_layer_guess()
        else:
            y0 = np.interp(solver.x, prev.x, prev.y)

        result = solver.solve(y0=y0, tol=tol, max_iter=max_iter, damping=damping)
        layer = solver.compute_layer_thickness(result=result, level=level)

        rows.append({
            "eps": eps,
            "iterations": result.iterations,
            "converged": result.converged,
            "x_left": layer["x_left"],
            "x_right": layer["x_right"],
            "thickness": layer["thickness"],
        })
        solutions[eps] = (solver, result)
        prev = result

    df = pd.DataFrame(rows).sort_values("eps").reset_index(drop=True)
    return df, solutions


def fit_power_law(df):
    x = np.log(df["eps"].to_numpy(dtype=float))
    y = np.log(df["thickness"].to_numpy(dtype=float))
    power, log_const = np.polyfit(x, y, 1)
    return {"power": float(power), "const": float(np.exp(log_const))}


def plot_thickness_vs_eps(df, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(df["eps"], df["thickness"], marker="o")
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel("Толщина внутреннего слоя")
    ax.grid(True, which="both", alpha=0.3)
    return ax


def solve_zero_boundary_family(eps, n=2001, tol=1e-8, max_iter=150, damping=0.8):
    zero_solver = QuasilinearizedBVP(eps=eps, A=0.0, B=0.0, n=n)
    zero_result = zero_solver.solve(y0=np.zeros_like(zero_solver.x), tol=tol, max_iter=max_iter)

    plus_solver = QuasilinearizedBVP(eps=eps, A=0.0, B=0.0, n=n)
    plus_result = plus_solver.solve(
        y0=plus_solver.positive_guess(1.2), tol=tol, max_iter=max_iter, damping=damping
    )

    minus_solver = QuasilinearizedBVP(eps=eps, A=0.0, B=0.0, n=n)
    minus_result = minus_solver.solve(
        y0=minus_solver.negative_guess(1.2), tol=tol, max_iter=max_iter, damping=damping
    )

    return {
        "zero": (zero_solver, zero_result),
        "plus": (plus_solver, plus_result),
        "minus": (minus_solver, minus_result),
    }


def plot_zero_boundary_solutions(solution_dict, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    for name, (solver, result) in solution_dict.items():
        ax.plot(result.x, result.y, label=name)
    ax.set_xlabel("x")
    ax.set_ylabel("y(x)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax
