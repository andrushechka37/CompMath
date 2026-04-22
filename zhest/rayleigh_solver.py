from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import LinAlgWarning, solve
from scipy.optimize import root

Array = np.ndarray


@dataclass
class SolveStats:
    steps: int = 0
    newton_iterations: int = 0
    newton_failures: int = 0
    rejected_substeps: int = 0


class ODESystem:
    def f(self, t: float, y: Array) -> Array:
        raise NotImplementedError

    def jac(self, t: float, y: Array) -> Array:
        raise NotImplementedError


class RayleighSystem(ODESystem):
    def __init__(self, mu: float = 1000.0):
        self.mu = float(mu)

    def f(self, t: float, y: Array) -> Array:
        x, v = y
        return np.array([v, self.mu * (1.0 - v * v) * v - x], dtype=float)

    def jac(self, t: float, y: Array) -> Array:
        _, v = y
        return np.array([[0.0, 1.0], [-1.0, self.mu * (1.0 - 3.0 * v * v)]], dtype=float)


class BaseSolver:
    def __init__(self, system: ODESystem, name: str):
        self.system = system
        self.name = name
        self.stats = SolveStats()

    def solve(self, t_span: Tuple[float, float], y0: Array, h: float) -> Dict[str, Array]:
        raise NotImplementedError


class ImplicitRKBase(BaseSolver):
    newton_tol: float = 1e-10
    newton_maxiter: int = 12
    min_step: float = 1e-6
    adaptive_rtol: float = 1e-4
    adaptive_atol: float = 1e-8

    def __init__(self, system: ODESystem, name: str):
        super().__init__(system, name)
        A, b, c, order = self.butcher_tableau()
        self.A = np.asarray(A, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.c = np.asarray(c, dtype=float)
        self.order = int(order)
        self.s = len(self.b)
        self.is_dirk = bool(np.allclose(self.A, np.tril(self.A)))

    def butcher_tableau(self) -> Tuple[Array, Array, Array, int]:
        raise NotImplementedError

    def _stage_states(self, y: Array, h: float, K: Array) -> List[Array]:
        states: List[Array] = []
        for i in range(self.s):
            yi = y.copy()
            for j in range(self.s):
                yi += h * self.A[i, j] * K[j]
            states.append(yi)
        return states

    def _initial_guess(self, t: float, y: Array, h: float) -> Array:
        f0 = self.system.f(t, y)
        K = np.empty((self.s, y.size), dtype=float)
        for i in range(self.s):
            K[i] = self.system.f(t + self.c[i] * h, y + self.c[i] * h * f0)
        return K

    def _safe_linear_solve(self, mat: Array, rhs: Array) -> Array:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", LinAlgWarning)
                return solve(mat, rhs, assume_a="gen", check_finite=True)
        except (LinAlgWarning, ValueError, np.linalg.LinAlgError) as exc:
            self.stats.newton_failures += 1
            raise FloatingPointError(f"ill-conditioned Newton matrix: {exc}") from exc

    def _apply_damping(self, state: Array, delta: Array, current_norm: float, residual_fn) -> Array:
        for damping in (1.0, 0.5, 0.25, 0.1):
            trial = state + damping * delta
            trial_norm = residual_fn(trial)
            if np.isfinite(trial_norm) and trial_norm < current_norm:
                return trial
        self.stats.newton_failures += 1
        raise FloatingPointError("Newton backtracking failed")

    def _solve_full(self, t: float, y: Array, h: float) -> Array:
        n = y.size
        eye = np.eye(n)
        K = self._initial_guess(t, y, h)

        def residual(stages: Array) -> Tuple[Array, float, List[Array], List[Array]]:
            states = self._stage_states(y, h, stages)
            values = [self.system.f(t + self.c[i] * h, states[i]) for i in range(self.s)]
            res = stages - np.vstack(values)
            norm = float(np.linalg.norm(res.reshape(-1), ord=np.inf))
            return res, norm, states, values

        for _ in range(self.newton_maxiter):
            res, res_norm, states, _ = residual(K)
            self.stats.newton_iterations += 1
            if res_norm < self.newton_tol:
                return K

            big = np.zeros((self.s * n, self.s * n), dtype=float)
            for i in range(self.s):
                jac_i = self.system.jac(t + self.c[i] * h, states[i])
                for j in range(self.s):
                    block = -h * self.A[i, j] * jac_i
                    if i == j:
                        block = eye + block
                    big[i * n : (i + 1) * n, j * n : (j + 1) * n] = block

            delta = self._safe_linear_solve(big, -res.reshape(-1)).reshape(self.s, n)
            K = self._apply_damping(
                K,
                delta,
                res_norm,
                lambda trial: residual(trial)[1],
            )

        self.stats.newton_failures += 1
        raise FloatingPointError("global Newton method did not converge")

    def _solve_dirk(self, t: float, y: Array, h: float) -> Array:
        n = y.size
        eye = np.eye(n)
        f0 = self.system.f(t, y)
        K = np.zeros((self.s, n), dtype=float)

        for i in range(self.s):
            ti = t + self.c[i] * h
            base = y.copy()
            for j in range(i):
                base += h * self.A[i, j] * K[j]
            ki = self.system.f(ti, base + h * self.A[i, i] * f0)

            def residual(stage_value: Array) -> Tuple[Array, float]:
                state = base + h * self.A[i, i] * stage_value
                res = stage_value - self.system.f(ti, state)
                return res, float(np.linalg.norm(res, ord=np.inf))

            for _ in range(self.newton_maxiter):
                res, res_norm = residual(ki)
                self.stats.newton_iterations += 1
                if res_norm < self.newton_tol:
                    K[i] = ki
                    break

                jac = self.system.jac(ti, base + h * self.A[i, i] * ki)
                delta = self._safe_linear_solve(eye - h * self.A[i, i] * jac, -res)
                ki = self._apply_damping(ki, delta, res_norm, lambda trial: residual(trial)[1])
            else:
                sol = root(
                    lambda z: residual(np.asarray(z, dtype=float))[0],
                    ki,
                    jac=lambda z: eye - h * self.A[i, i] * self.system.jac(ti, base + h * self.A[i, i] * z),
                    method="hybr",
                )
                if not sol.success:
                    self.stats.newton_failures += 1
                    raise FloatingPointError(f"stage {i + 1} root fallback failed: {sol.message}")
                K[i] = np.asarray(sol.x, dtype=float)

        return K

    def solve_stages(self, t: float, y: Array, h: float) -> Array:
        return self._solve_dirk(t, y, h) if self.is_dirk else self._solve_full(t, y, h)

    def step(self, t: float, y: Array, h: float) -> Array:
        y_new = y + h * (self.b @ self.solve_stages(t, y, h))
        if not np.all(np.isfinite(y_new)):
            raise FloatingPointError("non-finite state produced")
        return y_new

    def solve(self, t_span: Tuple[float, float], y0: Array, h: float) -> Dict[str, Array]:
        t0, tf = map(float, t_span)
        t = t0
        y = np.array(y0, dtype=float)
        ts = [t]
        ys = [y.copy()]
        self.stats = SolveStats()

        while t < tf - 1e-15:
            queue = [min(h, tf - t)]
            while queue:
                h_step = queue.pop(0)
                try:
                    y_full = self.step(t, y, h_step)
                    if h_step > 2.0 * self.min_step:
                        y_half = self.step(t, y, 0.5 * h_step)
                        y_half = self.step(t + 0.5 * h_step, y_half, 0.5 * h_step)
                        scale = self.adaptive_atol + self.adaptive_rtol * max(
                            float(np.linalg.norm(y, ord=np.inf)),
                            float(np.linalg.norm(y_half, ord=np.inf)),
                        )
                        err = float(
                            np.linalg.norm((y_half - y_full) / max(2**self.order - 1, 1), ord=np.inf) / scale
                        )
                        if not np.isfinite(err) or err > 1.0:
                            raise FloatingPointError("local error too large")
                        y_new = y_half
                    else:
                        y_new = y_full
                except Exception as exc:
                    if h_step <= self.min_step:
                        raise RuntimeError(f"step became too small near t={t:.6g}: {exc}") from exc
                    half = 0.5 * h_step
                    queue = [half, half] + queue
                    self.stats.rejected_substeps += 1
                    continue

                t += h_step
                y = y_new
                ts.append(t)
                ys.append(y.copy())
                self.stats.steps += 1

        return {"t": np.array(ts), "y": np.vstack(ys), "stats": self.stats}

    def stability_function(self, z: Array) -> Array:
        z = np.asarray(z, dtype=complex)
        ident = np.eye(self.s, dtype=complex)
        e = np.ones(self.s, dtype=complex)
        out = np.empty_like(z, dtype=complex)
        for idx, zi in np.ndenumerate(z):
            out[idx] = 1.0 + zi * (self.b @ np.linalg.solve(ident - zi * self.A, e))
        return out


class RadauIIA1(ImplicitRKBase):
    def __init__(self, system: ODESystem):
        super().__init__(system, "Radau IIA order 1")

    def butcher_tableau(self) -> Tuple[Array, Array, Array, int]:
        return np.array([[1.0]]), np.array([1.0]), np.array([1.0]), 1


class RadauIIA3(ImplicitRKBase):
    def __init__(self, system: ODESystem):
        super().__init__(system, "Radau IIA order 3")

    def butcher_tableau(self) -> Tuple[Array, Array, Array, int]:
        return (
            np.array([[5.0 / 12.0, -1.0 / 12.0], [3.0 / 4.0, 1.0 / 4.0]]),
            np.array([3.0 / 4.0, 1.0 / 4.0]),
            np.array([1.0 / 3.0, 1.0]),
            3,
        )


class RadauIIA5(ImplicitRKBase):
    def __init__(self, system: ODESystem):
        super().__init__(system, "Radau IIA order 5")

    def butcher_tableau(self) -> Tuple[Array, Array, Array, int]:
        sq6 = math.sqrt(6.0)
        return (
            np.array(
                [
                    [(88.0 - 7.0 * sq6) / 360.0, (296.0 - 169.0 * sq6) / 1800.0, (-2.0 + 3.0 * sq6) / 225.0],
                    [(296.0 + 169.0 * sq6) / 1800.0, (88.0 + 7.0 * sq6) / 360.0, (-2.0 - 3.0 * sq6) / 225.0],
                    [(16.0 - sq6) / 36.0, (16.0 + sq6) / 36.0, 1.0 / 9.0],
                ]
            ),
            np.array([(16.0 - sq6) / 36.0, (16.0 + sq6) / 36.0, 1.0 / 9.0]),
            np.array([(4.0 - sq6) / 10.0, (4.0 + sq6) / 10.0, 1.0]),
            5,
        )


class SDIRK2(ImplicitRKBase):
    def __init__(self, system: ODESystem):
        super().__init__(system, "SDIRK order 2")

    def butcher_tableau(self) -> Tuple[Array, Array, Array, int]:
        gamma = 1.0 - math.sqrt(2.0) / 2.0
        return (
            np.array([[gamma, 0.0], [1.0 - gamma, gamma]]),
            np.array([1.0 - gamma, gamma]),
            np.array([gamma, 1.0]),
            2,
        )


class LibrarySolver(BaseSolver):
    def __init__(self, system: ODESystem, rtol: float = 1e-9, atol: float = 1e-11):
        super().__init__(system, "SciPy Radau baseline")
        self.rtol = rtol
        self.atol = atol

    def solve(self, t_span: Tuple[float, float], y0: Array, h: float) -> Dict[str, Array]:
        t0, tf = map(float, t_span)
        grid = np.arange(t0, tf + 0.25 * h, h, dtype=float)
        grid = grid[grid <= tf + 1e-12]
        if grid.size == 0 or grid[-1] < tf:
            grid = np.append(grid, tf)
        sol = solve_ivp(
            self.system.f,
            t_span=t_span,
            y0=y0,
            method="Radau",
            t_eval=grid,
            jac=self.system.jac,
            rtol=self.rtol,
            atol=self.atol,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        self.stats = SolveStats(steps=max(int(sol.nfev), 0))
        return {"t": sol.t.copy(), "y": sol.y.T.copy(), "stats": self.stats}


def build_methods(system: ODESystem) -> List[ImplicitRKBase]:
    methods: List[ImplicitRKBase] = [RadauIIA1(system), RadauIIA3(system), RadauIIA5(system), SDIRK2(system)]
    methods[0].adaptive_rtol = 1e-3
    return methods


def compare_with_baseline(
    solvers: Sequence[BaseSolver],
    baseline: BaseSolver,
    t_span: Tuple[float, float],
    y0: Array,
    h_values: Iterable[float],
) -> Dict[str, List[Dict[str, float]]]:
    table: Dict[str, List[Dict[str, float]]] = {solver.name: [] for solver in solvers}
    for h in h_values:
        base = baseline.solve(t_span, y0, float(h))
        base_x = base["y"][:, 0]
        for solver in solvers:
            res = solver.solve(t_span, y0, float(h))
            x_interp = np.interp(base["t"], res["t"], res["y"][:, 0])
            table[solver.name].append(
                {
                    "h": float(h),
                    "max_error_x": float(np.max(np.abs(x_interp - base_x))),
                    "steps": float(res["stats"].steps),
                    "newton_iterations": float(res["stats"].newton_iterations),
                    "newton_failures": float(res["stats"].newton_failures),
                    "rejected_substeps": float(res["stats"].rejected_substeps),
                }
            )
    return table


def save_solution_plots(results: Dict[str, Dict[str, Array]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(11, 6))
    for name, data in results.items():
        plt.plot(data["t"], data["y"][:, 0], label=name)
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.title("Rayleigh equation: coordinate x(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "solution_x.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 6))
    for name, data in results.items():
        plt.plot(data["t"], data["y"][:, 1], label=name)
    plt.xlabel("t")
    plt.ylabel("v(t)")
    plt.title("Rayleigh equation: velocity v(t) = x'(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "solution_v.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 7))
    for name, data in results.items():
        plt.plot(data["y"][:, 0], data["y"][:, 1], label=name)
    plt.xlabel("x")
    plt.ylabel("v")
    plt.title("Rayleigh equation: phase portrait on [0, 1000]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "phase_portrait.png", dpi=180)
    plt.close()


def save_error_plot(error_table: Dict[str, List[Dict[str, float]]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    for name, rows in error_table.items():
        plt.loglog([row["h"] for row in rows], [row["max_error_x"] for row in rows], marker="o", label=name)
    plt.xlabel("step h")
    plt.ylabel("max error in x vs SciPy Radau")
    plt.title("Comparison with SciPy Radau baseline")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "error_vs_step.png", dpi=180)
    plt.close()


def save_stability_plots(solvers: Iterable[ImplicitRKBase], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.linspace(-20.0, 5.0, 320)
    y = np.linspace(-15.0, 15.0, 320)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    for solver in solvers:
        M = np.abs(solver.stability_function(Z))
        plt.figure(figsize=(7.5, 6.5))
        plt.contourf(X, Y, M <= 1.0, levels=[-0.5, 0.5, 1.5], alpha=0.8)
        plt.contour(X, Y, M, levels=[1.0], linewidths=1.5)
        plt.axhline(0.0, linewidth=0.8)
        plt.axvline(0.0, linewidth=0.8)
        plt.xlabel("Re(z)")
        plt.ylabel("Im(z)")
        plt.title(f"Stability region: {solver.name}")
        plt.tight_layout()
        plt.savefig(out_dir / f"stability_region_{solver.name.lower().replace(' ', '_')}.png", dpi=180)
        plt.close()


def format_report(
    error_table: Dict[str, List[Dict[str, float]]],
    mu: float,
    y0: Array,
    t_span: Tuple[float, float],
    solution_step: float,
    comparison_steps: Sequence[float],
) -> str:
    lines = [
        "Rayleigh equation assignment report",
        f"mu = {mu}",
        f"Initial state: x(0) = {y0[0]}, v(0) = {y0[1]}",
        f"Integration interval: [{t_span[0]}, {t_span[1]}]",
        f"Solution plot step = {solution_step}",
        f"Comparison steps = {list(comparison_steps)}",
        "",
        "Comparison with SciPy Radau baseline:",
    ]
    for name, rows in error_table.items():
        lines.append(name)
        for row in rows:
            lines.append(
                f"  h={row['h']:>4g} | max_error_x={row['max_error_x']:.6e} | "
                f"steps={int(row['steps']):6d} | newton_iters={int(row['newton_iterations']):6d} | "
                f"newton_fails={int(row['newton_failures']):4d} | rejects={int(row['rejected_substeps']):4d}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_report(report: str, out_file: Path) -> None:
    out_file.write_text(report, encoding="utf-8")


def run_assignment(
    out_dir: Path,
    *,
    mu: float = 1000.0,
    y0: Optional[Array] = None,
    t_span: Tuple[float, float] = (0.0, 1000.0),
    solution_step: float = 0.5,
    comparison_steps: Sequence[float] = (2.0, 1.0, 0.5),
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if y0 is None:
        y0 = np.array([0.0, 1e-3], dtype=float)

    system = RayleighSystem(mu=mu)
    methods = build_methods(system)
    baseline = LibrarySolver(system)

    solution_results: Dict[str, Dict[str, Array]] = {baseline.name: baseline.solve(t_span, y0, solution_step)}
    for method in methods:
        solution_results[method.name] = method.solve(t_span, y0, solution_step)

    error_table = compare_with_baseline(methods, LibrarySolver(system), t_span, y0, comparison_steps)

    save_solution_plots(solution_results, out_dir)
    save_error_plot(error_table, out_dir)
    save_stability_plots(methods, out_dir)

    report = format_report(error_table, mu, y0, t_span, solution_step, comparison_steps)
    write_report(report, out_dir / "report.txt")

    return {
        "system": system,
        "methods": methods,
        "solutions": solution_results,
        "error_table": error_table,
        "report": report,
        "out_dir": out_dir,
    }
