import math
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class BaseRayleighSolver(ABC):
    def __init__(
        self,
        mu=1000.0,
        t0=0.0,
        T=1000.0,
        x0=0.0,
        v0=0.001,
        h=1e-3,
        newton_tol=1e-10,
        newton_maxiter=1200,
        sample_stride=1000,
    ):
        self.mu = float(mu)
        self.t0 = float(t0)
        self.T = float(T)
        self.y0 = np.array([x0, v0], dtype=float)
        self.h = float(h)
        self.newton_tol = float(newton_tol)
        self.newton_maxiter = int(newton_maxiter)
        self.sample_stride = int(sample_stride)

    def f(self, y):
        x, v = y
        return np.array([v, self.mu * (1.0 - v * v) * v - x], dtype=float)

    def jac(self, y):
        _, v = y
        return np.array([
            [0.0, 1.0],
            [-1.0, self.mu * (1.0 - 3.0 * v * v)],
        ], dtype=float)

    def _newton_stages(self, y_n, h, A, c): # Y_i = y_n + h * sum_j a_ij * f(Y_j)
        s = len(c)                          # y_{n+1} = y_n + h * sum_i b_i f(Y_i)
        dim = len(y_n)                      
        Y = np.tile(y_n, (s, 1))

        for _ in range(self.newton_maxiter):
            F = np.array([self.f(Y[i]) for i in range(s)]) # F_j = f(Y_j)
            Jf = np.array([self.jac(Y[i]) for i in range(s)])

            G = np.zeros((s, dim), dtype=float)
            for i in range(s): # G_i(Y) = Y_i - y_n - h * sum_j a_ij f(Y_j)
                rhs = y_n.copy()
                for j in range(s):
                    rhs += h * A[i, j] * F[j]
                G[i] = Y[i] - rhs

            g = G.reshape(s * dim) 
            if np.linalg.norm(g, ord=np.inf) < self.newton_tol: # ||G(Y)|| ≈ 0
                return Y

            J = np.zeros((s * dim, s * dim), dtype=float)
            I = np.eye(dim)

            for i in range(s):
                for j in range(s):
                    block = -h * A[i, j] * Jf[j]
                    if i == j:
                        block += I
                    r0 = i * dim
                    c0 = j * dim
                    J[r0:r0 + dim, c0:c0 + dim] = block
                                           # f(x) ≈ f(x_k) + f’(x_k)(x - x_k); 0 ≈ f(x_k) + f’(x_k)(x - x_k); x = x_k - f(x_k) / f’(x_k)
            delta = np.linalg.solve(J, -g) # G(Y^k) + J(Y^k) delta ≈ 0
            Y += delta.reshape(s, dim)     # J(Y^k) delta = -G(Y^k)

            if np.linalg.norm(delta, ord=np.inf) < self.newton_tol:
                return Y

        raise RuntimeError(f"Newton did not converge for {self.__class__.__name__}")

    def _solve_rk(self, A, b, c, keep_history=False):
        h = self.h
        n_steps = int(round((self.T - self.t0) / h))
        t = self.t0
        y = self.y0.copy()

        hist_t, hist_y = None, None
        if keep_history:
            hist_t = [t]
            hist_y = [y.copy()]

        for n in range(1, n_steps + 1):
            Y = self._newton_stages(y, h, A, c) # Y_i = y_n + h * sum_j a_ij * f(Y_j)
            F = np.array([self.f(Y[i]) for i in range(len(c))]) # F_i = f(Y_i)
            y = y + h * np.sum(b[:, None] * F, axis=0) # y_{n+1} = y_n + h * sum_i b_i * f(Y_i)
            t = self.t0 + n * h

            if keep_history and (n % self.sample_stride == 0 or n == n_steps):
                hist_t.append(t)
                hist_y.append(y.copy())

        if keep_history:
            return np.array(hist_t), np.array(hist_y)
        return t, y

    @abstractmethod
    def solve(self, keep_history=False):
        raise NotImplementedError


class ImplicitEulerSolver(BaseRayleighSolver):
    def solve(self, keep_history=False):
        A = np.array([[1.0]], dtype=float)
        b = np.array([1.0], dtype=float)
        c = np.array([1.0], dtype=float)
        return self._solve_rk(A, b, c, keep_history=keep_history)


class RadauIIA3Solver(BaseRayleighSolver):
    def solve(self, keep_history=False):
        A = np.array([
            [5.0 / 12.0, -1.0 / 12.0],
            [3.0 / 4.0, 1.0 / 4.0],
        ], dtype=float)
        b = np.array([3.0 / 4.0, 1.0 / 4.0], dtype=float)
        c = np.array([1.0 / 3.0, 1.0], dtype=float)
        return self._solve_rk(A, b, c, keep_history=keep_history)


class RadauIIA5Solver(BaseRayleighSolver):
    def solve(self, keep_history=False):
        sqrt6 = math.sqrt(6.0)
        A = np.array([
            [(88.0 - 7.0 * sqrt6) / 360.0, (296.0 - 169.0 * sqrt6) / 1800.0, (-2.0 + 3.0 * sqrt6) / 225.0],
            [(296.0 + 169.0 * sqrt6) / 1800.0, (88.0 + 7.0 * sqrt6) / 360.0, (-2.0 - 3.0 * sqrt6) / 225.0],
            [(16.0 - sqrt6) / 36.0, (16.0 + sqrt6) / 36.0, 1.0 / 9.0],
        ], dtype=float)
        b = np.array([(16.0 - sqrt6) / 36.0, (16.0 + sqrt6) / 36.0, 1.0 / 9.0], dtype=float)
        c = np.array([(4.0 - sqrt6) / 10.0, (4.0 + sqrt6) / 10.0, 1.0], dtype=float)
        return self._solve_rk(A, b, c, keep_history=keep_history)


class SDIRK2Solver(BaseRayleighSolver):
    def _newton_single_stage(self, y_init, y_base, h_gamma):
        y = y_init.copy()
        I = np.eye(len(y))

        for _ in range(self.newton_maxiter):
            g = y - y_base - h_gamma * self.f(y)
            if np.linalg.norm(g, ord=np.inf) < self.newton_tol:
                return y

            J = I - h_gamma * self.jac(y)
            delta = np.linalg.solve(J, -g)
            y += delta

            if np.linalg.norm(delta, ord=np.inf) < self.newton_tol:
                return y

        raise RuntimeError(f"Newton did not converge for {self.__class__.__name__}")

    def solve(self, keep_history=False):
        sqrt2 = math.sqrt(2.0)
        gamma = (2.0 - sqrt2) / 2.0
        a21 = 1.0 - gamma
        b1 = 0.5
        b2 = 0.5

        h = self.h
        n_steps = int(round((self.T - self.t0) / h))
        t = self.t0
        y = self.y0.copy()

        hist_t, hist_y = None, None
        if keep_history:
            hist_t = [t]
            hist_y = [y.copy()]

        for n in range(1, n_steps + 1):
            Y1 = self._newton_single_stage(y.copy(), y, h * gamma)
            K1 = self.f(Y1)

            base2 = y + h * a21 * K1
            Y2 = self._newton_single_stage(Y1.copy(), base2, h * gamma)
            K2 = self.f(Y2)

            y = y + h * (b1 * K1 + b2 * K2)
            t = self.t0 + n * h

            if keep_history and (n % self.sample_stride == 0 or n == n_steps):
                hist_t.append(t)
                hist_y.append(y.copy())

        if keep_history:
            return np.array(hist_t), np.array(hist_y)
        return t, y


def get_butcher_tableau(name):
    if name == "ImplicitEuler":
        A = np.array([[1.0]], dtype=float)
        b = np.array([1.0], dtype=float)
    elif name == "RadauIIA3":
        A = np.array([[5.0 / 12.0, -1.0 / 12.0], [3.0 / 4.0, 1.0 / 4.0]], dtype=float)
        b = np.array([3.0 / 4.0, 1.0 / 4.0], dtype=float)
    elif name == "RadauIIA5":
        sqrt6 = math.sqrt(6.0)
        A = np.array([
            [(88.0 - 7.0 * sqrt6) / 360.0, (296.0 - 169.0 * sqrt6) / 1800.0, (-2.0 + 3.0 * sqrt6) / 225.0],
            [(296.0 + 169.0 * sqrt6) / 1800.0, (88.0 + 7.0 * sqrt6) / 360.0, (-2.0 - 3.0 * sqrt6) / 225.0],
            [(16.0 - sqrt6) / 36.0, (16.0 + sqrt6) / 36.0, 1.0 / 9.0],
        ], dtype=float)
        b = np.array([(16.0 - sqrt6) / 36.0, (16.0 + sqrt6) / 36.0, 1.0 / 9.0], dtype=float)
    elif name == "SDIRK2":
        sqrt2 = math.sqrt(2.0)
        gamma = (2.0 - sqrt2) / 2.0
        A = np.array([[gamma, 0.0], [1.0 - gamma, gamma]], dtype=float)
        b = np.array([0.5, 0.5], dtype=float)
    else:
        raise ValueError(name)
    return A, b


def stability_function(A, b, z):
    e = np.ones(len(b), dtype=complex)
    I = np.eye(len(b), dtype=complex)
    return 1.0 + z * (b.astype(complex) @ np.linalg.solve(I - z * A.astype(complex), e))


def plot_stability_regions(method_names, xlim=(-12, 4), ylim=(-10, 10), points=500):
    xx = np.linspace(xlim[0], xlim[1], points)
    yy = np.linspace(ylim[0], ylim[1], points)
    X, Y = np.meshgrid(xx, yy)
    Z = X + 1j * Y

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axes = axes.ravel()

    for ax, name in zip(axes, method_names):
        A, b = get_butcher_tableau(name)
        R = np.vectorize(lambda z: stability_function(A, b, z))(Z)
        M = np.abs(R)

        ax.contourf(X, Y, (M <= 1.0).astype(float), levels=[-0.5, 0.5, 1.5], alpha=0.7)
        ax.contour(X, Y, M, levels=[1.0], linewidths=1.5)
        ax.axhline(0.0, linewidth=0.8)
        ax.axvline(0.0, linewidth=0.8)
        ax.set_title(name)
        ax.set_xlabel("Re(z)")
        ax.set_ylabel("Im(z)")

    return fig


def run_solver(task):
    solver_cls, params = task
    solver = solver_cls(**params)
    t_final, y_final = solver.solve(keep_history=False)
    return {
        "method": solver_cls.__name__,
        "t_final": t_final,
        "x_T": float(y_final[0]),
        "v_T": float(y_final[1]),
    }


def run_all_parallel(params):
    solver_classes = [
        ImplicitEulerSolver,
        RadauIIA3Solver,
        RadauIIA5Solver,
        SDIRK2Solver,
    ]
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_solver, (cls, params)) for cls in solver_classes]
        results = [future.result() for future in as_completed(futures)]
    order = {cls.__name__: i for i, cls in enumerate(solver_classes)}
    results.sort(key=lambda r: order[r["method"]])
    return pd.DataFrame(results)


def compare_with_reference(df, ref_method="RadauIIA5Solver"):
    ref_row = df.loc[df["method"] == ref_method].iloc[0]
    out = df.copy()
    out["|x_T - x_ref|"] = np.abs(out["x_T"] - ref_row["x_T"])
    out["|v_T - v_ref|"] = np.abs(out["v_T"] - ref_row["v_T"])
    return out


def plot_sampled_trajectories(params, methods=None):
    if methods is None:
        methods = [
            ImplicitEulerSolver,
            RadauIIA3Solver,
            RadauIIA5Solver,
            SDIRK2Solver,
        ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)

    for solver_cls in methods:
        solver = solver_cls(**params)
        t, y = solver.solve(keep_history=True)
        axes[0].plot(t, y[:, 0], label=solver_cls.__name__)
        axes[1].plot(t, y[:, 1], label=solver_cls.__name__)

    axes[0].set_ylabel("x(t)")
    axes[1].set_ylabel("v(t)")
    axes[1].set_xlabel("t")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    return fig



def plot_solutions_vs_time_combined(params, methods=None):
    if methods is None:
        methods = [
            ImplicitEulerSolver,
            RadauIIA3Solver,
            RadauIIA5Solver,
            SDIRK2Solver,
        ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)

    for solver_cls in methods:
        solver = solver_cls(**params)
        t, y = solver.solve(keep_history=True)
        axes[0].plot(t, y[:, 0], label=solver_cls.__name__)
        axes[1].plot(t, y[:, 1], label=solver_cls.__name__)

    axes[0].set_title("Решение x(t)")
    axes[0].set_ylabel("x(t)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Решение v(t)")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("v(t)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    return fig

if __name__ == "__main__":
    params = dict(mu=1000.0, t0=0.0, T=1000.0, x0=0.0, v0=0.001, h=1e-3, sample_stride=2000)

    df = run_all_parallel(params)
    print("Final values:")
    print(df.to_string(index=False))
    print()

    cmp_df = compare_with_reference(df, ref_method="RadauIIA5Solver")
    print("Comparison vs RadauIIA5Solver:")
    print(cmp_df.to_string(index=False))
    print()

    plot_sampled_trajectories(params)
    plt.show()

    plot_stability_regions(["ImplicitEuler", "RadauIIA3", "RadauIIA5", "SDIRK2"])
    plt.show()





