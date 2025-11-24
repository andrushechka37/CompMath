import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from abc import ABC, abstractmethod


class LotkaValterraProblem(ABC):
    def __init__(
        self,
        a: float, b: float, c: float, d: float,
        x0: float, y0: float,
        t: np.ndarray,
        phase_style: str = "k-",
    ) -> None:
        """
        Lotka–Volterra parameters:
        a — prey (x) natural growth rate,
        b — predation rate coefficient,
        c — predator (y) growth rate per consumed prey,
        d — predator natural death rate.
        x0, y0 — initial populations,
        t      — time grid.
        phase_style — style for phase portrait (right plot), e.g. 'k-' or 'ko'.
        """
        self.a_coef = float(a)
        self.b_coef = float(b)
        self.c_coef = float(c)
        self.d_coef = float(d)

        self.x0 = float(x0)
        self.y0 = float(y0)
        self.t = np.asarray(t, float)

        self.phase_style = phase_style 

        self.x: np.ndarray | None = None
        self.y: np.ndarray | None = None

    def get_right_side_system(self, x: float, y: float) -> tuple[float, float]:
        fx = self.a_coef * x - self.b_coef * x * y
        fy = self.c_coef * x * y - self.d_coef * y
        return fx, fy

    @abstractmethod
    def solve_system(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def solve(self) -> tuple[np.ndarray, np.ndarray]:
        self.x, self.y = self.solve_system()
        return self.x, self.y

    def plot(self, title: str = "") -> None:
        if self.x is None or self.y is None:
            raise RuntimeError("Call .solve() before .plot().")

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        axes[0].plot(self.t, self.x, 'r', label="Жертвы")
        axes[0].plot(self.t, self.y, 'b', label="Хищники")
        axes[0].set_xlabel("Время")
        axes[0].set_ylabel("Количество животных")
        axes[0].legend()

        axes[1].plot(self.x, self.y, self.phase_style)
        axes[1].set_xlabel("Количество жертв")
        axes[1].set_ylabel("Количество хищников")
        if title:
            axes[1].set_title(title)

        plt.tight_layout()
        plt.show()


# ---------- RK1 ----------

class RK1_LotkaValterraProblem(LotkaValterraProblem):
    def solve_system(self) -> tuple[np.ndarray, np.ndarray]:
        t = self.t
        x = np.zeros_like(t)
        y = np.zeros_like(t)
        x[0], y[0] = self.x0, self.y0

        for n in range(len(t) - 1):
            h = t[n + 1] - t[n]
            fx, fy = self.get_right_side_system(x[n], y[n])
            x[n + 1] = x[n] + h * fx
            y[n + 1] = y[n] + h * fy

        return x, y


# ---------- RK4 ----------

class RK4_LotkaValterraProblem(LotkaValterraProblem):
    def solve_system(self) -> tuple[np.ndarray, np.ndarray]:
        t = self.t
        x = np.zeros_like(t)
        y = np.zeros_like(t)
        x[0], y[0] = self.x0, self.y0

        for n in range(len(t) - 1):
            h = t[n + 1] - t[n]

            k1x, k1y = self.get_right_side_system(x[n], y[n])

            k2x, k2y = self.get_right_side_system(
                x[n] + 0.5 * h * k1x,
                y[n] + 0.5 * h * k1y,
            )

            k3x, k3y = self.get_right_side_system(
                x[n] + 0.5 * h * k2x,
                y[n] + 0.5 * h * k2y,
            )

            k4x, k4y = self.get_right_side_system(
                x[n] + h * k3x,
                y[n] + h * k3y,
            )

            x[n + 1] = x[n] + (h / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
            y[n + 1] = y[n] + (h / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)

        return x, y


class OdeintLotkaValterraProblem(LotkaValterraProblem):
    def _rhs_vec(self, XY, t):
        x, y = XY
        fx, fy = self.get_right_side_system(x, y)
        return [fx, fy]

    def solve_system(self) -> tuple[np.ndarray, np.ndarray]:
        XY = odeint(self._rhs_vec, [self.x0, self.y0], self.t)
        x = XY[:, 0]
        y = XY[:, 1]
        return x, y


# ---------- main ----------

if __name__ == "__main__":
    a, b, c, d = 0.4, 0.002, 0.001, 0.7
    x0, y0 = 600.0, 400.0
    t = np.linspace(0.0, 50.0, 101)

    ode_problem = OdeintLotkaValterraProblem(a, b, c, d, x0, y0, t, phase_style="ko")
    rk1_problem = RK1_LotkaValterraProblem(a, b, c, d, x0, y0, t, phase_style="ko")
    rk4_problem = RK4_LotkaValterraProblem(a, b, c, d, x0, y0, t, phase_style="ko")

    x_ode, y_ode   = ode_problem.solve()
    x_rk1, y_rk1   = rk1_problem.solve()
    x_rk4, y_rk4   = rk4_problem.solve()

    ode_problem.plot("odeint")
    rk1_problem.plot("RK1")
    rk4_problem.plot("RK4")

    err_rk1 = np.max(np.sqrt((x_rk1 - x_ode)**2 + (y_rk1 - y_ode)**2))
    err_rk4 = np.max(np.sqrt((x_rk4 - x_ode)**2 + (y_rk4 - y_ode)**2))
    print("\nОценка точности относительно решения odeint:")
    print(f"  RK1  ≈ {err_rk1:.3e}")
    print(f"  RK4  ≈ {err_rk4:.3e}")