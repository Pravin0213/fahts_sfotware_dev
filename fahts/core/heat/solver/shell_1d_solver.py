"""
Crank-Nicolson (θ=1/2) transient heat solver for shell/plate elements.

Governing equation (through thickness, x = 0 outer, x = t inner):
    M · Ṫ + K · T = Q(T)

Boundary conditions:
    x = 0 (outer/fire-exposed):
        q = ε·σ·(T_fire_K⁴ − T_K⁴)  +  h_conv·(T_fire − T)  [+  q_prescribed]
    x = thickness (inner):
        ∂T/∂n = 0   (adiabatic)

Crank-Nicolson incremental form (SINTEF FAHTS Eq. 3.2.30):
    A · ΔTi = B

    A = Ki + (2/Δt) · Mi
    B = Qi - K_{i-1} · T_{i-1} + M_{i-1} · Ṫ_{i-1}

    Ti    = T_{i-1} + ΔTi
    Ṫi    = (2/Δt) · ΔTi − Ṫ_{i-1}

Initial rate (t=0):
    Ṫ0 = M0⁻¹ · (Q0 − K0 · T0)
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from fahts.core.heat.section_mesh.shell_mesh import ShellMesh1D
from fahts.core.model.material import SteelMaterial

_SIGMA = 5.67e-8  # Stefan-Boltzmann [W/(m²·K⁴)]


class Shell1DSolver:
    """
    Crank-Nicolson 1-D FEM heat solver for a shell cross-section.

    Args:
        mesh:             ShellMesh1D from ShellMesher.
        material:         SteelMaterial — provides k(T), cp(T), rho.
        fire_temp:        Callable(t [s]) → T_fire [°C].
        epsilon_m:        Resultant emissivity; 0.0 when using q_prescribed.
        h_conv:           Convective coefficient [W/(m²·K)].
        T0:               Initial uniform nodal temperature [°C] (default 20.0).
        q_prescribed_fn:  Optional Callable(t) → prescribed flux [W/m²] (RadiationBall).
    """

    def __init__(
        self,
        mesh: ShellMesh1D,
        material: SteelMaterial,
        fire_temp: Callable[[float], float],
        epsilon_m: float,
        h_conv: float,
        T0: float = 20.0,
        q_prescribed_fn: Callable[[float], float] | None = None,
    ) -> None:
        self._mesh = mesh
        self._mat = material
        self._fire_temp = fire_temp
        self._eps = epsilon_m
        self._h_conv = h_conv
        self._T0 = T0
        self._q_fn: Callable[[float], float] = (
            q_prescribed_fn if q_prescribed_fn is not None else lambda _t: 0.0
        )

        # CN state — initialised lazily on first step() call or explicitly by run()
        self._K_prev: np.ndarray | None = None
        self._M_prev: np.ndarray | None = None
        self._T_dot_prev: np.ndarray | None = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _assemble_step(
        self, T_prev: np.ndarray, t: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Assemble K_i (conductivity + Robin convective stiffness), M_i (lumped
        capacitance), and Q_i (Robin load vector) at the given state.

        Returns:
            K_i: (n, n) dense — total stiffness including K_conv at outer node
            M_i: (n,) lumped capacitance
            Q_i: (n,) load vector (convection + radiation + prescribed flux)
        """
        T_mean = float(np.mean(T_prev))
        k   = self._mat.conductivity(T_mean)
        cp  = self._mat.specific_heat(T_mean)
        rho = self._mat.rho
        T_fire   = self._fire_temp(t)
        q_presc  = self._q_fn(t)

        K_i = _assemble_K_1d(self._mesh, k)
        M_i = _assemble_C_1d(self._mesh, rho, cp)
        Q_i = np.zeros(self._mesh.n_nodes)

        outer    = self._mesh.outer_node
        T_fire_K = T_fire + 273.15
        T_out_K  = T_prev[outer] + 273.15

        # Convective contribution: stiffness part → K_i; load part → Q_i
        K_i[outer, outer] += self._h_conv
        Q_i[outer]        += self._h_conv * T_fire

        # Radiative contribution (explicit)
        Q_i[outer] += self._eps * _SIGMA * (T_fire_K**4 - T_out_K**4)

        # Prescribed flux (RadiationBall / USERFLUX)
        if q_presc:
            Q_i[outer] += q_presc

        return K_i, M_i, Q_i

    def _init_rate(self, T0: np.ndarray, t0: float = 0.0) -> None:
        """
        Initialise CN state from the system equation at t=t0:
            Ṫ0 = M0⁻¹ · (Q0 − K0 · T0)
        """
        K0, M0, Q0 = self._assemble_step(T0, t0)
        residual    = Q0 - K0 @ T0
        T_dot0      = residual / M0  # M0 is lumped diagonal
        self._K_prev     = K0.copy()
        self._M_prev     = M0.copy()
        self._T_dot_prev = T_dot0

    # ── Public API ────────────────────────────────────────────────────────────

    def step(self, T_prev: np.ndarray, dt: float, t: float) -> np.ndarray:
        """
        One Crank-Nicolson step ending at time t.

        Args:
            T_prev: Nodal temperatures from previous step [°C], shape (n_nodes,).
            dt:     Time step size [s].
            t:      Current time (end of step) [s].

        Returns:
            T_new: Updated nodal temperatures [°C], shape (n_nodes,).
        """
        # Lazy initialisation: first call assumes T_prev is the t=0 state
        if self._K_prev is None:
            self._init_rate(T_prev, t0=0.0)

        K_i, M_i, Q_i = self._assemble_step(T_prev, t)

        # CN system: A · ΔT = B
        two_over_dt = 2.0 / dt
        A = K_i + np.diag(two_over_dt * M_i)
        B = Q_i - self._K_prev @ T_prev + self._M_prev * self._T_dot_prev

        dT        = np.linalg.solve(A, B)
        T_new     = T_prev + dT
        T_dot_new = two_over_dt * dT - self._T_dot_prev

        # Carry state forward
        self._K_prev     = K_i.copy()
        self._M_prev     = M_i.copy()
        self._T_dot_prev = T_dot_new

        return T_new

    def run(
        self,
        t_end: float,
        dt: float,
        output_dt: float | None = None,
        callback: Callable[[float, np.ndarray], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Full transient simulation from t = 0 to t = t_end.

        Args:
            t_end:     Total simulation time [s].
            dt:        Time step [s].
            output_dt: Output interval [s]. Defaults to dt.
            callback:  Optional callback(t, T) called at every time step.

        Returns:
            times:     (n_out,) array of output times [s], starting at 0.
            T_history: (n_out, n_nodes) nodal temperatures [°C].
        """
        n = self._mesh.n_nodes
        T = np.full(n, self._T0, dtype=float)
        out_dt    = output_dt if output_dt is not None else dt
        out_every = max(1, round(out_dt / dt))

        # Initialise CN state from t=0 (reset so run() is repeatable)
        self._K_prev = None
        self._init_rate(T, t0=0.0)

        out_times: list[float] = [0.0]
        out_T: list[np.ndarray] = [T.copy()]

        n_steps = max(1, round(t_end / dt))
        t = 0.0

        for step_i in range(n_steps):
            dt_step = min(dt, t_end - t)
            if dt_step <= 0.0:
                break
            t += dt_step
            T = self.step(T, dt_step, t)

            if callback is not None:
                callback(t, T)

            if (step_i + 1) % out_every == 0 or step_i == n_steps - 1:
                out_times.append(t)
                out_T.append(T.copy())

        return np.array(out_times), np.array(out_T)


# ── 1-D assembly helpers ──────────────────────────────────────────────────────

def _assemble_K_1d(mesh: ShellMesh1D, k: float) -> np.ndarray:
    """Assemble (n_nodes, n_nodes) conductivity matrix for 1-D rod FEM."""
    n = mesh.n_nodes
    K = np.zeros((n, n))
    for a, b in mesh.segments:
        a, b = int(a), int(b)
        L = abs(float(mesh.nodes[b]) - float(mesh.nodes[a]))
        K[a, a] += k / L
        K[a, b] -= k / L
        K[b, a] -= k / L
        K[b, b] += k / L
    return K


def _assemble_C_1d(mesh: ShellMesh1D, rho: float, cp: float) -> np.ndarray:
    """Assemble (n_nodes,) lumped capacitance vector for 1-D rod FEM."""
    C = np.zeros(mesh.n_nodes)
    for a, b in mesh.segments:
        a, b = int(a), int(b)
        L = abs(float(mesh.nodes[b]) - float(mesh.nodes[a]))
        C[a] += rho * cp * L * 0.5
        C[b] += rho * cp * L * 0.5
    return C
