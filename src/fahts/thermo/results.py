"""Flash result containers: ``Phase`` (one equilibrium phase) and ``FlashResult``.
"""

from __future__ import annotations


class Phase:
    """One equilibrium phase.  Molar properties per mol of this phase."""

    __slots__ = (
        "name",
        "beta",
        "x",
        "T",
        "P",
        "Z",
        "v",
        "M",
        "h",
        "u",
        "s",
        "cp",
        "cv",
        "w",
        "dPdT_v",
        "dPdv_T",
        "mu",
        "k",
    )

    def __init__(self, **kw):
        self.mu = None
        self.k = None
        for k_, v_ in kw.items():
            setattr(self, k_, v_)

    def copy(self, beta=None):
        p = Phase(**{s: getattr(self, s) for s in self.__slots__})
        if beta is not None:
            p.beta = beta
        return p

    @property
    def rho(self):  # kg/m3
        return self.M / self.v

    @property
    def rho_mol(self):  # mol/m3
        return 1.0 / self.v

    @property
    def h_mass(self):
        return self.h / self.M

    @property
    def u_mass(self):
        return self.u / self.M

    @property
    def s_mass(self):
        return self.s / self.M

    @property
    def cp_mass(self):
        return self.cp / self.M

    @property
    def cv_mass(self):
        return self.cv / self.M

    def __repr__(self):
        return (
            f"<Phase {self.name} beta={self.beta:.6g} rho={self.rho:.4g} kg/m3 "
            f"M={self.M * 1e3:.4g} g/mol Z={self.Z:.5g}>"
        )


class FlashResult:
    """Result of a flash.  beta = phase mole fractions of the feed; totals per mol feed."""

    def __init__(self, T, P, z, phases, kind="", K=None, names=None):
        self.T = T
        self.P = P
        self.z = z
        self.phases = phases
        self.kind = kind
        self.K = K
        self.names = names
        self.N = 1.0
        self.sigma = None
        self.info = {}

    def _get(self, name):
        for p in self.phases:
            if p.name == name:
                return p
        return None

    @property
    def vapour(self):
        return self._get("vapour")

    @property
    def liquid(self):
        return self._get("liquid")

    @property
    def aqueous(self):
        return self._get("aqueous")

    def beta_of(self, name):
        return sum(p.beta for p in self.phases if p.name == name)

    @property
    def beta_V(self):
        return self.beta_of("vapour")

    @property
    def beta_L(self):
        return self.beta_of("liquid")

    @property
    def beta_W(self):
        return self.beta_of("aqueous")

    def _tot(self, attr):
        return sum(p.beta * getattr(p, attr) for p in self.phases)

    @property
    def v(self):
        return self._tot("v")

    @property
    def h(self):
        return self._tot("h")

    @property
    def u(self):
        return self._tot("u")

    @property
    def s(self):
        return self._tot("s")

    @property
    def M(self):
        return self._tot("M")

    @property
    def rho(self):
        return self.M / self.v

    @property
    def phase_names(self):
        return tuple(p.name for p in self.phases)

    def mass_fractions(self):
        m = {p.name: p.beta * p.M for p in self.phases}
        tot = sum(m.values())
        return {k: v / tot for k, v in m.items()}

    def volume_fractions(self):
        vv = {p.name: p.beta * p.v for p in self.phases}
        tot = sum(vv.values())
        return {k: v / tot for k, v in vv.items()}

    def summary(self):
        s = [f"T={self.T:.3f} K  P={self.P / 1e5:.5g} bar  phases={self.phase_names}"]
        for p in self.phases:
            comp = " ".join(f"{n}={x:.4g}" for n, x in zip(self.names, p.x) if x > 1e-8)
            s.append(
                f"  {p.name:8s} beta={p.beta:.5f} rho={p.rho:9.3f} M={p.M * 1e3:8.3f} "
                f"Z={p.Z:.4f}  {comp}"
            )
        return "\n".join(s)

    def __repr__(self):
        return f"<FlashResult {self.phase_names} T={self.T:.2f} P={self.P:.5g}>"
