"""
Unit tests for Tasks 3.3 and 3.4 — Quad4 element matrices, global assembly,
and Robin boundary condition.

Analytical reference (unit square [0,1]×[0,1], k=1, ρ·cₚ=1):

  K_analytical = (1/6) × [[4,−1,−2,−1],
                            [−1, 4,−1,−2],
                            [−2,−1, 4,−1],
                            [−1,−2,−1, 4]]
  (rows sum to 0; one zero eigenvalue for uniform temperature)

  C_consistent = (1/36) × [[4,2,1,2],
                             [2,4,2,1],
                             [1,2,4,2],
                             [2,1,2,4]]
  (sum = 1 = ρ·cₚ·A; row sums = 1/4 each)

  C_lumped = (1/4) × I₄  (each node gets A/4 = 1/4 of thermal mass)
"""
import pytest
import numpy as np
import scipy.sparse as sp
from fahts.core.heat.solver.fem_2d_section import (
    quad4_conductivity_matrix,
    quad4_capacity_matrix,
    assemble_K,
    assemble_C_lumped,
    add_robin_bc,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _unit_square() -> np.ndarray:
    """Unit square [0,1]×[0,1], CCW node order (y, z)."""
    return np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def _rect(dy: float, dz: float) -> np.ndarray:
    """Axis-aligned rectangle with given dimensions."""
    return np.array([[0.0, 0.0], [dy, 0.0], [dy, dz], [0.0, dz]])


# Known analytical K for unit square (k=1)
_K_ref = np.array([
    [ 4, -1, -2, -1],
    [-1,  4, -1, -2],
    [-2, -1,  4, -1],
    [-1, -2, -1,  4],
], dtype=float) / 6.0

# Known analytical C_consistent for unit square (rho=1, cp=1)
_C_ref_consistent = np.array([
    [4, 2, 1, 2],
    [2, 4, 2, 1],
    [1, 2, 4, 2],
    [2, 1, 2, 4],
], dtype=float) / 36.0

# Known analytical C_lumped for unit square (rho=1, cp=1)
_C_ref_lumped = np.diag([0.25, 0.25, 0.25, 0.25])


# ── Conductivity matrix ───────────────────────────────────────────────────────

class TestConductivityMatrix:
    def test_shape(self):
        K = quad4_conductivity_matrix(_unit_square(), k=1.0)
        assert K.shape == (4, 4)

    def test_unit_square_analytical(self):
        K = quad4_conductivity_matrix(_unit_square(), k=1.0)
        np.testing.assert_allclose(K, _K_ref, atol=1e-12)

    def test_symmetry(self):
        K = quad4_conductivity_matrix(_unit_square(), k=1.0)
        np.testing.assert_allclose(K, K.T, atol=1e-14)

    def test_row_sums_zero(self):
        """K·[1,1,1,1]^T = 0 (uniform T → zero flux)."""
        K = quad4_conductivity_matrix(_unit_square(), k=1.0)
        np.testing.assert_allclose(K.sum(axis=1), 0.0, atol=1e-13)

    def test_scales_with_k(self):
        K1 = quad4_conductivity_matrix(_unit_square(), k=1.0)
        K5 = quad4_conductivity_matrix(_unit_square(), k=5.0)
        np.testing.assert_allclose(K5, 5.0 * K1, atol=1e-12)

    def test_positive_semi_definite(self):
        """All eigenvalues ≥ 0; exactly one zero eigenvalue."""
        K = quad4_conductivity_matrix(_unit_square(), k=1.0)
        eigvals = np.linalg.eigvalsh(K)
        assert eigvals.min() >= -1e-12
        assert np.sum(np.abs(eigvals) < 1e-10) == 1  # one zero

    def test_rectangular_element_row_sums_zero(self):
        K = quad4_conductivity_matrix(_rect(0.2, 0.008), k=50.0)
        np.testing.assert_allclose(K.sum(axis=1), 0.0, atol=1e-10)

    def test_rectangular_element_symmetry(self):
        K = quad4_conductivity_matrix(_rect(0.2, 0.008), k=50.0)
        np.testing.assert_allclose(K, K.T, atol=1e-14)

    def test_rectangular_element_positive_semi_definite(self):
        K = quad4_conductivity_matrix(_rect(0.05, 0.008), k=50.0)
        eigvals = np.linalg.eigvalsh(K)
        assert eigvals.min() >= -1e-10

    def test_area_scales_k(self):
        """Doubling element width should proportionally change K values."""
        K1 = quad4_conductivity_matrix(_rect(1.0, 1.0), k=1.0)
        K2 = quad4_conductivity_matrix(_rect(2.0, 1.0), k=1.0)
        # Both should have row sums = 0
        np.testing.assert_allclose(K2.sum(axis=1), 0.0, atol=1e-12)
        # K is not simply proportional to area for non-square elements
        # but the (0,0) entry should change predictably
        assert K2[0, 0] != K1[0, 0]

    def test_diagonal_dominant_or_equal(self):
        """Diagonal entries must be ≥ sum of absolute values of off-diagonals."""
        K = quad4_conductivity_matrix(_unit_square(), k=1.0)
        for i in range(4):
            off = sum(abs(K[i, j]) for j in range(4) if j != i)
            assert K[i, i] >= off - 1e-12


# ── Capacitance matrix ────────────────────────────────────────────────────────

class TestCapacityMatrix:
    def test_shape(self):
        C = quad4_capacity_matrix(_unit_square(), rho=1.0, cp=1.0)
        assert C.shape == (4, 4)

    def test_lumped_unit_square_analytical(self):
        C = quad4_capacity_matrix(_unit_square(), rho=1.0, cp=1.0, lumped=True)
        np.testing.assert_allclose(C, _C_ref_lumped, atol=1e-12)

    def test_consistent_unit_square_analytical(self):
        C = quad4_capacity_matrix(_unit_square(), rho=1.0, cp=1.0, lumped=False)
        np.testing.assert_allclose(C, _C_ref_consistent, atol=1e-12)

    def test_lumped_is_diagonal(self):
        C = quad4_capacity_matrix(_unit_square(), rho=7850.0, cp=600.0, lumped=True)
        off_diag = C - np.diag(np.diag(C))
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-12)

    def test_consistent_symmetry(self):
        C = quad4_capacity_matrix(_unit_square(), rho=1.0, cp=1.0, lumped=False)
        np.testing.assert_allclose(C, C.T, atol=1e-14)

    def test_lumped_symmetry(self):
        C = quad4_capacity_matrix(_unit_square(), rho=1.0, cp=1.0, lumped=True)
        np.testing.assert_allclose(C, C.T, atol=1e-14)

    def test_consistent_total_mass(self):
        """Sum of all entries = ρ·cₚ·A (total thermal mass of element)."""
        rho, cp = 7850.0, 500.0
        coords = _unit_square()
        A = 1.0  # area of unit square
        C = quad4_capacity_matrix(coords, rho=rho, cp=cp, lumped=False)
        np.testing.assert_allclose(C.sum(), rho * cp * A, rtol=1e-12)

    def test_lumped_total_mass(self):
        rho, cp = 7850.0, 500.0
        coords = _unit_square()
        A = 1.0
        C = quad4_capacity_matrix(coords, rho=rho, cp=cp, lumped=True)
        np.testing.assert_allclose(C.sum(), rho * cp * A, rtol=1e-12)

    def test_lumped_equal_diagonal_for_rectangle(self):
        """Rectangular element: all four lumped diagonal entries must be equal."""
        C = quad4_capacity_matrix(_rect(0.2, 0.008), rho=7850.0, cp=600.0, lumped=True)
        diag = np.diag(C)
        np.testing.assert_allclose(diag, diag[0], atol=1e-10)

    def test_scales_with_rho_cp(self):
        C1 = quad4_capacity_matrix(_unit_square(), rho=1.0, cp=1.0, lumped=False)
        C2 = quad4_capacity_matrix(_unit_square(), rho=2.0, cp=3.0, lumped=False)
        np.testing.assert_allclose(C2, 6.0 * C1, atol=1e-12)

    def test_consistent_positive_definite(self):
        """Consistent mass matrix must be positive definite."""
        C = quad4_capacity_matrix(_unit_square(), rho=7850.0, cp=500.0, lumped=False)
        eigvals = np.linalg.eigvalsh(C)
        assert eigvals.min() > 0

    def test_lumped_positive_definite(self):
        """Lumped mass matrix must have all positive diagonal entries."""
        C = quad4_capacity_matrix(_unit_square(), rho=7850.0, cp=500.0, lumped=True)
        assert np.diag(C).min() > 0

    def test_rectangular_mass_conservation(self):
        """ρ·cₚ·A must be conserved for any rectangle."""
        rho, cp = 7850.0, 500.0
        dy, dz = 0.04, 0.008
        A = dy * dz
        coords = _rect(dy, dz)
        for lump in [True, False]:
            C = quad4_capacity_matrix(coords, rho=rho, cp=cp, lumped=lump)
            np.testing.assert_allclose(C.sum(), rho * cp * A, rtol=1e-12,
                                       err_msg=f"lumped={lump}")


# ── Integration with SectionMesh from Task 3.2 ───────────────────────────────

class TestIntegrationWithMesh:
    """Verify element matrices work correctly on actual BoxMesher output."""

    def setup_method(self):
        from fahts.core.model.section import BoxSection
        from fahts.core.heat.section_mesh import BoxMesher
        sec = BoxSection(sid=1, H=0.20, W=0.20, T_side=0.008, T_bot=0.008, T_top=0.008)
        self.mesh = BoxMesher(sec, elem_size=0.04, n_layers=1).build()
        self.k = 50.0    # approx. k for steel at 20°C
        self.rho = 7850.0
        self.cp = 440.0

    def test_all_element_K_are_symmetric(self):
        for q in self.mesh.quads:
            coords = self.mesh.nodes[q]
            K = quad4_conductivity_matrix(coords, self.k)
            np.testing.assert_allclose(K, K.T, atol=1e-10,
                                       err_msg=f"K not symmetric for quad {q}")

    def test_all_element_K_row_sums_zero(self):
        for q in self.mesh.quads:
            coords = self.mesh.nodes[q]
            K = quad4_conductivity_matrix(coords, self.k)
            np.testing.assert_allclose(K.sum(axis=1), 0.0, atol=1e-9)

    def test_all_element_C_mass_conservation(self):
        """Each element's total mass = ρ·cₚ·element_area."""
        for q in self.mesh.quads:
            coords = self.mesh.nodes[q]
            A_elem = 0.5 * abs(
                (coords[2, 0] - coords[0, 0]) * (coords[3, 1] - coords[1, 1])
                - (coords[3, 0] - coords[1, 0]) * (coords[2, 1] - coords[0, 1])
            )
            C = quad4_capacity_matrix(coords, self.rho, self.cp, lumped=False)
            np.testing.assert_allclose(C.sum(), self.rho * self.cp * A_elem,
                                       rtol=1e-10)

    def test_global_mass_from_elements_equals_section_mass(self):
        """
        Sum of all element masses (lumped diagonal) should equal ρ·cₚ·A_steel.
        This is an assembly consistency check without needing the full assembler.
        """
        # Accumulate per-node lumped mass into global array
        n_nodes = self.mesh.n_nodes
        global_lump = np.zeros(n_nodes)
        for q in self.mesh.quads:
            coords = self.mesh.nodes[q]
            C = quad4_capacity_matrix(coords, self.rho, self.cp, lumped=True)
            for local_i, global_i in enumerate(q):
                global_lump[global_i] += C[local_i, local_i]

        expected = self.rho * self.cp * self.mesh.steel_area
        np.testing.assert_allclose(global_lump.sum(), expected, rtol=1e-9)


# ── Task 3.4: Global assembly ─────────────────────────────────────────────────

def _box_mesh():
    from fahts.core.model.section import BoxSection
    from fahts.core.heat.section_mesh import BoxMesher
    sec = BoxSection(sid=1, H=0.20, W=0.20, T_side=0.008, T_bot=0.008, T_top=0.008)
    return BoxMesher(sec, elem_size=0.04, n_layers=1).build()


class TestAssembleK:
    def setup_method(self):
        self.mesh = _box_mesh()
        self.k = 50.0

    def test_shape(self):
        K = assemble_K(self.mesh, self.k)
        n = self.mesh.n_nodes
        assert K.shape == (n, n)

    def test_is_sparse(self):
        K = assemble_K(self.mesh, self.k)
        assert sp.issparse(K)

    def test_symmetry(self):
        K = assemble_K(self.mesh, self.k)
        diff = (K - K.T).toarray()
        np.testing.assert_allclose(diff, 0.0, atol=1e-12)

    def test_row_sums_zero(self):
        """K·[1,…,1]^T = 0 before BC application."""
        K = assemble_K(self.mesh, self.k)
        ones = np.ones(self.mesh.n_nodes)
        np.testing.assert_allclose(K @ ones, 0.0, atol=1e-10)

    def test_positive_semi_definite(self):
        """All eigenvalues ≥ 0; exactly one zero eigenvalue for connected mesh."""
        K = assemble_K(self.mesh, self.k)
        eigvals = np.linalg.eigvalsh(K.toarray())
        assert eigvals.min() >= -1e-8
        assert np.sum(eigvals < 1e-8) == 1

    def test_scales_with_k(self):
        K1 = assemble_K(self.mesh, 1.0)
        K5 = assemble_K(self.mesh, 5.0)
        np.testing.assert_allclose(K5.toarray(), 5.0 * K1.toarray(), atol=1e-12)


class TestAssembleCLumped:
    def setup_method(self):
        self.mesh = _box_mesh()
        self.rho = 7850.0
        self.cp = 440.0

    def test_shape(self):
        C = assemble_C_lumped(self.mesh, self.rho, self.cp)
        assert C.shape == (self.mesh.n_nodes,)

    def test_all_positive(self):
        C = assemble_C_lumped(self.mesh, self.rho, self.cp)
        assert np.all(C > 0)

    def test_total_mass_conservation(self):
        """Sum of diagonal = ρ·cₚ·A_steel (per unit beam length)."""
        C = assemble_C_lumped(self.mesh, self.rho, self.cp)
        expected = self.rho * self.cp * self.mesh.steel_area
        np.testing.assert_allclose(C.sum(), expected, rtol=1e-9)

    def test_scales_with_rho_cp(self):
        C1 = assemble_C_lumped(self.mesh, 1.0, 1.0)
        C2 = assemble_C_lumped(self.mesh, 2.0, 3.0)
        np.testing.assert_allclose(C2, 6.0 * C1, atol=1e-12)


# ── Task 3.4: Robin BC ────────────────────────────────────────────────────────

class TestAddRobinBC:
    """Tests for add_robin_bc() on a single 1-m horizontal edge."""

    def _single_edge(self):
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        edge_pairs = np.array([[0, 1]])
        return nodes, edge_pairs

    def _run(self, T_fire, T_prev_val, epsilon_m=0.5, h_conv=25.0):
        nodes, edge_pairs = self._single_edge()
        K = sp.lil_matrix((2, 2))
        f = np.zeros(2)
        T_prev = np.full(2, T_prev_val)
        add_robin_bc(K, f, edge_pairs, nodes, T_fire, T_prev, epsilon_m, h_conv)
        return K.toarray(), f

    def test_modifies_K(self):
        K, _ = self._run(T_fire=800.0, T_prev_val=20.0)
        assert np.any(K != 0.0)

    def test_modifies_f(self):
        _, f = self._run(T_fire=800.0, T_prev_val=20.0)
        assert np.any(f != 0.0)

    def test_equilibrium_at_fire_temperature(self):
        """
        When T_prev = T_fire everywhere, net nodal flux must be zero.
        Verifies: f - K_robin · T_fire = 0 (both convection and radiation).
        """
        T_fire = 800.0
        K, f = self._run(T_fire=T_fire, T_prev_val=T_fire)
        net = f - K @ np.full(2, T_fire)
        np.testing.assert_allclose(net, 0.0, atol=1e-6)

    def test_heating_cold_steel(self):
        """Cold steel (20°C) exposed to hot fire (900°C) must be heated (f > 0)."""
        _, f = self._run(T_fire=900.0, T_prev_val=20.0)
        assert np.all(f > 0.0)

    def test_convective_K_structure(self):
        """
        Convective contribution: K = h·L/6·[[2,1],[1,2]] for L=1m.
        Radiation does not modify K (purely explicit).
        """
        h = 25.0
        K, _ = self._run(T_fire=800.0, T_prev_val=20.0, h_conv=h, epsilon_m=0.0)
        K_expected = h / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
        np.testing.assert_allclose(K, K_expected, atol=1e-12)

    def test_zero_epsilon_no_radiation(self):
        """ε_m = 0 → radiative term vanishes, only convection contributes."""
        T_fire = 900.0
        K0, f0 = self._run(T_fire=T_fire, T_prev_val=20.0, epsilon_m=0.0)
        K1, f1 = self._run(T_fire=T_fire, T_prev_val=20.0, epsilon_m=0.5)
        # K unchanged by radiation
        np.testing.assert_allclose(K0, K1, atol=1e-12)
        # f is larger with radiation (more heat input)
        assert np.all(f1 > f0)

    def test_convective_f_value(self):
        """f_conv = h·T_fire·L/2·[1,1] for a single edge of L=1m."""
        h, T_fire = 25.0, 800.0
        _, f = self._run(T_fire=T_fire, T_prev_val=T_fire, epsilon_m=0.0, h_conv=h)
        # At equilibrium (T_prev=T_fire, ε=0): f = K·T_fire → net=0 ✓
        # But f itself = h·T_fire·L/2 per node
        # → verify f = K @ [T_fire, T_fire]
        K, f = self._run(T_fire=T_fire, T_prev_val=T_fire, epsilon_m=0.0, h_conv=h)
        np.testing.assert_allclose(f, K @ np.array([T_fire, T_fire]), atol=1e-10)

    def test_symmetric_nodes_give_equal_f(self):
        """Uniform T_prev on a uniform edge → both nodes get equal flux."""
        _, f = self._run(T_fire=900.0, T_prev_val=20.0)
        np.testing.assert_allclose(f[0], f[1], atol=1e-10)

    def test_scales_with_h_conv(self):
        """Doubling h_conv doubles the convective contribution to K and f."""
        K1, f1 = self._run(T_fire=900.0, T_prev_val=20.0, h_conv=25.0, epsilon_m=0.0)
        K2, f2 = self._run(T_fire=900.0, T_prev_val=20.0, h_conv=50.0, epsilon_m=0.0)
        np.testing.assert_allclose(K2, 2.0 * K1, atol=1e-12)
        np.testing.assert_allclose(f2, 2.0 * f1, atol=1e-12)


class TestRobinBCOnMesh:
    """Integration test: Robin BC on the full BOX section mesh."""

    def setup_method(self):
        self.mesh = _box_mesh()
        self.k = 50.0
        self.rho = 7850.0
        self.cp = 440.0

    def test_equilibrium_uniform_fire_temperature(self):
        """
        If all nodes are at T_fire, applying Robin BC then computing
        (K + K_robin)·T_fire - f should give zero at every boundary node.
        """
        T_fire = 1000.0
        T_prev = np.full(self.mesh.n_nodes, T_fire)

        K = assemble_K(self.mesh, self.k).tolil()
        f = np.zeros(self.mesh.n_nodes)
        add_robin_bc(K, f, self.mesh.outer_edge_pairs, self.mesh.nodes,
                     T_fire=T_fire, T_prev=T_prev, epsilon_m=0.7, h_conv=25.0)

        residual = K.toarray() @ T_prev - f
        # Only boundary nodes get BC; interior nodes have residual=0 from K·T=0
        np.testing.assert_allclose(residual, 0.0, atol=1e-4)

    def test_heating_increases_f(self):
        """Cold mesh (20°C) exposed to fire: all boundary nodes must receive positive flux."""
        T_fire = 900.0
        T_prev = np.full(self.mesh.n_nodes, 20.0)

        K = assemble_K(self.mesh, self.k).tolil()
        f = np.zeros(self.mesh.n_nodes)
        add_robin_bc(K, f, self.mesh.outer_edge_pairs, self.mesh.nodes,
                     T_fire=T_fire, T_prev=T_prev, epsilon_m=0.7, h_conv=25.0)

        # Boundary nodes must have positive load
        boundary_nodes = np.unique(self.mesh.outer_edge_pairs)
        assert np.all(f[boundary_nodes] > 0.0)

    def test_inner_nodes_have_zero_f(self):
        """Interior nodes not on any exposed boundary receive zero direct flux."""
        T_fire = 900.0
        T_prev = np.full(self.mesh.n_nodes, 20.0)

        f = np.zeros(self.mesh.n_nodes)
        K = assemble_K(self.mesh, self.k).tolil()
        add_robin_bc(K, f, self.mesh.outer_edge_pairs, self.mesh.nodes,
                     T_fire=T_fire, T_prev=T_prev, epsilon_m=0.7, h_conv=25.0)

        outer_nodes = set(self.mesh.outer_edge_pairs.ravel().tolist())
        interior = [i for i in range(self.mesh.n_nodes) if i not in outer_nodes]
        np.testing.assert_allclose(f[interior], 0.0, atol=1e-12)
