"""
Unit tests for SteelMaterial thermal properties (EN 1993-1-2 Annex C / §3.4.1).

Task 3.1 acceptance tests:
  - conductivity(T) follows two-branch formula (§3.4.1.3)
  - specific_heat(T) follows four-branch formula with 735°C peak (§3.4.1.2)
  - density(T) returns constant rho
  - Inputs below 20°C and above 1200°C are clamped
  - MATERIAL_STANDARD constant labels the source correctly
"""
import pytest
from fahts.core.model.material import MATERIAL_STANDARD, SteelMaterial

S355 = SteelMaterial(mid=1, E=210e9, nu=0.3, fy=355e6, rho=7850.0, alpha_T=12e-6, name="S355")


# ── conductivity k(T) ────────────────────────────────────────────────────────

class TestConductivity:
    def test_at_20C(self):
        # k = 54 - 3.33e-2 * 20 = 53.334
        assert abs(S355.conductivity(20.0) - 53.334) < 0.01

    def test_at_400C(self):
        # k = 54 - 3.33e-2 * 400 = 40.68
        assert abs(S355.conductivity(400.0) - 40.68) < 0.01

    def test_at_800C_boundary(self):
        # T=800 is still in lower branch: k = 54 - 26.64 = 27.36
        k = S355.conductivity(800.0)
        assert abs(k - 27.36) < 0.05

    def test_at_900C(self):
        assert abs(S355.conductivity(900.0) - 27.3) < 0.01

    def test_at_1200C(self):
        assert abs(S355.conductivity(1200.0) - 27.3) < 0.01

    def test_decreases_with_temperature_below_800(self):
        k_low = S355.conductivity(100.0)
        k_high = S355.conductivity(700.0)
        assert k_low > k_high

    def test_clamp_below_20(self):
        assert S355.conductivity(0.0) == S355.conductivity(20.0)

    def test_clamp_above_1200(self):
        assert S355.conductivity(1500.0) == S355.conductivity(1200.0)


# ── specific heat cp(T) ───────────────────────────────────────────────────────

class TestSpecificHeat:
    def test_at_20C(self):
        # cp = 425 + 7.73e-1*20 - 1.69e-3*400 + 2.22e-6*8000
        expected = 425.0 + 7.73e-1 * 20 - 1.69e-3 * 20**2 + 2.22e-6 * 20**3
        assert abs(S355.specific_heat(20.0) - expected) < 1.0

    def test_at_300C(self):
        expected = 425.0 + 7.73e-1 * 300 - 1.69e-3 * 300**2 + 2.22e-6 * 300**3
        assert abs(S355.specific_heat(300.0) - expected) < 1.0

    def test_at_600C_branch_boundary(self):
        expected = 425.0 + 7.73e-1 * 600 - 1.69e-3 * 600**2 + 2.22e-6 * 600**3
        assert abs(S355.specific_heat(600.0) - expected) < 1.0

    def test_at_700C_high_cp(self):
        # 600 < T ≤ 735: cp = 666 + 13002/(738-T)
        expected = 666.0 + 13002.0 / (738.0 - 700.0)
        assert abs(S355.specific_heat(700.0) - expected) < 1.0

    def test_at_735C_peak_region(self):
        # Just below 735 should give a very large value (phase transformation)
        cp_734 = S355.specific_heat(734.0)
        cp_20 = S355.specific_heat(20.0)
        assert cp_734 > cp_20 * 5  # peak is dramatically higher

    def test_at_800C(self):
        # 735 < T ≤ 900: cp = 545 + 17820/(T-731)
        expected = 545.0 + 17820.0 / (800.0 - 731.0)
        assert abs(S355.specific_heat(800.0) - expected) < 1.0

    def test_at_1000C(self):
        # T > 900: cp = 650
        assert abs(S355.specific_heat(1000.0) - 650.0) < 0.01

    def test_at_1200C(self):
        assert abs(S355.specific_heat(1200.0) - 650.0) < 0.01

    def test_clamp_below_20(self):
        assert S355.specific_heat(0.0) == S355.specific_heat(20.0)

    def test_clamp_above_1200(self):
        assert S355.specific_heat(1500.0) == S355.specific_heat(1200.0)

    def test_positive_everywhere(self):
        temps = [20, 100, 300, 500, 600, 650, 700, 720, 734, 750, 800, 900, 1000, 1200]
        for T in temps:
            assert S355.specific_heat(T) > 0, f"cp must be positive at T={T}°C"


# ── density ───────────────────────────────────────────────────────────────────

class TestDensity:
    def test_constant_at_any_temperature(self):
        assert S355.density(20.0) == 7850.0
        assert S355.density(600.0) == 7850.0
        assert S355.density(1200.0) == 7850.0

    def test_default_argument(self):
        assert S355.density() == 7850.0


# ── MATERIAL_STANDARD constant ────────────────────────────────────────────────

class TestMaterialStandard:
    def test_constant_is_string(self):
        assert isinstance(MATERIAL_STANDARD, str)

    def test_constant_identifies_en1993(self):
        assert "EN1993" in MATERIAL_STANDARD or "EN 1993" in MATERIAL_STANDARD

    def test_constant_references_annex_c(self):
        assert "Annex C" in MATERIAL_STANDARD or "AnnexC" in MATERIAL_STANDARD

    def test_constant_exact_value(self):
        assert MATERIAL_STANDARD == "EN1993-1-2:2005 Annex C"

    def test_conductivity_at_20C_matches_ec3(self):
        # k = 54 - 3.33e-2 × 20 = 53.34 W/(m·K)  [EN 1993-1-2 §3.4.1.3]
        assert abs(S355.conductivity(20.0) - 53.334) < 0.01

    def test_specific_heat_at_20C_matches_ec3(self):
        # cp = 425 + 7.73e-1×20 - 1.69e-3×400 + 2.22e-6×8000 ≈ 439.83 J/(kg·K)
        expected = 425.0 + 7.73e-1 * 20 - 1.69e-3 * 20**2 + 2.22e-6 * 20**3
        assert abs(S355.specific_heat(20.0) - expected) < 0.01

    def test_density_matches_ec3(self):
        # EN 1993-1-2 §3.4.1.1: ρ = 7850 kg/m³ (constant)
        assert S355.density() == 7850.0
