"""Tests for Phase 4.8 screenshot / animation methods on SceneManager."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fahts.renderer.scene_manager import SceneManager
from fahts.core.results.temperature_field import TemperatureField


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sm():
    mgr = SceneManager(off_screen=True)
    yield mgr
    mgr.close()


@pytest.fixture(scope="module")
def sm_with_model():
    from fahts.core.io.usfos_reader import read_usfos_fem
    fem_path = Path(__file__).parents[1] / "examples" / "models" / "model_file.fem"
    if not fem_path.exists():
        pytest.skip("model_file.fem not found")
    mgr = SceneManager(off_screen=True)
    model = read_usfos_fem(fem_path)
    mgr.load_model(model)
    yield mgr
    mgr.close()


def _make_tf(n_elems: int = 2, n_steps: int = 4) -> TemperatureField:
    rng = np.random.default_rng(7)
    times = np.linspace(0.0, 3600.0, n_steps)
    eids = list(range(1, n_elems + 1))
    T_cen = rng.uniform(20.0, 700.0, (n_steps, n_elems))
    return TemperatureField(times=times, element_ids=eids, T_centroid=T_cen)


# ── screenshot ────────────────────────────────────────────────────────────────

class TestScreenshot:
    def test_creates_png(self, sm, tmp_path):
        out = tmp_path / "scene.png"
        sm.screenshot(out)
        assert out.exists()

    def test_accepts_string_path(self, sm, tmp_path):
        out = tmp_path / "scene.png"
        sm.screenshot(str(out))
        assert out.exists()

    def test_nonzero_file_size(self, sm, tmp_path):
        out = tmp_path / "scene.png"
        sm.screenshot(out)
        assert out.stat().st_size > 0


# ── save_animation ────────────────────────────────────────────────────────────

class TestSaveAnimation:
    def test_gif_created(self, sm, tmp_path):
        tf = _make_tf()
        out = tmp_path / "anim.gif"
        n = sm.save_animation(tf, out, fps=5)
        assert out.exists()
        assert n == len(tf.times)

    def test_gif_nonzero_size(self, sm, tmp_path):
        tf = _make_tf()
        out = tmp_path / "anim.gif"
        sm.save_animation(tf, out, fps=5)
        assert out.stat().st_size > 0

    def test_returns_frame_count(self, sm, tmp_path):
        n_steps = 6
        tf = _make_tf(n_steps=n_steps)
        out = tmp_path / "anim.gif"
        result = sm.save_animation(tf, out, fps=5)
        assert result == n_steps

    def test_progress_callback_called(self, sm, tmp_path):
        n_steps = 3
        tf = _make_tf(n_steps=n_steps)
        out = tmp_path / "anim.gif"
        calls: list[tuple[int, int]] = []
        sm.save_animation(tf, out, fps=5, progress_callback=lambda c, t: calls.append((c, t)))
        assert len(calls) == n_steps
        assert calls[-1] == (n_steps, n_steps)

    def test_progress_callback_sequential(self, sm, tmp_path):
        n_steps = 4
        tf = _make_tf(n_steps=n_steps)
        out = tmp_path / "anim.gif"
        frames: list[int] = []
        sm.save_animation(tf, out, fps=5, progress_callback=lambda c, t: frames.append(c))
        assert frames == list(range(1, n_steps + 1))

    def test_accepts_string_path(self, sm, tmp_path):
        tf = _make_tf()
        out = tmp_path / "anim.gif"
        sm.save_animation(tf, str(out), fps=5)
        assert out.exists()

    def test_invalid_extension_raises_value_error(self, sm, tmp_path):
        tf = _make_tf()
        with pytest.raises(ValueError, match="Unsupported"):
            sm.save_animation(tf, tmp_path / "anim.xyz", fps=5)

    def test_mp4_without_ffmpeg_raises_import_error(self, sm, tmp_path):
        tf = _make_tf()
        out = tmp_path / "anim.mp4"
        try:
            import imageio_ffmpeg  # noqa: F401
            pytest.skip("imageio-ffmpeg installed — MP4 would succeed")
        except ImportError:
            with pytest.raises(ImportError, match="imageio-ffmpeg"):
                sm.save_animation(tf, out, fps=5)

    def test_gif_is_valid_gif_file(self, sm, tmp_path):
        """Check the GIF magic bytes in the output."""
        tf = _make_tf()
        out = tmp_path / "anim.gif"
        sm.save_animation(tf, out, fps=5)
        with out.open("rb") as fh:
            magic = fh.read(6)
        assert magic[:3] == b"GIF"

    def test_no_progress_callback_no_crash(self, sm, tmp_path):
        tf = _make_tf()
        out = tmp_path / "anim.gif"
        sm.save_animation(tf, out, fps=5, progress_callback=None)
        assert out.exists()

    def test_single_frame_animation(self, sm, tmp_path):
        tf = _make_tf(n_steps=1)
        out = tmp_path / "anim.gif"
        n = sm.save_animation(tf, out, fps=5)
        assert out.exists()
        assert n == 1
