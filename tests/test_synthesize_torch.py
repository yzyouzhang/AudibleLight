#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Parity tests: PyTorch (CPU / GPU) convolution kernels vs. the NumPy reference in audiblelight/synthesize.py"""

import numpy as np
import pytest
import torch

import audiblelight.synthesize as syn
from audiblelight import config
from audiblelight import synthesize_torch as st
from tests import utils_tests

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
TOL = {torch.float64: 1e-10, torch.float32: 1e-4}
SR = config.SAMPLE_RATE


def _rel_err(ref, out):
    return np.abs(ref - out).max() / max(np.abs(ref).max(), 1e-30)


def _reference_time_variant(irs, audio, ir_times, fft_size, win_size, hop_size):
    """`syn.time_variant_convolution` without the Event object."""
    ir_spec = syn.stft(irs, fft_size, win_size, hop_size)
    audio_spec = syn.stft(audio, fft_size, win_size, hop_size)
    w_ir = syn.generate_interpolation_matrix(ir_times, SR, hop_size)
    spatial_stft = syn.perform_time_variant_convolution(audio_spec, ir_spec, w_ir)
    return syn.istft_overlap_synthesis(spatial_stft, fft_size, win_size, hop_size).T


def _case(rng, n_irs, seconds, n_ch=4, ir_len=8000):
    audio = rng.standard_normal(int(seconds * SR))
    irs = rng.standard_normal((n_ch, n_irs, ir_len)) * np.exp(-np.arange(ir_len) / 2000)
    return audio, irs, np.linspace(0, seconds, n_irs)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_time_invariant_parity(device, dtype):
    rng = np.random.default_rng(0)
    audio, irs, _ = _case(rng, 1, 2.0)
    ref = syn.time_invariant_convolution(audio, irs[:, 0].T)
    out = st.time_invariant_convolution_torch(audio, irs[:, 0].T, device, dtype)
    assert out.shape == ref.shape
    assert _rel_err(ref, out.cpu().numpy()) < TOL[dtype]


@pytest.mark.parametrize(
    "audio, ir, error_message",
    [
        (np.ones((5, 2)), np.ones((5, 4)), "Only mono input is supported"),
        (np.ones(5), np.ones(5), "Expected shape of IR should be"),
    ],
)
def test_time_invariant_invalid(audio, ir, error_message):
    with pytest.raises(ValueError, match=error_message):
        st.time_invariant_convolution_torch(audio, ir, "cpu")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize(
    "n_irs, seconds, n_ch, fft_size, win_size, hop_size",
    [
        (2, 1.0, 4, 512, 256, 128),  # AudibleLight defaults
        (5, 2.5, 4, 512, 256, 128),
        (12, 3.0, 9, 512, 256, 128),
        (3, 1.5, 4, 1024, 512, 256),  # SpatialScaper-style framing
        (4, 1.0, 1, 256, 256, 64),
    ],
)
def test_time_variant_parity(device, dtype, n_irs, seconds, n_ch, fft_size, win_size, hop_size):
    rng = np.random.default_rng(n_irs + n_ch)
    audio, irs, ir_times = _case(rng, n_irs, seconds, n_ch)
    ref = _reference_time_variant(irs, audio, ir_times, fft_size, win_size, hop_size)
    out = st.time_variant_convolution_torch(
        irs, audio, ir_times, SR, fft_size, win_size, hop_size, device=device, dtype=dtype
    ).cpu().numpy()
    assert out.shape == ref.shape
    assert _rel_err(ref, out) < TOL[dtype]


@pytest.mark.parametrize("device", DEVICES)
def test_time_variant_batch_parity(device):
    rng = np.random.default_rng(3)
    items = []
    for _ in range(6):
        n_irs = int(rng.integers(2, 15))
        audio, irs, ir_times = _case(rng, n_irs, rng.uniform(0.5, 3.0), ir_len=int(rng.choice([4000, 9000])))
        items.append(dict(audio=audio, irs=irs, ir_times=ir_times))
    outs = st.time_variant_convolution_batch(items, SR, device=device, dtype=torch.float64)
    for it, out in zip(items, outs):
        ref = _reference_time_variant(
            it["irs"], it["audio"], it["ir_times"], config.FFT_SIZE, config.WIN_SIZE, config.HOP_SIZE
        )
        assert out.shape == ref.shape
        assert _rel_err(ref, out.cpu().numpy()) < 1e-10


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_scene_render_parity(oyens_scene_no_overlap, device, dtype):
    """Ray-traced (RLR) scene: render with NumPy, then re-render the same IRs with torch."""
    scene = oyens_scene_no_overlap
    scene.clear_events()
    scene.clear_ambience()
    for _ in range(2):
        scene.add_event(event_type="static", duration=2.0)
        scene.add_event(
            event_type="moving",
            filepath=utils_tests.SOUNDEVENT_DIR / "music/000010.mp3",
            spatial_resolution=2,
            duration=2,
            spatial_velocity=1,
        )
    scene.add_ambience(noise="white")
    syn.validate_scene(scene)

    syn.render_audio_for_all_scene_events(scene)
    syn.generate_scene_audio_from_events(scene)
    ref_events = {a: ev.spatial_audio["mic000"].copy() for a, ev in scene.events.items()}
    ref_mix = scene.audio["mic000"].copy()

    for ev in scene.events.values():  # keep the IRs and loaded audio, drop the rendering
        ev.spatial_audio.clear()
    syn.render_audio_for_all_scene_events(scene, device=device, dtype=dtype)
    syn.generate_scene_audio_from_events(scene)

    tol = 1e-6 if dtype == "float64" else 1e-4  # reference FFTs float32 audio in single precision
    for alias, ev in scene.events.items():
        out = ev.spatial_audio["mic000"]
        assert out.shape == ref_events[alias].shape
        assert _rel_err(ref_events[alias], out) < tol, alias
    assert _rel_err(ref_mix, scene.audio["mic000"]) < tol
