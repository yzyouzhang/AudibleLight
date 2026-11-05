"""Speed + parity benchmark for spatial audio simulation: AudibleLight and SpatialScaper.

Compares the reference NumPy convolution of each tool against the PyTorch kernels
(``audiblelight.synthesize_torch`` / ``spatialscaper.spatialize_torch``) on every
available device, and times complete pipelines (IR generation vs convolution).

    python scripts/benchmark/benchmark_spatial_simulation.py --out results.json
    python scripts/benchmark/benchmark_spatial_simulation.py --quick

SpatialScaper sections are skipped if ``spatialscaper`` is not importable; its
pipeline section needs SpatialScaper's ``tests/synthetic_assets.py`` (pass
``--spatialscaper-repo``).
"""
import argparse
import json
import os
import sys
import tempfile
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from loguru import logger  # noqa: E402

logger.remove()

import audiblelight.synthesize as syn  # noqa: E402
from audiblelight import synthesize_torch as alt  # noqa: E402

try:
    from spatialscaper.spatialize import spatialize as ss_spatialize
    from spatialscaper.spatialize_torch import spatialize_torch as ss_spatialize_torch

    HAVE_SS = True
except ImportError:
    HAVE_SS = False

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
DTYPES = {"f64": torch.float64, "f32": torch.float32}
RESULTS = {"env": {}, "kernels": [], "pipelines": []}


def sync(dev):
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()


def timeit(fn, dev="cpu", repeats=3, warmup=True):
    if warmup:
        fn()
    sync(dev)
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        sync(dev)
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def rel_err(ref, out):
    return float(np.abs(ref - out).max() / max(np.abs(ref).max(), 1e-30))


def make_case(rng, sr, seconds, n_irs, ir_seconds, n_ch=4):
    n_ir = int(ir_seconds * sr)
    audio = rng.standard_normal(int(seconds * sr))
    irs = rng.standard_normal((n_ch, n_irs, n_ir)) * np.exp(-np.arange(n_ir) / (0.15 * sr))
    ir_times = np.linspace(0, seconds, n_irs) if n_irs > 1 else np.zeros(1)
    return audio, irs, ir_times


# ---------------------------------------------------------------- kernels
def al_numpy(audio, irs, ir_times, sr):
    if irs.shape[1] == 1:
        return syn.time_invariant_convolution(audio, irs[:, 0].T)
    ir_spec = syn.stft(irs)
    w = syn.generate_interpolation_matrix(ir_times, sr, 128)
    return syn.istft_overlap_synthesis(
        syn.perform_time_variant_convolution(syn.stft(audio), ir_spec, w)
    ).T


def al_torch(audio, irs, ir_times, sr, dev, dt):
    if irs.shape[1] == 1:
        return alt.time_invariant_convolution_torch(audio, irs[:, 0].T, dev, dt)
    return alt.time_variant_convolution_torch(irs, audio, ir_times, sr, device=dev, dtype=dt)


def bench_kernels(quick, repeats):
    rng = np.random.default_rng(0)
    # (tool, sr, audio s, n IRs, IR s, channels, what it models)
    cases = []
    if HAVE_SS:
        cases += [
            ("SpatialScaper", 24000, 10, 1, 0.3, 4, "static source, measured RIR"),
            ("SpatialScaper", 24000, 10, 2, 0.3, 4, "moving, 2 RIRs"),
            ("SpatialScaper", 24000, 10, 50, 0.3, 4, "moving, typical DCASE trajectory"),
            ("SpatialScaper", 24000, 10, 50, 1.0, 4, "moving, long RIRs"),
            ("SpatialScaper", 24000, 10, 50, 0.3, 32, "moving, em32 Eigenmike"),
        ]
    cases += [
        ("AudibleLight", 44100, 10, 1, 1.65, 4, "static, ray-traced IR"),
        ("AudibleLight", 44100, 5, 10, 0.3, 4, "moving, SOFA IRs"),
        ("AudibleLight", 44100, 5, 10, 1.65, 4, "moving, ray-traced IRs (2 Hz)"),
        ("AudibleLight", 44100, 10, 40, 1.65, 4, "moving, ray-traced IRs (4 Hz)"),
    ]
    if quick:
        cases = [c for c in cases if c[3] in (2, 10)]
    for tool, sr, secs, n_irs, ir_s, n_ch, what in cases:
        audio, irs, ir_times = make_case(rng, sr, secs, n_irs, ir_s, n_ch)
        if tool == "SpatialScaper":
            ref_fn = lambda: ss_spatialize(audio, irs, ir_times, sr)  # noqa: E731
            t_fn = lambda d, t: ss_spatialize_torch(  # noqa: E731
                audio, irs, ir_times, sr, device=d, dtype=t, return_tensor=True
            )
        else:
            ref_fn = lambda: al_numpy(audio, irs, ir_times, sr)  # noqa: E731
            t_fn = lambda d, t: al_torch(audio, irs, ir_times, sr, d, t)  # noqa: E731
        t0 = time.perf_counter()
        ref = ref_fn()
        first = time.perf_counter() - t0
        slow = first > 5
        t_np = first if slow else timeit(ref_fn, repeats=repeats, warmup=False)
        row = dict(tool=tool, sr=sr, audio_s=secs, n_irs=n_irs, ir_s=ir_s, channels=n_ch,
                   what=what, numpy_s=t_np, torch={})
        for dev in DEVICES:
            for dn, dt in DTYPES.items():
                out = t_fn(dev, dt)
                err = rel_err(ref, out.cpu().numpy())
                t = timeit(lambda: t_fn(dev, dt), dev, repeats)
                row["torch"][f"{dev}-{dn}"] = dict(seconds=t, speedup=t_np / t, rel_err=err)
        RESULTS["kernels"].append(row)
        cells = " | ".join(
            f"{k} {v['seconds']:.3f}s ({v['speedup']:.0f}x, err {v['rel_err']:.0e})"
            for k, v in row["torch"].items()
        )
        print(f"[kernel] {tool:13s} {secs:>3}s {n_irs:>3} IRs x {ir_s:.2f}s {n_ch:>2}ch "
              f"| numpy {t_np:7.3f}s | {cells}", flush=True)


# -------------------------------------------------------------- pipelines
def _al_scene(backend, n_static, n_moving, duration):
    from audiblelight.core import Scene
    from audiblelight.worldstate import WorldStateSOFA
    from tests import utils_tests

    kw = dict(duration=duration, fg_path=utils_tests.SOUNDEVENT_DIR, max_overlap=n_static + n_moving)
    if backend == "rlr":
        sc = Scene(backend="rlr", backend_kwargs=dict(
            waypoints_json=utils_tests.OYENS_WAYPOINTS_PATH, mesh=utils_tests.OYENS_PATH), **kw)
        sc.add_microphone(microphone_type="ambeovr")
    else:
        sc = Scene(backend=WorldStateSOFA(sofa=utils_tests.TEST_RESOURCES / "daga_foa.sofa",
                                          sample_rate=44100), **kw)
    for _ in range(n_static):
        sc.add_event(event_type="static", duration=5.0)
    for _ in range(n_moving):
        sc.add_event(event_type="moving", duration=5.0, spatial_resolution=2, spatial_velocity=1)
    return sc


def bench_audiblelight_pipeline(backend, n_static, n_moving, duration=30):
    np.random.seed(0)
    import random

    random.seed(0)
    sc = _al_scene(backend, n_static, n_moving, duration)
    t = time.perf_counter()
    sc.state.simulate()
    t_sim = time.perf_counter() - t
    for ev in sc.events.values():
        ev.load_audio()  # decode once so every backend below times only the convolution

    def render(device, dtype="float32"):
        for ev in sc.events.values():
            ev.spatial_audio.clear()
        syn.render_audio_for_all_scene_events(sc, device=device, dtype=dtype)
        sync(device or "cpu")
        syn.generate_scene_audio_from_events(sc)

    t_np = timeit(lambda: render(None), repeats=1, warmup=False)
    ref = sc.audio["mic000"].copy()
    row = dict(tool="AudibleLight", backend=backend, duration_s=duration,
               n_static=n_static, n_moving=n_moving,
               n_irs=int(sum(len(e) for e in sc.events.values())),
               ir_s=float(sc.state.get_irs()["mic000"].shape[-1] / 44100),
               ir_generation_s=t_sim, numpy_render_s=t_np, torch={})
    for dev in DEVICES:
        for dn in ("float64", "float32"):
            t = timeit(lambda: render(dev, dn), dev, repeats=1)
            row["torch"][f"{dev}-{dn[-2:]}"] = dict(
                render_s=t, render_speedup=t_np / t,
                total_speedup=(t_sim + t_np) / (t_sim + t),
                rel_err=rel_err(ref, sc.audio["mic000"]))
    RESULTS["pipelines"].append(row)
    print(f"[pipeline] AudibleLight/{backend}: {row['n_irs']} IRs ({row['ir_s']:.2f}s) | IR generation "
          f"{t_sim:.2f}s | numpy render {t_np:.2f}s | "
          + " | ".join(f"{k} {v['render_s']:.2f}s ({v['render_speedup']:.1f}x render, "
                       f"{v['total_speedup']:.2f}x total, err {v['rel_err']:.0e})" for k, v in row["torch"].items()),
          flush=True)


def bench_spatialscaper_pipeline(ss_repo, n_events=6, duration=30):
    sys.path.insert(0, os.path.join(ss_repo, "tests"))
    from synthetic_assets import make_assets

    import spatialscaper as ss
    from spatialscaper.dataloader import SpatialAudioSynthesizer, SpatialScaperDataset

    root = tempfile.mkdtemp()
    fg, rir = make_assets(root, rooms=("metu",), fmts=("mic",))
    captured = {}
    ss.core.save_output = lambda a, l, x, sr, labels, fmt="mic": captured.update(audio=x)

    def gen(device, dtype="float64"):
        ss.utils.set_seed(7)
        s = ss.Scaper(duration=duration, foreground_dir=fg, rir_dir=rir, room="metu",
                      use_room_ambient_noise=False, device=device, torch_dtype=dtype)
        s.add_background()
        for _ in range(n_events):
            s.add_event()
        s.generate("x", "x")
        return captured["audio"].copy()

    ref = gen(None)
    t_np = timeit(lambda: gen(None), repeats=1, warmup=False)
    row = dict(tool="SpatialScaper", backend="sofa (synthetic)", duration_s=duration,
               n_events=n_events, numpy_generate_s=t_np, torch={})
    for dev in DEVICES:
        for dn in ("float64", "float32"):
            t = timeit(lambda: gen(dev, dn), dev, repeats=1)
            row["torch"][f"{dev}-{dn[-2:]}"] = dict(generate_s=t, speedup=t_np / t,
                                                     rel_err=rel_err(ref, gen(dev, dn)))
    # dataloader: plan on CPU, synthesize a batch of 8 scenes in one go
    ds = SpatialScaperDataset(fg, rir, duration=duration, n_events=n_events, split=None,
                              num_scenes=8, seed=0, use_room_ambient_noise=False)
    t = time.perf_counter()
    plans = [ds[i] for i in range(8)]
    t_plan = (time.perf_counter() - t) / 8
    for dev in DEVICES:
        synth = SpatialAudioSynthesizer(device=dev, dtype=torch.float32)
        t = timeit(lambda: synth(plans), dev, repeats=1) / 8
        row["torch"][f"{dev}-f32"]["dataloader_plan_s_per_scene"] = t_plan
        row["torch"][f"{dev}-f32"]["dataloader_synth_s_per_scene"] = t
    RESULTS["pipelines"].append(row)
    print(f"[pipeline] SpatialScaper: numpy generate {t_np:.2f}s | " + " | ".join(
        f"{k} {v['generate_s']:.2f}s ({v['speedup']:.1f}x, err {v['rel_err']:.0e})"
        for k, v in row["torch"].items()), flush=True)
    for k, v in row["torch"].items():
        if "dataloader_synth_s_per_scene" in v:
            print(f"           dataloader {k}: plan {v['dataloader_plan_s_per_scene']:.2f}s/scene (CPU worker), "
                  f"synth {v['dataloader_synth_s_per_scene']:.3f}s/scene (batched)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-kernels", action="store_true")
    ap.add_argument("--skip-pipelines", action="store_true")
    ap.add_argument("--spatialscaper-repo", default=None)
    args = ap.parse_args()
    RESULTS["env"] = dict(torch=torch.__version__, cpu_threads=torch.get_num_threads(),
                          cpu_count=os.cpu_count(),
                          cuda=torch.cuda.get_device_name() if torch.cuda.is_available() else None)
    print(RESULTS["env"], flush=True)
    if not args.skip_kernels:
        bench_kernels(args.quick, args.repeats)
    if not args.skip_pipelines:
        bench_audiblelight_pipeline("sofa", 3, 2)
        bench_audiblelight_pipeline("rlr", 3, 2)
        if HAVE_SS and args.spatialscaper_repo:
            bench_spatialscaper_pipeline(args.spatialscaper_repo)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(RESULTS, f, indent=2)


if __name__ == "__main__":
    main()
