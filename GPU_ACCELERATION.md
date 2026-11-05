# GPU acceleration of AudibleLight's convolution

This branch (`gpu-convolution`) adds PyTorch versions of AudibleLight's convolutions, which run on any device. The original NumPy/SciPy code is unchanged and is still the default. It also adds a benchmark that times AudibleLight and SpatialScaper together.

## What changed

| File | What |
| --- | --- |
| `audiblelight/synthesize_torch.py` | Torch ports of `time_invariant_convolution` (static events) and `time_variant_convolution` (moving events), plus `time_variant_convolution_batch` (many events in one call). |
| `audiblelight/synthesize.py` | `render_event_audio(..., device, dtype)` and `render_audio_for_all_scene_events(..., device, dtype)`. With a device set, all moving events of a microphone are batched together. `device=None` (default) means NumPy. |
| `audiblelight/core.py` | `Scene.generate(..., device=None, dtype="float32")` passes the choice through. |
| `tests/test_synthesize_torch.py` | Kernel, batch and full ray-traced scene parity vs NumPy (CPU, plus CUDA when available). |
| `scripts/benchmark/benchmark_spatial_simulation.py` | Kernel and pipeline benchmark for AudibleLight and SpatialScaper (IR generation vs convolution). |
| `scripts/benchmark/results_cpu_4threads.txt` | Results on the build machine (no GPU). |

### How the speed-up works

AudibleLight's moving-source convolution was copied from SpatialScaper: a Python loop over every output STFT frame. At AudibleLight's defaults (44.1 kHz, hop of 128 samples) that is ~345 iterations per second of audio, each touching every IR frame. The torch version computes the same sum as one FFT convolution per IR along the frame axis. It restricts each IR to the frames where its interpolation weight is non-zero and batches IRs and events together. Same algorithm as SpatialScaper's `spatialize_torch`, adapted to AudibleLight's `(channels, samples)` layout and separate `fft_size`/`win_size`.

### What was *not* ported

* **Ray tracing (`rlr` backend)** stays on CPU: `rlr-audio-propagation` is a compiled C++ library with no GPU entry point.
* **Augmentations** (`pedalboard`) and **audio decoding** (`librosa`) stay on CPU. They are per-event and I/O-bound.
* **`WorldStateShoebox`** is not implemented in this version, so it was not benchmarked.

### Verified results (no GPU here: torch on 4 CPU threads)

* **Parity:** float64 matches NumPy to ~1e-15 relative on the kernels and float32 to ~3e-7. Full ray-traced scenes, with static and moving events plus ambience, match to < 1e-6 (float64) and < 1e-4 (float32). The existing `tests/test_synthesize.py` still passes; its 4 video-plotter tests need `pyvista`.
* **Speed (CPU only):**

| Case | NumPy | Torch f32 (CPU) |
| --- | --- | --- |
| Moving event, 10 s, 40 ray-traced IRs (1.65 s each) | 331 s | 1.35 s (244×) |
| Moving event, 5 s, 10 IRs × 0.3 s | 9.2 s | 0.12 s (75×) |
| Static event, 10 s | 0.11 s | 0.03 s (4×) |
| Scene, SOFA backend (3 static + 2 moving), end to end | 5.7 s | 0.44 s (13×) |
| Scene, `rlr` ray tracing (3 static + 2 moving), end to end | 39.1 s | 38.5 s (1.01×): ray tracing takes 38 s |

## Usage

```python
scene.generate(output_dir="out", device="cuda")                   # float32
scene.generate(output_dir="out", device="cuda", dtype="float64")  # matches NumPy to ~1e-15
```

## Once you have a GPU machine

1. **Check the CUDA build of PyTorch:** `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"`. AudibleLight already depends on torch; make sure it is a CUDA wheel, not the CPU one.
2. **Run the parity tests.** The CUDA cases switch on automatically: `pytest tests/test_synthesize_torch.py`. Expect float64 errors ≤ 1e-10 on kernels and float32 ≤ 1e-4.
3. **Benchmark both tools.** With SpatialScaper's `gpu-spatialize-dataloader` branch installed next to this repo:
   ```bash
   python scripts/benchmark/benchmark_spatial_simulation.py --spatialscaper-repo ../spatialscaper --out results_gpu.json
   ```
   It adds `cuda-f32`/`cuda-f64` columns. Compare against `scripts/benchmark/results_cpu_4threads.txt`, and commit the new results file.
4. **Use float32** unless you need to reproduce NumPy outputs exactly. Consumer GPUs run float64 at 1/32–1/64 of float32 speed.
5. **Plan around ray tracing, not convolution.** With the `rlr` backend, a 30 s scene spent 38 s ray tracing and < 1 s convolving, so a GPU will not change end-to-end time much. To generate datasets faster:
   * run one scene per process across all CPU cores (`joblib`, already a dependency), with each process sending its convolutions to the GPU;
   * reuse one `WorldState` (mesh loaded once) for many scenes;
   * lower the ray count (`config.NUM_RAYS`) if the acoustic quality allows it, and check against a few reference IRs;
   * or ray-trace once, save the IRs to SOFA (`spatialscaper.sofa_utils.create_srir_sofa`), and reuse them through the SOFA backend or SpatialScaper. That makes scene generation convolution-bound, which is where the GPU helps (13× end to end even on CPU).
6. **Tune memory** with `max_chunk_elems` in `time_variant_convolution_batch` (default `2**26` complex elements per buffer, ~0.5 GB in complex64). Lower it on out-of-memory errors (long IRs, many capsules, many moving events); raise it if the GPU is under-utilised.
7. **Profile if the speed-up looks low** (`torch.profiler`, `nsys`). The kernels copy inputs from NumPy and return NumPy per call, to stay compatible with `Event.spatial_audio`. If transfers dominate, keep the scene's audio on the GPU end to end, which would be the next refactor.
8. **Validate on full data.** Generate a few scenes from `scripts/seld/generate_dataset.py` twice: once as-is, once with `scene.generate(..., device="cuda")`. The script has no device flag yet; add one. Then compare the WAVs. The comparison is meaningful only with the same IRs, so pass `ignore_cache=False` or reuse the scene.
9. **Optional:** upstream to `AudibleLight/AudibleLight` as a PR once GPU results look good.
