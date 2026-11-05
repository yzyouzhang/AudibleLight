"""PyTorch (GPU-capable) convolution kernels for :mod:`audiblelight.synthesize`.

The reference NumPy implementation of the time-variant convolution
(:func:`audiblelight.synthesize.perform_time_variant_convolution`) loops in Python
over every output STFT frame. At AudibleLight's defaults (44.1 kHz, hop of 128
samples) that is ~345 iterations per second of audio, each touching every IR frame.

This module computes the same quantity with batched FFTs instead:

    Y[t, f, c] = sum_l sum_k  S_ir[k, f, c, l] * W[t - k, l] * X[t - k, f]
               = sum_l ( S_ir[:, f, c, l]  *conv*  (W[:, l] * X[:, f]) )[t]

i.e. one linear convolution along the frame axis per IR. Each IR's interpolation
weight is only non-zero over a short span of frames (linear cross-fading), so each
IR is convolved only over that span; IRs are bucketed by span length and processed
in batches. :func:`time_variant_convolution_batch` lays several events out on one
zero-gapped STFT timeline so a whole scene (or batch of scenes) costs a handful of
kernel launches.

Results match the NumPy implementation to floating point round-off (~1e-15 relative
in float64, ~1e-7 in float32); see ``tests/test_synthesize_torch.py``.
"""

import math
from typing import Optional, Sequence, Union

import numpy as np
import torch

from audiblelight import config

__all__ = [
    "resolve_device",
    "stft_torch",
    "perform_time_variant_convolution_torch",
    "istft_overlap_synthesis_torch",
    "time_invariant_convolution_torch",
    "time_variant_convolution_torch",
    "time_variant_convolution_batch",
]

_COMPLEX = {torch.float32: torch.complex64, torch.float64: torch.complex128}
TensorLike = Union[np.ndarray, torch.Tensor]


def resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """``None``/"auto" -> cuda if available, else mps, else cpu."""
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _as_tensor(x: TensorLike, device, dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)


def _next_fast_len(n: int) -> int:
    """Smallest 2^a * 3^b * 5^c >= n."""
    best = 1 << max(0, (n - 1).bit_length())
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            q = p35
            while q < n:
                q *= 2
            best = min(best, q)
            p35 *= 3
        p5 *= 5
    return best


def _interp_matrix(ir_times: np.ndarray, sr, hop_size) -> np.ndarray:
    # Imported lazily: synthesize imports heavy optional deps
    from audiblelight.synthesize import generate_interpolation_matrix

    return generate_interpolation_matrix(ir_times, sr, hop_size)


def stft_torch(
    y: torch.Tensor,
    fft_size: int = config.FFT_SIZE,
    win_size: int = config.WIN_SIZE,
    hop_size: int = config.HOP_SIZE,
) -> torch.Tensor:
    """Torch port of :func:`audiblelight.synthesize.stft`: (..., samples) -> (frames, freq, ...)."""
    n = torch.arange(win_size, device=y.device, dtype=y.dtype)
    window = torch.sin(math.pi / win_size * n) ** 2
    n_frames = 2 * int(np.ceil(y.shape[-1] / (2.0 * hop_size))) + 1
    y = torch.nn.functional.pad(
        y, (win_size - hop_size, n_frames * hop_size - y.shape[-1])
    )
    frames = y.unfold(-1, win_size, hop_size)  # (..., frames, win)
    spec = torch.fft.rfft(frames * window, n=fft_size, dim=-1)  # (..., frames, freq)
    nd = spec.ndim
    return spec.permute(nd - 2, nd - 1, *range(nd - 2))


def _active_spans(w: torch.Tensor):
    """Per-column [start, start + span) of the non-zero rows of ``w`` (T, L) and the
    weights re-aligned to start: (L, max_span), zero padded."""
    n = w.shape[0]
    active = w != 0
    t_idx = torch.arange(n, device=w.device)[:, None]
    starts = torch.where(active, t_idx, n).amin(0)
    ends = torch.where(active, t_idx, -1).amax(0) + 1
    spans = (ends - starts).clamp_min(0)
    max_span = int(spans.max()) if spans.numel() else 0
    idx = starts[:, None] + torch.arange(max_span, device=w.device)
    valid = idx < ends[:, None]
    w_seg = torch.gather(w.T, 1, idx.clamp_max(n - 1)) * valid
    return starts, spans, w_seg


def _segment_tv_conv(x, s_ir, w_seg, starts, spans, max_chunk_elems=2**26):
    """``Y = sum_l conv_t(s_ir[..., l], W[:, l, None] * x)`` with column ``l`` of ``W``
    equal to ``w_seg[l, :spans[l]]`` on frames ``[starts[l], starts[l] + spans[l])``."""
    n = x.shape[0]
    k, n_freq, n_ch, _ = s_ir.shape
    device = x.device
    fc = n_freq * n_ch
    order = sorted((sp, l) for l, sp in enumerate(spans.tolist()) if sp > 0)
    pad = (order[-1][0] if order else 0) + k
    y = torch.zeros((n + pad, fc), dtype=x.dtype, device=device)
    i = 0
    while i < len(order):
        # largest group of consecutive (span-sorted) IRs within the memory budget
        j = i + 1
        while j < len(order):
            if (j - i + 1) * _next_fast_len(order[j][0] + k - 1) * fc > max_chunk_elems:
                break
            j += 1
        span = order[j - 1][0]
        ids = torch.tensor([l for _, l in order[i:j]], device=device)
        i = j
        n_conv = span + k - 1
        n_fft = _next_fast_len(n_conv)
        g_start = starts[ids]
        g = ids.numel()
        idx = (g_start[:, None] + torch.arange(span, device=device)).clamp_max(n - 1)
        z = w_seg[ids, :span][..., None] * x[idx]  # (G, span, F)
        zf = torch.fft.fft(z, n=n_fft, dim=1)
        hf = torch.fft.fft(s_ir[..., ids].permute(3, 0, 1, 2), n=n_fft, dim=1)
        yg = torch.fft.ifft(hf * zf[..., None], dim=1)[:, :n_conv]  # (G, n_conv, F, C)
        del zf, hf
        out_idx = (g_start[:, None] + torch.arange(n_conv, device=device)).flatten()
        y.index_add_(0, out_idx, yg.reshape(g * n_conv, fc))
    return y[:n].reshape(n, n_freq, n_ch)


def perform_time_variant_convolution_torch(
    s_audio: torch.Tensor, s_ir: torch.Tensor, w_ir: torch.Tensor, max_chunk_elems=2**26
) -> torch.Tensor:
    """Torch equivalent of :func:`audiblelight.synthesize.perform_time_variant_convolution`.

    Arguments:
        s_audio: (frames, freq) complex signal spectrogram.
        s_ir: (ir_frames, freq, channels, n_irs) complex IR spectrograms.
        w_ir: (frames, n_irs) real IR mixing weights.

    Returns:
        (frames, freq, channels) complex spectrogram.
    """
    n = min(s_audio.shape[0], w_ir.shape[0])
    starts, spans, w_seg = _active_spans(w_ir[:n])
    return _segment_tv_conv(s_audio[:n], s_ir, w_seg, starts, spans, max_chunk_elems)


def istft_overlap_synthesis_torch(
    spatial_stft: torch.Tensor,
    fft_size: int = config.FFT_SIZE,
    win_size: int = config.WIN_SIZE,
    hop_size: int = config.HOP_SIZE,
) -> torch.Tensor:
    """Torch port of :func:`audiblelight.synthesize.istft_overlap_synthesis`: -> (samples, channels)."""
    n_frames, _, n_ch = spatial_stft.shape
    frames = torch.fft.irfft(spatial_stft, n=fft_size, dim=1, norm="forward")
    out_len = (n_frames - 1) * hop_size + fft_size
    cols = frames.permute(2, 1, 0).reshape(1, n_ch * fft_size, n_frames)
    audio = torch.nn.functional.fold(
        cols, output_size=(1, out_len), kernel_size=(1, fft_size), stride=(1, hop_size)
    ).reshape(n_ch, out_len)
    return audio[:, win_size : n_frames * hop_size].T


def time_invariant_convolution_torch(
    audio: TensorLike, ir: TensorLike, device=None, dtype=torch.float32
) -> torch.Tensor:
    """Torch equivalent of :func:`audiblelight.synthesize.time_invariant_convolution`.

    Arguments:
        audio: (n_samples,) mono audio.
        ir: (n_ir_samples, n_channels) impulse response.

    Returns:
        (n_channels, n_samples + n_ir_samples - 1) tensor (full convolution).
    """
    device = resolve_device(device)
    audio = _as_tensor(audio, device, dtype)
    ir = _as_tensor(ir, device, dtype)
    if audio.ndim != 1:
        raise ValueError(
            f"Only mono input is supported, but got {audio.ndim} dimensions!"
        )
    if ir.ndim != 2:
        raise ValueError(
            f"Expected shape of IR should be (n_samples, n_channels), but got ({tuple(ir.shape)}) instead"
        )
    n = audio.shape[0] + ir.shape[0] - 1
    n_fft = _next_fast_len(n)
    spec = torch.fft.rfft(audio, n=n_fft)[:, None] * torch.fft.rfft(ir, n=n_fft, dim=0)
    return torch.fft.irfft(spec, n=n_fft, dim=0)[:n].T


def time_variant_convolution_torch(
    irs: TensorLike,
    audio: TensorLike,
    ir_times: np.ndarray,
    sample_rate,
    fft_size: int = config.FFT_SIZE,
    win_size: int = config.WIN_SIZE,
    hop_size: int = config.HOP_SIZE,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Torch equivalent of :func:`audiblelight.synthesize.time_variant_convolution`.

    Arguments:
        irs: (n_channels, n_irs, n_ir_samples) impulse responses.
        audio: (n_samples,) mono audio.
        ir_times: (n_irs,) start time of each IR, in seconds.

    Returns:
        (n_channels, n_out_samples) tensor.
    """
    return time_variant_convolution_batch(
        [dict(irs=irs, audio=audio, ir_times=ir_times)],
        sample_rate,
        fft_size,
        win_size,
        hop_size,
        device=device,
        dtype=dtype,
    )[0]


def time_variant_convolution_batch(
    items: Sequence[dict],
    sample_rate,
    fft_size: int = config.FFT_SIZE,
    win_size: int = config.WIN_SIZE,
    hop_size: int = config.HOP_SIZE,
    device=None,
    dtype=torch.float32,
    max_chunk_elems: int = 2**26,
) -> list:
    """Time-variant convolution of many (audio, IRs) pairs at once.

    Equivalent to calling :func:`time_variant_convolution_torch` on each item, but all
    items share one IR STFT, one batched convolution and one overlap-add.

    Arguments:
        items: dicts with ``audio`` (n_samples,), ``irs`` (n_channels, n_irs,
            n_ir_samples) and ``ir_times`` (n_irs,). All items must have the same
            number of channels; IR lengths may differ.

    Returns:
        list of (n_channels, n_out_samples) tensors on ``device``.
    """
    device = resolve_device(device)
    cdtype = _COMPLEX[dtype]
    fft_size, win_size, hop_size = int(fft_size), int(win_size), int(hop_size)
    audios = [_as_tensor(it["audio"], device, dtype) for it in items]
    irs = [_as_tensor(it["irs"], device, dtype) for it in items]
    if not items:
        return []

    # IR spectrograms for every item at once (zero padding an IR is exact)
    n_ir_max = max(h.shape[-1] for h in irs)
    irs_all = torch.cat(
        [torch.nn.functional.pad(h, (0, n_ir_max - h.shape[-1])) for h in irs], dim=1
    )
    s_ir = stft_torch(irs_all, fft_size, win_size, hop_size)  # (K, F, C, L_tot)
    k, n_freq, _, _ = s_ir.shape
    # frames between items: convolution tails and overlap-add never cross over
    gap = k + int(np.ceil(fft_size / hop_size)) + 1

    x_parts, starts, spans, w_segs, layout = [], [], [], [], []
    off = 0
    for it, a in zip(items, audios):
        x = stft_torch(a, fft_size, win_size, hop_size)
        w = _as_tensor(
            _interp_matrix(np.asarray(it["ir_times"]), sample_rate, hop_size), device, dtype
        )
        n = min(x.shape[0], w.shape[0])
        st, sp, ws = _active_spans(w[:n])
        x_parts += [x[:n], torch.zeros((gap, n_freq), dtype=cdtype, device=device)]
        starts.append(st + off)
        spans.append(sp)
        w_segs.append(ws)
        layout.append((off, n))
        off += n + gap
    max_span = max(w.shape[1] for w in w_segs)
    w_seg = torch.cat(
        [torch.nn.functional.pad(w, (0, max_span - w.shape[1])) for w in w_segs]
    )
    y = _segment_tv_conv(
        torch.cat(x_parts), s_ir, w_seg, torch.cat(starts), torch.cat(spans), max_chunk_elems
    )
    keep = torch.zeros(y.shape[0], dtype=torch.bool, device=device)
    for o, n in layout:
        keep[o : o + n] = True
    audio_all = istft_overlap_synthesis_torch(
        y * keep[:, None, None], fft_size, win_size, hop_size
    )  # (samples, C), starting at win_size
    return [audio_all[o * hop_size : (o + n) * hop_size - win_size].T for o, n in layout]
