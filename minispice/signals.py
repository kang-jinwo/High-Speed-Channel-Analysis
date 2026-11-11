"""신호 쪽 도구: 파형, 푸리에 급수, PRBS, 컨볼루션, 아이 다이어그램."""
from __future__ import annotations

import numpy as np


# ---- 파형 (과도 해석의 wave 인자로 넣는다) ---------------------------------
def step(v: float = 1.0, t0: float = 0.0, tr: float = 0.0):
    """t0 에서 시작해 tr 동안 선형으로 올라가는 스텝. tr = 0 이면 이상적인 스텝."""
    if tr <= 0:
        return lambda t: v * (np.asarray(t) > t0)
    return lambda t: v * np.clip((np.asarray(t) - t0) / tr, 0.0, 1.0)


def pulse(v: float, t0: float, width: float, tr: float = 0.0):
    """폭 width 인 펄스 한 개 = 스텝 - (width 만큼 늦은 스텝)."""
    up, down = step(v, t0, tr), step(v, t0 + width, tr)
    return lambda t: up(t) - down(t)


def trapezoid_clock(v: float, period: float, duty: float = 0.5, tr: float = 0.0):
    """사다리꼴 클럭. duty 는 50% 지점 기준 폭/주기, 상승·하강 시간은 tr 로 같다."""
    def wave(t):
        ph = np.mod(np.asarray(t, dtype=float), period)
        hi = duty * period
        if tr <= 0:
            return v * (ph < hi)
        rise = np.clip(ph / tr, 0, 1)
        fall = np.clip((ph - hi) / tr, 0, 1)
        return v * (rise - fall)
    return wave


# ---- 푸리에 급수 ------------------------------------------------------------
def trapezoid_harmonics(n, v: float, duty: float, tr_over_T: float) -> np.ndarray:
    """사다리꼴 클럭의 푸리에 계수 크기 |c_n| (손 유도 결과).

        |c_n| = V·d·|sinc(n·d)|·|sinc(n·tr/T)|,     sinc(x) = sin(πx)/(πx)

    단측 스펙트럼의 n차 고조파 진폭은 2|c_n| (n ≥ 1).
    """
    n = np.asarray(n, dtype=float)
    return np.abs(v * duty * np.sinc(n * duty) * np.sinc(n * tr_over_T))


def fourier_coeffs(samples: np.ndarray, n_max: int) -> np.ndarray:
    """한 주기를 균일 샘플링한 파형에서 c_0 … c_nmax 를 수치로 구한다 (FFT)."""
    return (np.fft.rfft(samples) / len(samples))[: n_max + 1]


# ---- 비트열 -----------------------------------------------------------------
_PRBS_TAPS = {7: (7, 6), 15: (15, 14)}       # x⁷+x⁶+1,  x¹⁵+x¹⁴+1


def prbs(order: int, n_bits: int) -> np.ndarray:
    """LFSR 의사 난수 비트열. 주기 2^order - 1 이고 order 비트짜리 패턴이 (전부 0 빼고) 다 나온다."""
    t1, t2 = _PRBS_TAPS[order]
    mask = (1 << order) - 1
    state, out = mask, np.empty(n_bits, dtype=int)
    for i in range(n_bits):
        bit = ((state >> (t1 - 1)) ^ (state >> (t2 - 1))) & 1
        state = ((state << 1) | bit) & mask
        out[i] = bit
    return out


def prbs7(n_bits: int) -> np.ndarray:
    return prbs(7, n_bits)


def bits_to_wave(bits, ui: float, v: float = 1.0, tr: float = 0.0):
    """NRZ 파형 x(t) = Σ b_k·p(t - kT).  과도 해석에 직접 넣어 컨볼루션 결과와 비교할 때 쓴다."""
    bits = np.asarray(bits, dtype=float)
    edges = np.diff(np.concatenate([[0.0], bits]))       # 비트가 바뀌는 지점마다 ±스텝
    ks = np.flatnonzero(edges)

    def wave(t):
        t = np.asarray(t, dtype=float)
        y = np.zeros_like(t)
        for k in ks:
            y += edges[k] * step(v, k * ui, tr)(t)
        return y
    return wave


# ---- 컨볼루션과 아이 --------------------------------------------------------
def convolve_bits(bits, pulse_response: np.ndarray, spu: int) -> np.ndarray:
    """y[n] = Σ_k b_k · p[n - k·spu]   (LTI 중첩 = 임펄스열 ∗ 펄스 응답)

    pulse_response: 폭 1 UI 짜리 펄스를 넣었을 때의 출력, 샘플 간격 UI/spu.
    """
    train = np.zeros(len(bits) * spu)
    train[::spu] = bits
    return np.convolve(train, pulse_response)[: len(train)]


def eye_metrics(y: np.ndarray, bits, spu: int, skip: int = 16) -> dict:
    """아이 높이와 폭.  보낸 비트를 알고 있으므로 그것으로 1/0 을 가른다.

    샘플링 지연 d 마다  높이(d) = min(y | 1 보냄) - max(y | 0 보냄).
    높이가 가장 큰 d 가 최적 샘플링 시점이고, 그 주변에서 높이 > 0 인 구간이 아이 폭.
    """
    bits = np.asarray(bits)
    max_delay = 8 * spu                                  # 채널 지연은 8 UI 안쪽이라고 본다
    ks = np.arange(skip, len(bits) - 8)                  # 앞쪽 과도 구간과 끝은 버린다
    heights = np.empty(max_delay)
    for d in range(max_delay):
        s = y[ks * spu + d]
        heights[d] = s[bits[ks] == 1].min() - s[bits[ks] == 0].max()
    best = int(np.argmax(heights))
    lo = hi = best
    while lo > 0 and heights[lo - 1] > 0: lo -= 1
    while hi < max_delay - 1 and heights[hi + 1] > 0: hi += 1
    width = (hi - lo + 1) / spu if heights[best] > 0 else 0.0
    return {"height": float(heights[best]), "width_ui": min(width, 1.0),
            "delay_samples": best, "heights": heights}


def eye_traces(y: np.ndarray, spu: int, delay: int, skip: int = 16) -> np.ndarray:
    """그림용: 최적 샘플링 시점이 가운데 오도록 2 UI 씩 잘라 겹친다. (구간 수 × 2·spu)"""
    start = skip * spu + delay - spu
    n = (len(y) - start) // spu - 2
    return np.stack([y[start + k * spu: start + (k + 2) * spu] for k in range(n)])
