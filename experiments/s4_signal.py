"""4단계 — 신호.  클럭의 푸리에 급수로 고조파를 보고, 비트열 ∗ 펄스 응답으로 아이를 그린다.
손풀이: notes/04_푸리에와_아이다이어그램.md
"""
import numpy as np
from common import AQUA, BLUE, INK, MUTED, ORANGE, compare, plt, save
from minispice import Circuit, add_tline, solve_ac, transient
from minispice.signals import (bits_to_wave, convolve_bits, eye_metrics, eye_traces,
                               fourier_coeffs, prbs, prbs7, pulse, trapezoid_clock,
                               trapezoid_harmonics)

# ============ (가) 1 GHz 사다리꼴 클럭의 고조파 ==============================
T, tr, V = 1e-9, 0.1e-9, 1.0
n = np.arange(1, 26)
samples = trapezoid_clock(V, T, 0.5, tr)(np.arange(4096) * T / 4096)
c_fft = np.abs(fourier_coeffs(samples, 25))
c_hand = trapezoid_harmonics(np.arange(26), V, 0.5, tr / T)
compare("1 GHz 클럭 (duty 50%, tr = 100 ps): 고조파 진폭 2|c_n| [V]", [
    ("DC 성분 V·d", c_hand[0], c_fft[0]),
    ("1차 (1 GHz)", 2 * c_hand[1], 2 * c_fft[1]),
    ("3차 (3 GHz)", 2 * c_hand[3], 2 * c_fft[3]),
    ("5차 (5 GHz)", 2 * c_hand[5], 2 * c_fft[5]),
    ("9차 (9 GHz)", 2 * c_hand[9], 2 * c_fft[9]),
    ("10차: sinc(n·tr/T) 의 첫 영점", 0.0, 2 * c_fft[10]),
])

fig, ax = plt.subplots(figsize=(7.5, 4))
odd = n[n % 2 == 1]
dB = lambda x: 20 * np.log10(np.maximum(x, 1e-6))
ax.vlines(odd, -80, dB(2 * c_fft[odd]), color=BLUE, lw=2)
ax.plot(odd, dB(2 * c_fft[odd]), "o", color=BLUE, ms=6, mec="white", mew=1, label="FFT of sampled clock (odd harmonics)")
nn = np.linspace(1, 25, 400)
ax.plot(nn, dB(2 * V / (np.pi * nn)), color=MUTED, lw=1.2, ls="--", label="Envelope 1: 2V/(πn), -20 dB/dec")
ax.plot(nn, dB(2 * V / (np.pi * nn) * np.minimum(1, 1 / (np.pi * nn * tr / T))), color=ORANGE, lw=1.5,
        label="Envelope 2: extra -20 dB/dec above 1/(π·tr)")
ax.axvline(1 / (np.pi * tr / T), color=MUTED, lw=0.8, ls=":")
ax.annotate("knee 1/(π·tr) = 3.2 GHz", (1 / (np.pi * tr / T), -8), xytext=(5, 0),
            textcoords="offset points", fontsize=8, color=MUTED)
ax.set(xscale="log", ylim=(-80, 5), xlabel="Harmonic number n (× 1 GHz)", ylabel="Amplitude [dBV]",
       title="Clock spectrum: the rise time, not the clock rate, sets the bandwidth")
ax.set_xticks([1, 3, 5, 10, 20], ["1", "3", "5", "10", "20"])
ax.legend(loc="upper right")
save(fig, "s4_clock_harmonics.png")

# ============ (나) RC 채널의 아이: 손 공식 1 - 2e^(-T/τ) ====================
R, C = 50.0, 2e-12
tau, spu = R * C, 64
bits = prbs7(600)
rows = []
for ratio in (4.0, 2.0, 1.0):                      # UI / τ
    ui = ratio * tau
    ckt = Circuit().V("V1", "in", "0", wave=pulse(1, 0, ui)).R("R", "in", "out", R).C("C", "out", "0", C)
    _, res = transient(ckt, 24 * ui, ui / spu)
    m = eye_metrics(convolve_bits(bits, res["out"], spu), bits, spu)
    rows.append((f"UI = {ratio:g}τ ({1e-9/ui:.3g} Gbps): 아이 높이", 1 - 2 * np.exp(-ratio), m["height"]))
compare("RC 채널 (τ = 100 ps) 아이 높이 = 1 - 2e^(-UI/τ)", rows)
print(f"  아이가 닫히는 한계: UI = τ·ln2 = {tau*np.log(2)*1e12:.1f} ps → {1e-9/(tau*np.log(2)):.2f} Gbps")

# ============ (다) 전송선 채널: 소스 정합에 따른 아이 =======================
Z0, Td, CL, ui = 50.0, 1e-9, 1e-12, 0.4e-9         # 2.5 Gbps, 왕복 지연 2·Td = 5 UI
tr_bit, dt = 0.08e-9, 0.4e-9 / 64
bits = prbs(15, 3000)                              # 반사가 여러 비트에 걸치므로 긴 패턴이 필요


def channel(Rs, wave):
    ckt = Circuit().V("V1", "s", "0", wave=wave).R("Rs", "s", "a", Rs)
    add_tline(ckt, "T", "a", "b", Z0, Td, 60)
    return ckt.C("CL", "b", "0", CL)


def eye_hand(Rs):
    """원단 개방(Γ_L = 1) 일 때 펄스 응답 = 주 펄스 (1-Γs) 와 5 UI 간격의 메아리 (1-Γs)·Γs^k.
    최악 아이 높이 = 주 펄스 - Σ|메아리| = (1-Γs)·(1-2|Γs|)/(1-|Γs|)."""
    g = (Rs - Z0) / (Rs + Z0)
    return (1 - g) * (1 - 2 * abs(g)) / (1 - abs(g))


cases, rows = (50.0, 25.0, 15.0), []
fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
for ax, Rs, col in zip(axes, cases, (BLUE, ORANGE, AQUA)):
    _, res = transient(channel(Rs, pulse(1, 0, ui, tr_bit)), 150 * ui, dt)
    y = convolve_bits(bits, res["b"], spu)
    m = eye_metrics(y, bits, spu)
    gam = (Rs - Z0) / (Rs + Z0)
    rows.append((f"Rs = {Rs:g}Ω (Γs = {gam:+.3f}): 아이 높이 [V]", eye_hand(Rs), m["height"]))
    if Rs == 25.0:
        # 컨볼루션(LTI 중첩)이 맞는지: 같은 비트열을 통째로 과도 해석한 결과와 비교
        nb = 200
        _, direct = transient(channel(Rs, bits_to_wave(bits[:nb], ui, 1.0, tr_bit)), nb * ui - dt, dt)
        conv_err = np.abs(direct["b"] - y[:nb * spu]).max()
    tt = np.arange(2 * spu) / spu - 1
    for trace in eye_traces(y, spu, m["delay_samples"])[:400]:
        ax.plot(tt, trace, color=col, lw=0.6, alpha=0.2)
    state = f"eye height {m['height']:.2f} V" if m["height"] > 0.02 else "eye closed"
    ax.set(title=f"Rs = {Rs:g} Ω (Γs = {gam:+.2f}): {state}", xlabel="Time [UI]", xlim=(-1, 1))
    ax.axvline(0, color=INK, lw=0.6, ls=":")
axes[0].set_ylabel("v_rx [V]")
compare("전송선 채널 2.5 Gbps: 아이 높이 (1-Γs)(1-2|Γs|)/(1-|Γs|)", rows)
print("  (손 값이 음수 = 닫힌 아이. 손 공식은 C_L 과 상승 시간을 무시한 값)")
print(f"  아이가 닫히는 한계: |Γs| = 1/2  →  Rs = Z0/3 = {Z0/3:.1f} Ω")
print(f"  컨볼루션 vs 비트열을 통째로 과도 해석: 최대 차이 {conv_err:.1e} V")
save(fig, "s4_eye.png")

# 채널의 주파수 응답으로 같은 현상을 다시 본다 (2단계와 연결)
f = np.linspace(1e7, 5e9, 1500)
fig, ax = plt.subplots(figsize=(7.5, 3.6))
for Rs, col in zip(cases, (BLUE, ORANGE, AQUA)):
    ax.plot(f / 1e9, 20 * np.log10(np.abs(solve_ac(channel(Rs, None), f, "b"))), color=col, label=f"Rs = {Rs:g} Ω")
ax.set(title="Channel |H(f)|: mismatch turns the line into a resonator (peaks every 1/(2·Td))",
       xlabel="Frequency [GHz]", ylabel="|V_rx / V_src| [dB]")
ax.legend()
save(fig, "s4_channel_response.png")
