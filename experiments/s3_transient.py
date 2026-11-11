"""3단계 — 과도 응답.  x' = Ax 의 고유값(극점)으로 링잉을 예측하고 시간 영역 풀이로 확인한다.
손풀이: notes/03_상태방정식과_극점.md
"""
import numpy as np
from common import AQUA, BLUE, INK, MUTED, ORANGE, compare, plt, save
from minispice import Circuit, add_tline, poles, transient
from minispice.signals import step

L, C = 5e-9, 2e-12
w0 = 1 / np.sqrt(L * C)

# --- 직렬 RLC 스텝 응답: 극점 → 링잉 ---------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 1.6]})
rows = []
for R, col in zip((10.0, 50.0, 100.0), (BLUE, ORANGE, AQUA)):
    ckt = (Circuit().V("V1", "in", "0", wave=step(1.0)).R("R", "in", "m", R)
           .L("L", "m", "out", L).C("C", "out", "0", C))
    # 손으로 세운 상태행렬 (x = [v_C, i_L]) 의 고유값과, MNA 에서 뽑은 극점을 비교
    A = np.array([[0, 1 / C], [-1 / L, -R / L]])
    p_hand = np.linalg.eigvals(A)
    p_sim = poles(ckt)
    zeta = R / 2 * np.sqrt(C / L)
    rows.append((f"R={R:g}Ω: Re(s) = -R/2L [1/ns]", -R / (2 * L) / 1e9 if zeta < 1 else p_hand.real.max() / 1e9,
                 p_sim.real.max() / 1e9))
    t, res = transient(ckt, 1.2e-9, 1e-13)
    v = res["out"]
    if zeta < 1:
        wd = w0 * np.sqrt(1 - zeta ** 2)
        rows.append((f"R={R:g}Ω: Im(s) = ωd [rad/ns]", wd / 1e9, np.abs(p_sim.imag).max() / 1e9))
        rows.append((f"R={R:g}Ω: 오버슈트 e^(-πζ/√(1-ζ²))", np.exp(-np.pi * zeta / np.sqrt(1 - zeta ** 2)), v.max() - 1))
        rows.append((f"R={R:g}Ω: 첫 피크 시각 π/ωd [ps]", np.pi / wd * 1e12, t[np.argmax(v)] * 1e12))
    a1.plot(p_sim.real / 1e9, p_sim.imag / 1e9, "o", color=col, ms=7, mec="white", mew=1.5)
    a2.plot(t * 1e9, v, color=col, label=f"R = {R:g} Ω (ζ = {zeta:g})")
compare("직렬 RLC: 고유값으로 예측한 링잉 vs 과도 해석", rows)

a1.axhline(0, color=MUTED, lw=0.6); a1.axvline(0, color=MUTED, lw=0.6)
a1.set(title="Poles = eigenvalues of A", xlabel="Re(s) [1/ns]", ylabel="Im(s) [rad/ns]")
a2.axhline(1, color=MUTED, lw=0.8, ls=":")
a2.set(title="Step response: poles near the jω axis ring", xlabel="Time [ns]", ylabel="v_out [V]")
a2.legend(loc="lower right")
save(fig, "s3_rlc_step.png")

# --- 전송선: 반사 격자도(손) vs LC 사다리(시뮬) ----------------------------
Z0, Td, tr, N = 50.0, 1e-9, 0.2e-9, 100
dt = 1e-12


def far_end(Rs):
    ckt = Circuit().V("V1", "s", "0", wave=step(1.0, 0, tr)).R("Rs", "s", "a", Rs)
    add_tline(ckt, "T", "a", "b", Z0, Td, N)
    return transient(ckt, 9e-9, dt)


def lattice(Rs, n_bounce=4):
    """원단(개방, Γ_L = +1) 전압의 계단값. k 번째 도착은 t = (2k+1)·Td."""
    g_s, v0, levels, v = (Rs - Z0) / (Rs + Z0), Z0 / (Rs + Z0), [], 0.0
    for k in range(n_bounce):
        v += v0 * (g_s ** k) * 2            # 입사 + 반사(Γ_L = 1) = 2배
        levels.append(v)
    return levels


t, res = far_end(25.0)
levels = lattice(25.0)
mid = lambda k: res["b"][int(round(((2 * k + 2) * Td + tr / 2) / dt))]   # 계단 한가운데
compare("전송선 원단 전압 (Rs = 25 Ω, Z0 = 50 Ω, 원단 개방)", [
    ("1번째 계단 4/3 [V]", levels[0], mid(0)),
    ("2번째 계단 8/9 [V]", levels[1], mid(1)),
    ("3번째 계단 28/27 [V]", levels[2], mid(2)),
    ("4번째 계단 80/81 [V]", levels[3], mid(3)),
    ("도착 시각 Td [ns] (50% 통과)", (Td + tr / 2) * 1e9, t[np.argmax(res["b"] > levels[0] / 2)] * 1e9),
])

fig, ax = plt.subplots(figsize=(7.5, 4))
stairs_t = [0] + [x for k in range(4) for x in ((2 * k + 1) * Td * 1e9,) * 2] + [9]
stairs_v = [0, 0] + [x for lv in levels for x in (lv, lv)]
ax.plot(stairs_t, stairs_v, color=INK, lw=1.2, ls="--", label="Hand: lattice diagram (ideal step)")
ax.plot(t * 1e9, res["b"], color=BLUE, label="Sim: Rs = 25 Ω (Γs = -1/3), 100-section LC ladder")
t2, res2 = far_end(50.0)
ax.plot(t2 * 1e9, res2["b"], color=ORANGE, label="Sim: Rs = 50 Ω (matched source, Γs = 0)")
ax.set(title="Far-end voltage of an open 50 Ω line: reflections arrive every 2·Td",
       xlabel="Time [ns]", ylabel="v_far [V]", xlim=(0, 9))
ax.legend(loc="lower right")
save(fig, "s3_tline_reflection.png")
