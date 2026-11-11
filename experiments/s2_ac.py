"""2단계 — 주파수 해석.  페이저로 H(jω) 를 구해 보드선도를 그리고 손계산과 비교한다.
손풀이: notes/02_페이저와_보드선도.md
"""
import numpy as np
from common import AQUA, BLUE, MUTED, ORANGE, compare, plt, save
from minispice import Circuit, solve_ac

f = np.logspace(7, 11, 2001)                      # 10 MHz ~ 100 GHz

# --- RC 저역통과: 드라이버 50 Ω + 수신단 2 pF ------------------------------
R, C = 50.0, 2e-12
rc = Circuit().V("V1", "in", "0").R("R", "in", "out", R).C("C", "out", "0", C)
H_rc = solve_ac(rc, f, "out")
fc = 1 / (2 * np.pi * R * C)
f3 = f[np.argmin(np.abs(np.abs(H_rc) - 1 / np.sqrt(2)))]
compare("RC 저역통과", [
    ("-3 dB 주파수 1/(2πRC) [GHz]", fc / 1e9, f3 / 1e9),
    ("|H(fc)| = 1/√2", 1 / np.sqrt(2), abs(solve_ac(rc, [fc], "out")[0])),
    ("∠H(fc) = -45 [deg]", -45, np.degrees(np.angle(solve_ac(rc, [fc], "out")[0]))),
    ("|H(10 fc)| ≈ 1/10 (-20 dB/dec)", 1 / np.sqrt(101), abs(solve_ac(rc, [10 * fc], "out")[0])),
])

# --- 직렬 RLC: 패키지 인덕턴스 5 nH + 2 pF, R 을 바꿔 가며 ------------------
L = 5e-9
f0 = 1 / (2 * np.pi * np.sqrt(L * C))
rows, curves = [], {}
for Rs in (10.0, 50.0, 100.0):
    ckt = (Circuit().V("V1", "in", "0").R("R", "in", "m", Rs)
           .L("L", "m", "out", L).C("C", "out", "0", C))
    curves[Rs] = solve_ac(ckt, f, "out")
    Q = np.sqrt(L / C) / Rs
    rows.append((f"R={Rs:g}Ω: |H(f0)| = Q", Q, abs(solve_ac(ckt, [f0], "out")[0])))
    if Q > 1 / np.sqrt(2):                       # 피크가 생기는 조건
        peak_hand = Q / np.sqrt(1 - 1 / (4 * Q * Q))
        fpk_hand = f0 * np.sqrt(1 - 1 / (2 * Q * Q))
        k = np.argmax(np.abs(curves[Rs]))
        rows.append((f"R={Rs:g}Ω: 피크 크기", peak_hand, abs(curves[Rs][k])))
        rows.append((f"R={Rs:g}Ω: 피크 주파수 [GHz]", fpk_hand / 1e9, f[k] / 1e9))
compare(f"직렬 RLC (f0 = {f0/1e9:.4g} GHz, √(L/C) = {np.sqrt(L/C):g} Ω)", rows)

# --- 그림 --------------------------------------------------------------------
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5.6), sharex=True)
dB = lambda H: 20 * np.log10(np.abs(H))
a1.semilogx(f, dB(H_rc), color=MUTED, lw=1.5, ls="--", label="RC only (no L)")
for Rs, col in zip(curves, (BLUE, ORANGE, AQUA)):
    Q = np.sqrt(L / C) / Rs
    a1.semilogx(f, dB(curves[Rs]), color=col, label=f"RLC, R = {Rs:g} Ω (Q = {Q:g})")
    a2.semilogx(f, np.degrees(np.unwrap(np.angle(curves[Rs]))), color=col)
a2.semilogx(f, np.degrees(np.angle(H_rc)), color=MUTED, lw=1.5, ls="--")
a1.axvline(f0, color=MUTED, lw=0.8, ls=":")
a1.annotate("f0 = 1/(2π√LC)\n= 1.59 GHz", (f0, -42), xytext=(6, 0),
            textcoords="offset points", fontsize=8, color=MUTED)
a1.set(title="Series RLC into C: |H| peaks by Q = √(L/C)/R at f0",
       ylabel="|H| [dB]", ylim=(-60, 20))
a1.legend(loc="lower left")
a2.set(xlabel="Frequency [Hz]", ylabel="Phase [deg]", yticks=[-180, -135, -90, -45, 0])
save(fig, "s2_bode.png")
