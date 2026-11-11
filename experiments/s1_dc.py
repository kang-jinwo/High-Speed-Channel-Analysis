"""1단계 — 풀이 엔진.  노드 해석 G·v = i 를 세워 풀고, det = 0 인 회로를 잡아낸다.
손풀이: notes/01_노드해석과_행렬.md
"""
import numpy as np
from common import compare
from minispice import Circuit, SingularCircuitError, diagnose, solve_dc

np.set_printoptions(precision=4, suppress=True, linewidth=120)

# --- 브리지 회로: 손으로는 2×2 크래머 공식 ---------------------------------
bridge = Circuit.from_netlist("""
V1 a 0 10
R1 a b 1k
R2 b 0 2k
R3 a c 3k
R4 c 0 1k
R5 b c 5k      * 브리지 저항
""")
sys = bridge.build()
print("미지수:", sys.labels)
print("G [mS 단위로 보기 좋게 ×1000, 전압원 행·열은 ±1000]:\n", sys.G * 1e3)

v = solve_dc(bridge)
compare("브리지 회로 DC", [
    ("v(b) = 2400/385 [V]", 2400 / 385, v["v(b)"]),
    ("v(c) = 1150/385 [V]", 1150 / 385, v["v(c)"]),
    ("전원 전류 [mA]", (10 - 2400 / 385) / 1 + (10 - 1150 / 385) / 3, -v["i(V1)"] * 1e3),
])

# --- 잘못된 회로: 어디에도 연결되지 않은 섬 --------------------------------
broken = Circuit.from_netlist("""
V1 a 0 10
R1 a b 1k
R2 b 0 2k
R3 x y 1k      * 접지와 이어지지 않은 저항
""")
rank, size, suspects = diagnose(broken)
print(f"\n[떠 있는 노드] rank(G) = {rank} < {size}  →  det(G) = {np.linalg.det(broken.build().G):.3g}")
print("  영공간이 가리키는 미지수:", suspects)
try:
    solve_dc(broken)
except SingularCircuitError as e:
    print("  solve_dc:", e)

# --- 잘못된 회로 2: 전압원 두 개를 병렬로 ----------------------------------
loop = Circuit().V("V1", "a", "0", 5).V("V2", "a", "0", 3).R("R1", "a", "0", 1e3)
print("\n[전압원 루프] 영공간이 가리키는 미지수:", diagnose(loop)[2])
