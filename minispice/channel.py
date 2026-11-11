"""전송선(채널)을 LC 사다리로 근사해서 Circuit 에 붙인다.

전송선 한 토막 Δz:   직렬 (R·Δz + L·Δz),  병렬 C·Δz
    Z0 = √(L/C),   Td = 길이·√(LC)     →   L_total = Z0·Td,   C_total = Td/Z0

N 토막 사다리는 저역통과라서 차단 주파수 f_c ≈ N / (π·Td) 위는 통과시키지 못한다.
경험칙: 상승 시간 tr 인 신호를 보려면  N ≳ 10·Td / tr.
"""
from __future__ import annotations

from .circuit import Circuit


def add_tline(ckt: Circuit, name: str, a: str, b: str, z0: float, td: float,
              n: int = 50, r_total: float = 0.0) -> Circuit:
    """노드 a 에서 b 까지 특성 임피던스 z0, 지연 td 인 선로를 n 토막으로 넣는다.

    r_total 은 선로 전체의 직렬(도체) 저항. 0 이면 무손실.
    양 끝에 C/2 를 두는 π 형 배치라 좌우 대칭이다.
    """
    l_seg = z0 * td / n
    c_seg = td / z0 / n
    r_seg = r_total / n
    prev = a
    ckt.C(f"{name}_c0", a, "0", c_seg / 2)
    for k in range(1, n + 1):
        node = b if k == n else f"{name}_n{k}"
        if r_seg > 0:
            mid = f"{name}_m{k}"
            ckt.R(f"{name}_r{k}", prev, mid, r_seg)
            ckt.L(f"{name}_l{k}", mid, node, l_seg)
        else:
            ckt.L(f"{name}_l{k}", prev, node, l_seg)
        ckt.C(f"{name}_c{k}", node, "0", c_seg if k < n else c_seg / 2)
        prev = node
    return ckt
