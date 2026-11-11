"""해석 엔진.  전부 '행렬 하나를 푼다'로 귀결된다.

    DC      :  G·x = b                     (C 열림, L 단락)
    AC      :  (G + jωC)·X = B             (페이저)
    극점    :  det(G + sC) = 0  인 s        (일반화 고유값 문제)
    과도    :  C·x' = -G·x + B·u(t)         (사다리꼴 적분)
"""
from __future__ import annotations

import numpy as np

from .circuit import Circuit, System


class SingularCircuitError(Exception):
    """G 가 특이행렬(det = 0) → 회로가 유일한 해를 갖지 못한다."""

    def __init__(self, message, suspects):
        super().__init__(message)
        self.suspects = suspects       # 값이 정해지지 않는 미지수 이름들


def _system(c) -> System:
    return c.build() if isinstance(c, Circuit) else c


def diagnose(c, tol: float = 1e-12):
    """G 의 영공간을 구해 '정해지지 않는 미지수'를 찾는다.

    G·n = 0 인 n ≠ 0 이 있으면, 해 x 에 n 을 얼마든지 더해도 식이 성립한다.
    n 에서 0 이 아닌 성분이 바로 떠 있는(floating) 노드나 전압원 루프다.
    반환: (rank, 미지수 수, 의심 미지수 이름 목록)
    """
    s = _system(c)
    U, sv, Vt = np.linalg.svd(s.G)
    rank = int(np.sum(sv > tol * sv[0])) if sv.size and sv[0] > 0 else 0
    suspects = []
    for vec in Vt[rank:]:
        for i in np.flatnonzero(np.abs(vec) > 1e-6):
            if s.labels[i] not in suspects:
                suspects.append(s.labels[i])
    return rank, s.G.shape[0], suspects


def solve_dc(c) -> dict:
    """DC 동작점.  G·x = B·u_dc.  특이하면 원인을 알려 주며 실패한다."""
    s = _system(c)
    rank, size, suspects = diagnose(s)
    if rank < size:
        raise SingularCircuitError(
            f"G 가 특이행렬 (rank {rank} < {size}, det = 0). "
            f"값이 정해지지 않는 미지수: {', '.join(suspects)}", suspects)
    x = np.linalg.solve(s.G, s.B @ s.dc_values())
    return dict(zip(s.labels, x))


def solve_ac(c, freqs, out: str, source: str | None = None) -> np.ndarray:
    """전달함수 H(jω) = X_out / U_source.  주파수마다 (G + jωC)·X = B_source 를 푼다."""
    s = _system(c)
    j = 0 if source is None else [e.name for e in s.sources].index(source)
    k = s.index(out)
    w = 2j * np.pi * np.asarray(freqs, dtype=float)
    A = s.G[None, :, :] + w[:, None, None] * s.C[None, :, :]
    rhs = np.broadcast_to(s.B[:, j].astype(complex), (len(w), s.G.shape[0]))
    return np.linalg.solve(A, rhs[..., None])[:, k, 0]


def poles(c) -> np.ndarray:
    """자연 응답 x = v·e^{st} 를 G·x + C·x' = 0 에 넣으면 (G + sC)·v = 0.

    s = s0 + δ 로 옮기면 (G + s0·C)⁻¹C 의 고유값 μ 에 대해 δ = -1/μ.
    (C 가 특이해도 되고, s0 > 0 이면 수동 회로에서 G + s0·C 는 항상 가역이다.)
    μ = 0 은 '무한대 극점' = 동적이지 않은 대수 방정식이므로 버린다.
    """
    s = _system(c)
    gn, cn = np.abs(s.G).max(), np.abs(s.C).max()
    if cn == 0:
        return np.array([])
    s0 = gn / cn if gn > 0 else 1.0
    mu = np.linalg.eigvals(np.linalg.solve(s.G + s0 * s.C, s.C))
    mu = mu[np.abs(mu) > 1e-9 * np.abs(mu).max()]
    p = s0 - 1.0 / mu
    return p[np.argsort(-p.real - 1e-9 * np.abs(p.imag))]


def transient(c, t_stop: float, dt: float, theta: float = 0.5, x0=None):
    """시간 영역 풀이.  θ = 0.5 사다리꼴(2차 정확도), θ = 1 후진 오일러.

        (C/h + θG)·x₁ = (C/h - (1-θ)G)·x₀ + θ·b₁ + (1-θ)·b₀

    스텝마다 같은 행렬이므로 역행렬을 한 번만 구해 재사용한다.
    초기값은 u(0) 에서의 DC 동작점(구할 수 없으면 0).
    반환: t, Result (res['out'], res['i(L1)'] 처럼 꺼낸다)
    """
    s = _system(c)
    t = np.arange(0.0, t_stop + dt / 2, dt)
    u = np.empty((len(t), len(s.sources)))
    for j, e in enumerate(s.sources):
        u[:, j] = e.value if e.wave is None else e.wave(t)
    b = u @ s.B.T

    if x0 is None:
        try:
            x0 = np.linalg.solve(s.G, b[0])
        except np.linalg.LinAlgError:
            x0 = np.zeros(s.G.shape[0])
    inv = np.linalg.inv(s.C / dt + theta * s.G)
    step = inv @ (s.C / dt - (1 - theta) * s.G)

    x = np.empty((len(t), s.G.shape[0]))
    x[0] = x0
    for n in range(len(t) - 1):
        x[n + 1] = step @ x[n] + inv @ (theta * b[n + 1] + (1 - theta) * b[n])
    return t, Result(s, x)


class Result:
    def __init__(self, system: System, x: np.ndarray):
        self.system, self.x = system, x

    def __getitem__(self, label: str) -> np.ndarray:
        return self.x[:, self.system.index(label)]
