"""회로 → 행렬.  수정 노드 해석(MNA)으로  G·x + C·x' = B·u(t)  를 만든다.

미지수 벡터 x = [노드 전압들 | 전압원 전류들 | 인덕터 전류들]

스탬프 규칙 (KCL: 노드에서 '나가는' 전류의 합 = 0)
    R (a-b)      : G[a,a]+=g  G[b,b]+=g  G[a,b]-=g  G[b,a]-=g      (g = 1/R)
    C (a-b)      : 같은 모양을 C 행렬에
    I (a→b)      : b[a]-=I  b[b]+=I
    V (a=+, b=-) : 가지 전류 k 추가.  G[a,k]+=1 G[b,k]-=1 / G[k,a]+=1 G[k,b]-=1 / b[k]=V
    L (a-b)      : 가지 전류 k 추가.  G[a,k]+=1 G[b,k]-=1 / G[k,a]+=1 G[k,b]-=1 / C[k,k]-=L
                   (k행:  v_a - v_b - L·di/dt = 0)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

GROUND = "0"

_SUFFIX = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3,
           "k": 1e3, "meg": 1e6, "g": 1e9, "t": 1e12}
_NUM = re.compile(r"^([+-]?\d*\.?\d+(?:e[+-]?\d+)?)(meg|[fpnumkgt])?[a-z]*$")


def parse_value(text: str) -> float:
    """'1k' → 1000.0, '4.7n' → 4.7e-9, '10meg' → 1e7 (SPICE 표기)."""
    m = _NUM.match(text.strip().lower())
    if not m:
        raise ValueError(f"값을 읽을 수 없음: {text!r}")
    return float(m.group(1)) * _SUFFIX.get(m.group(2), 1.0)


@dataclass
class Element:
    kind: str                      # 'R' 'C' 'L' 'V' 'I'
    name: str
    a: str
    b: str
    value: float
    wave: Optional[Callable] = None    # 과도 해석용 u(t). 없으면 상수 value


@dataclass
class System:
    """G·x + C·x' = B·u.  x 의 앞 n_nodes 개가 노드 전압."""
    G: np.ndarray
    C: np.ndarray
    B: np.ndarray                  # (미지수 수) × (소스 수)
    nodes: list                    # 노드 이름 (접지 제외)
    branches: list                 # 가지 전류 이름 (V, L 소자 이름)
    sources: list                  # 소스 Element 목록 (B 의 열 순서)

    @property
    def labels(self):
        return [f"v({n})" for n in self.nodes] + [f"i({b})" for b in self.branches]

    def index(self, label: str) -> int:
        """'out' 또는 'v(out)' → 노드 전압 위치, 'i(L1)' → 가지 전류 위치."""
        if label in self.nodes:
            return self.nodes.index(label)
        return self.labels.index(label)

    def dc_values(self) -> np.ndarray:
        return np.array([s.value for s in self.sources], dtype=float)


class Circuit:
    def __init__(self, title: str = ""):
        self.title = title
        self.elements: list[Element] = []

    # ---- 소자 추가 -------------------------------------------------------
    def _add(self, kind, name, a, b, value, wave=None):
        if kind in "RCL" and value <= 0:
            raise ValueError(f"{name}: 값은 양수여야 함")
        self.elements.append(Element(kind, name, str(a), str(b), float(value), wave))
        return self

    def R(self, name, a, b, value): return self._add("R", name, a, b, value)
    def C(self, name, a, b, value): return self._add("C", name, a, b, value)
    def L(self, name, a, b, value): return self._add("L", name, a, b, value)

    def V(self, name, a, b, value=1.0, wave=None):
        """전압원. a 가 +, b 가 -.  wave(t) 를 주면 과도 해석에서 그 파형을 쓴다."""
        return self._add("V", name, a, b, value, wave)

    def I(self, name, a, b, value=1.0, wave=None):
        """전류원. 전류가 소자 안에서 a → b 로 흐른다 (b 노드로 주입)."""
        return self._add("I", name, a, b, value, wave)

    @classmethod
    def from_netlist(cls, text: str) -> "Circuit":
        """SPICE 비슷한 넷리스트.  한 줄에  이름 노드a 노드b 값.  '*' 는 주석."""
        ckt = cls()
        for raw in text.strip().splitlines():
            line = raw.split("*")[0].strip()
            if not line:
                continue
            name, a, b, val = line.split()[:4]
            kind = name[0].upper()
            if kind not in "RCLVI":
                raise ValueError(f"모르는 소자: {name}")
            ckt._add(kind, name, a, b, parse_value(val))
        return ckt

    # ---- 행렬 만들기 -----------------------------------------------------
    def build(self) -> System:
        nodes = []
        for e in self.elements:
            for n in (e.a, e.b):
                if n != GROUND and n not in nodes:
                    nodes.append(n)
        branches = [e.name for e in self.elements if e.kind in "VL"]
        sources = [e for e in self.elements if e.kind in "VI"]
        n, size = len(nodes), len(nodes) + len(branches)

        G = np.zeros((size, size))
        C = np.zeros((size, size))
        B = np.zeros((size, len(sources)))
        idx = lambda node: None if node == GROUND else nodes.index(node)

        def stamp_two_terminal(M, a, b, val):
            if a is not None: M[a, a] += val
            if b is not None: M[b, b] += val
            if a is not None and b is not None:
                M[a, b] -= val
                M[b, a] -= val

        def stamp_branch(a, b, k):
            if a is not None:
                G[a, k] += 1; G[k, a] += 1
            if b is not None:
                G[b, k] -= 1; G[k, b] -= 1

        for e in self.elements:
            a, b = idx(e.a), idx(e.b)
            if e.kind == "R":
                stamp_two_terminal(G, a, b, 1.0 / e.value)
            elif e.kind == "C":
                stamp_two_terminal(C, a, b, e.value)
            elif e.kind == "L":
                k = n + branches.index(e.name)
                stamp_branch(a, b, k)
                C[k, k] -= e.value
            elif e.kind == "V":
                k = n + branches.index(e.name)
                stamp_branch(a, b, k)
                B[k, sources.index(e)] = 1.0
            elif e.kind == "I":
                j = sources.index(e)
                if a is not None: B[a, j] -= 1.0
                if b is not None: B[b, j] += 1.0
        return System(G, C, B, nodes, branches, sources)
