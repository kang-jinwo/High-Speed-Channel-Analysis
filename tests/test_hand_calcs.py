"""손계산 값과 시뮬레이터가 일치하는지 확인한다.  notes/ 의 각 노트와 1:1 로 대응."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minispice import (Circuit, SingularCircuitError, add_tline, diagnose,
                       parse_value, poles, solve_ac, solve_dc, transient)
from minispice.signals import (bits_to_wave, convolve_bits, eye_metrics,
                               fourier_coeffs, prbs, prbs7, pulse, step,
                               trapezoid_clock, trapezoid_harmonics)

L, C = 5e-9, 2e-12


def rlc(R, wave=None):
    return (Circuit().V("V1", "in", "0", wave=wave).R("R", "in", "m", R)
            .L("L", "m", "out", L).C("C", "out", "0", C))


# ---- 1단계 -----------------------------------------------------------------
def test_parse_value():
    assert parse_value("1k") == 1e3
    assert parse_value("4.7n") == pytest.approx(4.7e-9)
    assert parse_value("10meg") == 1e7
    assert parse_value("2pF") == pytest.approx(2e-12)


def test_bridge_matches_cramer():
    ckt = Circuit.from_netlist("""
        V1 a 0 10
        R1 a b 1k
        R2 b 0 2k
        R3 a c 3k
        R4 c 0 1k
        R5 b c 5k
    """)
    v = solve_dc(ckt)
    assert v["v(b)"] == pytest.approx(2400 / 385)
    assert v["v(c)"] == pytest.approx(1150 / 385)


def test_current_source_direction():
    v = solve_dc(Circuit().I("I1", "0", "a", 2e-3).R("R1", "a", "0", 1e3))
    assert v["v(a)"] == pytest.approx(2.0)


def test_floating_node_is_reported():
    ckt = Circuit().V("V1", "a", "0", 1).R("R1", "a", "0", 1e3).R("R2", "x", "y", 1e3)
    with pytest.raises(SingularCircuitError) as err:
        solve_dc(ckt)
    assert set(err.value.suspects) == {"v(x)", "v(y)"}


def test_voltage_source_loop_is_reported():
    ckt = Circuit().V("V1", "a", "0", 5).V("V2", "a", "0", 3).R("R1", "a", "0", 1e3)
    assert set(diagnose(ckt)[2]) == {"i(V1)", "i(V2)"}


# ---- 2단계 -----------------------------------------------------------------
def test_rc_corner():
    R = 50.0
    ckt = Circuit().V("V1", "in", "0").R("R", "in", "out", R).C("C", "out", "0", C)
    H = solve_ac(ckt, [1 / (2 * np.pi * R * C)], "out")[0]
    assert abs(H) == pytest.approx(1 / np.sqrt(2))
    assert np.degrees(np.angle(H)) == pytest.approx(-45)


@pytest.mark.parametrize("R", [10.0, 50.0, 100.0])
def test_rlc_gain_at_resonance_is_q(R):
    f0 = 1 / (2 * np.pi * np.sqrt(L * C))
    H = solve_ac(rlc(R), [f0], "out")[0]
    assert abs(H) == pytest.approx(np.sqrt(L / C) / R)
    assert np.degrees(np.angle(H)) == pytest.approx(-90)


# ---- 3단계 -----------------------------------------------------------------
@pytest.mark.parametrize("R", [10.0, 50.0, 200.0])
def test_poles_equal_state_matrix_eigenvalues(R):
    A = np.array([[0, 1 / C], [-1 / L, -R / L]])
    key = lambda p: (round(p.real, -3), round(p.imag, -3))
    hand = sorted(np.linalg.eigvals(A), key=key)
    sim = sorted(poles(rlc(R)), key=key)
    assert np.allclose(sim, hand, rtol=1e-6)


def test_rlc_overshoot():
    R = 10.0
    zeta = R / 2 * np.sqrt(C / L)
    t, res = transient(rlc(R, step(1.0)), 1e-9, 1e-13)
    assert res["out"].max() - 1 == pytest.approx(np.exp(-np.pi * zeta / np.sqrt(1 - zeta ** 2)), rel=1e-3)


def test_tline_lattice_levels():
    z0, td, tr, dt = 50.0, 1e-9, 0.2e-9, 1e-12
    ckt = Circuit().V("V1", "s", "0", wave=step(1.0, 0, tr)).R("Rs", "s", "a", 25.0)
    add_tline(ckt, "T", "a", "b", z0, td, 100)
    _, res = transient(ckt, 7e-9, dt)
    for k, level in enumerate([4 / 3, 8 / 9, 28 / 27]):
        assert res["b"][int(round(((2 * k + 2) * td + tr / 2) / dt))] == pytest.approx(level, abs=0.01)


# ---- 4단계 -----------------------------------------------------------------
def test_clock_harmonics_match_fft():
    T, tr = 1e-9, 0.1e-9
    samples = trapezoid_clock(1.0, T, 0.5, tr)(np.arange(4096) * T / 4096)
    assert np.allclose(np.abs(fourier_coeffs(samples, 25)),
                       trapezoid_harmonics(np.arange(26), 1.0, 0.5, tr / T), atol=1e-6)


def test_prbs7_period():
    b = prbs7(254)
    assert np.array_equal(b[:127], b[127:]) and b[:127].sum() == 64


def test_rc_eye_height():
    R, spu = 50.0, 64
    ui = 2 * R * C
    ckt = Circuit().V("V1", "in", "0", wave=pulse(1, 0, ui)).R("R", "in", "out", R).C("C", "out", "0", C)
    _, res = transient(ckt, 24 * ui, ui / spu)
    bits = prbs7(600)
    m = eye_metrics(convolve_bits(bits, res["out"], spu), bits, spu)
    assert m["height"] == pytest.approx(1 - 2 * np.exp(-2), rel=0.02)


def test_convolution_equals_direct_transient():
    R, spu = 50.0, 32
    ui = 2 * R * C
    bits = prbs7(100)
    mk = lambda wave: Circuit().V("V1", "in", "0", wave=wave).R("R", "in", "out", R).C("C", "out", "0", C)
    _, p = transient(mk(pulse(1, 0, ui)), 30 * ui, ui / spu)
    _, d = transient(mk(bits_to_wave(bits, ui)), 100 * ui - ui / spu, ui / spu)
    assert np.allclose(convolve_bits(bits, p["out"], spu), d["out"], atol=1e-9)


def test_reflection_eye_height():
    z0, td, ui, spu = 50.0, 1e-9, 0.4e-9, 32
    ckt = Circuit().V("V1", "s", "0", wave=pulse(1, 0, ui, 0.08e-9)).R("Rs", "s", "a", 25.0)
    add_tline(ckt, "T", "a", "b", z0, td, 40)
    ckt.C("CL", "b", "0", 1e-12)
    _, res = transient(ckt, 100 * ui, ui / spu)
    bits = prbs(15, 3000)
    m = eye_metrics(convolve_bits(bits, res["b"], spu), bits, spu)
    assert m["height"] == pytest.approx(2 / 3, abs=0.03)
