"""Unit and integration tests (about 30 seconds)."""
import numpy as np
import pandas as pd
import pytest

from sal import channel, outliers, phy, positioning as pos, ranging, receiver
from sal.environment import Arrival, Waveguide, mackenzie, noise_psd_db, rayleigh_bottom, thorp


# ------------------------------------------------------------------ environment
def test_mackenzie_reference_value():
    # Mackenzie (1981) check value: 1550.744 m/s at 25 °C, 35 ppt, 1000 m
    assert mackenzie(25, 35, 1000) == pytest.approx(1550.744, abs=0.01)


def test_thorp_and_noise_are_physical():
    assert thorp(6) == pytest.approx(0.50, abs=0.02)
    assert thorp(30) > thorp(6) > thorp(1)
    psd = noise_psd_db(np.array([1.0, 6.0, 20.0]))
    assert psd[0] > psd[1] > psd[2]


def test_bottom_reflection_total_below_critical_angle():
    c1, c2 = 1490.0, 1700.0
    crit = np.arccos(c1 / c2)
    assert abs(rayleigh_bottom(crit * 0.5, c1, c2)) == pytest.approx(1.0, abs=1e-9)
    assert abs(rayleigh_bottom(np.radians(80), c1, c2)) < 1.0


def test_eigenrays_start_with_the_direct_path():
    wg = Waveguide()
    arr = wg.arrivals(1000, 10, 30, phy.FC)
    assert arr[0].n_surface == 0 and arr[0].n_bottom == 0
    assert arr[0].length == pytest.approx(np.hypot(1000, 20), rel=1e-9)
    assert all(a.delay <= b.delay for a, b in zip(arr, arr[1:]))


# ------------------------------------------------------------------ physical layer
def test_rrc_and_qpsk():
    h = phy.rrc()
    assert np.sum(h ** 2) == pytest.approx(1.0)
    bits = np.random.default_rng(0).integers(0, 2, 200)
    assert np.array_equal(phy.qpsk_to_bits(phy.bits_to_qpsk(bits)), bits)


def test_time_scale_undoes_doppler():
    pkt = phy.make_packet(rng=3)
    arr = [Arrival(0.3, 1.0, 0, 0, 0, 0)]
    a = 2.0 / 1500
    y, _ = channel.propagate(pkt.signal, arr, speed=2.0, c=1500)
    det = receiver.detect_and_doppler(y, pkt)
    assert det[2] == pytest.approx(a, rel=0.02)


@pytest.mark.parametrize("mode", list(receiver.MODES))
def test_ideal_channel_is_error_free(mode):
    pkt = phy.make_packet(rng=1)
    y, _ = channel.propagate(pkt.signal, [Arrival(0.5, 1e-3 * np.exp(0.7j), 0, 0, 0, 0)], c=1500)
    r = receiver.receive(y, pkt, mode, doppler=False)
    assert r.detected and r.ber == 0


def test_dfe_beats_single_tap_on_multipath():
    wg = Waveguide()
    arr = wg.arrivals(1000, 10, 30, phy.FC)
    pkt = phy.make_packet(rng=2)
    y, _ = channel.propagate(pkt.signal, arr, speed=1.0, c=wg.c, spread_hz=0.3, rng=2)
    y = channel.add_noise(y, 22, rng=2)
    dfe = receiver.receive(y, pkt, "dfe")
    one = receiver.receive(y, pkt, "single-tap")
    assert dfe.ber < 1e-3 < one.ber
    assert dfe.a_hat * wg.c == pytest.approx(1.0, abs=0.1)


def test_noise_is_scaled_to_the_requested_snr():
    y = np.ones(20_000, complex)
    for kind in ("wenz", "real"):
        n = channel.add_noise(y, 10.0, kind, rng=0) - y
        assert 10 * np.log10(1 / np.mean(np.abs(n) ** 2)) == pytest.approx(10.0, abs=0.05)


# ------------------------------------------------------------------ ranging
def test_doppler_bias_formula_matches_measurement():
    for v in (-2.0, 2.0):
        y, lead = channel.propagate(phy.chirp(), [Arrival(0.5, 1e-3, 0, 0, 0, 0)], speed=v, c=1500, lead=0.02,
                                    tail=0.02)
        t = ranging.estimate_toa(y, lead)
        assert t.t_first == pytest.approx(ranging.doppler_bias(v, 1500), abs=2e-6)


def test_estimator_close_to_crlb():
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(40):
        y, lead = channel.propagate(phy.chirp(), [Arrival(0.4, 1e-3, 0, 0, 0, 0)], c=1500, lead=0.02, tail=0.02)
        p = np.abs(y) ** 2
        y = channel.add_noise(y, 15, "wenz", rng, ref_power=p[p > 0.05 * p.max()].mean())
        errs.append(ranging.estimate_toa(y, lead).t_first)
    ratio = np.std(errs) / ranging.crlb_toa_std(15)
    assert 0.7 < ratio < 1.6


def test_twtt_short_range_is_accurate():
    p = ranging.twtt_ping(Waveguide(), 400, 30, 49, rng=0, sl=170)
    assert abs(p["r_first"] - p["slant"]) < 0.2


# ------------------------------------------------------------------ positioning
def test_least_squares_recovers_the_position():
    truth = np.array([420.0, 910.0])
    rho = [pos.slant(truth, b) for b in pos.BEACONS]
    assert np.allclose(pos.least_squares(rho), truth, atol=1e-6)


def test_ekf_learns_the_compass_bias():
    track = pos.lawnmower(legs=4)
    vel = pos.dvl_velocity(track, rng=0, noise=0.0, scale=0.0, heading_bias_deg=1.5, heading_noise_deg=0.0)
    f = pos.Ekf(track[0, 1:3])
    rng = np.random.default_rng(1)
    for i in range(1, len(track)):
        f.predict(vel[i - 1])
        if i % 10 == 0:
            for b in pos.BEACONS:
                f.update(pos.slant(track[i, 1:3], b) + rng.normal(0, 0.05), b)
    assert np.degrees(f.x[2]) == pytest.approx(1.5, abs=0.15)
    assert np.hypot(*(f.xy - track[-1, 1:3])) < 0.3


def test_outlier_label():
    df = pd.DataFrame({"r_first": [100.0, 101.0], "slant": [100.1, 100.0]})
    assert list(outliers.label(df)) == [0, 1]
