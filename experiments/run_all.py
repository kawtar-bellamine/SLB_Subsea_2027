"""Full study: environment, link (BER), Doppler, real noise, ranging, ML outlier detection and LBL navigation.

Writes results/*.csv, results/summary.json and results/figures/*.png. Seeded, about 6 minutes on a laptop.
Usage: python experiments/run_all.py [--quick]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import spectrogram  # noqa: E402

from sal import channel, outliers, phy, positioning as pos, ranging, receiver, style  # noqa: E402
from sal.environment import Waveguide, mackenzie, noise_psd_db, thorp, transmission_loss_db  # noqa: E402

OUT = ROOT / "results"
FIG = OUT / "figures"
QUICK = "--quick" in sys.argv
N_PKT = 4 if QUICK else 14
style.apply()
WG = Waveguide()
LINK = dict(rng_m=1000.0, z_tx=10.0, z_rx=30.0, speed=1.0, spread=0.5)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def link_packet(seed, snr, mode="dfe", noise="wenz", doppler=True, speed=None, rng_m=None, whiten=False):
    arr = WG.arrivals(rng_m or LINK["rng_m"], LINK["z_tx"], LINK["z_rx"], phy.FC)
    pkt = phy.make_packet(rng=seed)
    v = LINK["speed"] if speed is None else speed
    y, _ = channel.propagate(pkt.signal, arr, speed=v, c=WG.c, spread_hz=LINK["spread"], rng=seed)
    y = channel.add_noise(y, snr, noise, rng=seed + 10_000)
    return receiver.receive(y, pkt, mode, doppler=doppler, whiten=whiten), pkt


# ------------------------------------------------------------------ 1. environment

def fig_environment():
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    style.title(fig, "The channel: a 50 m deep shallow-water waveguide",
                "Coates-Wenz noise, Thorp absorption and image-method eigenrays; transmitter at 10 m, AUV at 30 m")
    f = np.linspace(0.1, 50, 400)
    ax[0].plot(f, noise_psd_db(f, 0.5, 5), color=style.CYAN, label="shipping 0.5, wind 5 m/s")
    ax[0].plot(f, noise_psd_db(f, 0.9, 12), color=style.AMBER, label="shipping 0.9, wind 12 m/s")
    ax[0].axvspan(4, 8, color=style.TEAL, alpha=0.15, label="modem band 4 to 8 kHz")
    ax[0].set_xscale("log"), ax[0].set_xlabel("frequency (kHz)"), ax[0].set_ylabel("dB re 1 µPa²/Hz")
    ax[0].set_title("Ambient noise"), ax[0].legend(loc="lower left")
    r = np.linspace(50, 5000, 300)
    for fk, colr in zip([2, 6, 15, 30], style.SERIES):
        ax[1].plot(r / 1000, transmission_loss_db(r, fk, k=2.0), color=colr, label=f"{fk} kHz ({thorp(fk):.2f} dB/km)")
    ax[1].invert_yaxis(), ax[1].set_xlabel("range (km)"), ax[1].set_ylabel("transmission loss (dB)")
    ax[1].set_title("Spreading plus absorption"), ax[1].legend(loc="upper right")
    arr = WG.arrivals(1000, 10, 30, phy.FC)
    t0 = arr[0].delay
    for a in arr:
        db = 20 * np.log10(abs(a.gain) / abs(arr[0].gain))
        colr = style.CYAN if a.n_surface == a.n_bottom == 0 else (style.AMBER if a.n_bottom == 0 else
                                                                   (style.TEAL if a.n_surface == 0 else style.VIOLET))
        ax[2].vlines((a.delay - t0) * 1000, -40, db, color=colr, lw=2.4)
        ax[2].plot((a.delay - t0) * 1000, db, "o", color=colr, ms=5)
        k_ = arr.index(a)
        ax[2].annotate(f"S{a.n_surface}B{a.n_bottom}", ((a.delay - t0) * 1000, db), xytext=(3, 4 + 9 * (k_ % 2 == 1 and db > -6)),
                       textcoords="offset points", fontsize=7, color=style.MUTED)
    ax[2].set_ylim(-40, 6), ax[2].set_xlabel("delay after the direct path (ms)"), ax[2].set_ylabel("dB vs direct")
    ax[2].set_title(f"Eigenrays at 1 km: {len(arr)} paths over {(arr[-1].delay - t0) * 1000:.1f} ms")
    fig.tight_layout(rect=style.rect(fig))
    fig.savefig(FIG / "fig1_environment.png", dpi=150)
    plt.close(fig)
    return {"sound_speed_m_s": round(WG.c, 1), "n_paths_1km": len(arr),
            "delay_spread_ms_1km": round((arr[-1].delay - t0) * 1000, 2),
            "delay_spread_symbols": round((arr[-1].delay - t0) * phy.RS, 1), "absorption_db_km_6khz": round(float(thorp(6)), 3)}


# ------------------------------------------------------------------ 2. link: BER versus SNR

def run_ber():
    rows = []
    snrs = [6, 9, 12, 15, 18, 21, 24]
    for mode in receiver.MODES:
        for s in snrs:
            for k in range(N_PKT):
                r, _ = link_packet(100 * k + s, s, mode)
                rows.append({"mode": mode, "snr_db": s, "seed": k, "ber": r.ber, "mse_db": r.mse_db,
                             "a_hat_speed": r.a_hat * WG.c})
        log(f"  BER {mode} done")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "link_ber.csv", index=False)
    return df


def per(df, by):
    g = df.groupby(by)
    return pd.DataFrame({"ber": g["ber"].mean(), "packet_ok_pct": 100 * g["ber"].apply(lambda b: np.mean(b < 1e-2)),
                         "mse_db": g["mse_db"].median()}).reset_index()


def fig_ber(df):
    s = per(df, ["mode", "snr_db"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    style.title(fig, "Adaptive equalisation makes the link work",
                f"1 km, AUV at {LINK['speed']} m/s, surface Doppler spread {LINK['spread']} Hz, QPSK 3 kBd, "
                f"{N_PKT} packets of 2,048 bits per point")
    names = {"single-tap": "single tap + PLL", "linear": "linear RLS equaliser", "dfe": "RLS decision feedback + PLL"}
    cols = {"single-tap": style.CORAL, "linear": style.AMBER, "dfe": style.CYAN}
    for m, g in s.groupby("mode", sort=False):
        colr = cols[m]
        ax[0].semilogy(g["snr_db"], np.maximum(g["ber"], 1e-5), "o-", color=colr, label=names[m])
        ax[1].plot(g["snr_db"], g["packet_ok_pct"], "o-", color=colr, label=names[m])
    ax[0].set_ylim(1e-5, 1), ax[0].set_xlabel("in-band SNR (dB)"), ax[0].set_ylabel("bit error rate (uncoded)")
    ax[0].set_title("Bit error rate (floor drawn at 1e-5)"), ax[0].legend()
    ax[1].set_xlabel("in-band SNR (dB)"), ax[1].set_ylabel("%"), ax[1].set_ylim(-3, 103)
    ax[1].set_title("Packets with BER below 1 %")
    fig.tight_layout(rect=style.rect(fig))
    fig.savefig(FIG / "fig2_ber.png", dpi=150)
    plt.close(fig)
    return s


def fig_constellation():
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    style.title(fig, "What the equaliser does to the received symbols",
                "Same packet, 1 km, 18 dB, AUV at 1 m/s: data symbols after each receiver")
    out = {}
    for a, mode, colr in zip(ax, ["single-tap", "linear", "dfe"], [style.CORAL, style.AMBER, style.CYAN]):
        r, pkt = link_packet(7, 18, mode)
        z = r.eq_out[pkt.n_train:]
        a.scatter(z.real, z.imag, s=4, color=colr, alpha=0.55, lw=0)
        a.scatter(phy.QPSK.real, phy.QPSK.imag, s=70, marker="+", color=style.WHITE)
        a.set_xlim(-2.2, 2.2), a.set_ylim(-2.2, 2.2), a.set_aspect("equal")
        a.set_title(f"{mode}: BER {r.ber:.4f}, MSE {r.mse_db:.1f} dB")
        out[mode] = {"ber": r.ber, "mse_db": round(r.mse_db, 1)}
    fig.tight_layout(rect=style.rect(fig))
    fig.savefig(FIG / "fig3_constellations.png", dpi=150)
    plt.close(fig)
    return out


# ------------------------------------------------------------------ 3. Doppler

def run_doppler():
    rows = []
    for v in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        for comp in (True, False):
            for k in range(N_PKT):
                r, _ = link_packet(500 + k, 18, "dfe", doppler=comp, speed=v)
                rows.append({"speed": v, "compensation": comp, "seed": k, "ber": r.ber,
                             "speed_hat": r.a_hat * WG.c})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "doppler.csv", index=False)
    return df


def fig_doppler(df):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    style.title(fig, "Doppler: estimate it from the preamble and postamble, then resample",
                "RLS-DFE receiver, 1 km, 18 dB; the time-scale estimate comes from the shift of the multipath "
                "signature between the two chirps")
    for comp, colr, lab in [(True, style.CYAN, "with time-scale compensation"),
                            (False, style.CORAL, "PLL only, no resampling")]:
        g = df[df["compensation"] == comp].groupby("speed")["ber"]
        ok = df[df["compensation"] == comp].groupby("speed")["ber"].apply(lambda b: 100 * np.mean(b < 1e-2))
        ax[0].plot(ok.index, ok.values, "o-", color=colr, label=lab)
    ax[0].set_xlabel("closing speed (m/s)"), ax[0].set_ylabel("packets with BER < 1 % (%)"), ax[0].legend()
    ax[0].set_ylim(-3, 103), ax[0].set_title("Packet success versus speed")
    d = df[df["compensation"]]
    ax[1].scatter(d["speed"] + np.random.default_rng(0).normal(0, 0.03, len(d)), d["speed_hat"], s=16,
                  color=style.CYAN, alpha=0.7)
    ax[1].plot([0, 3], [0, 3], color=style.MUTED, lw=1, ls="--")
    err = (d["speed_hat"] - d["speed"]).abs()
    ax[1].set_xlabel("true speed (m/s)"), ax[1].set_ylabel("estimated speed (m/s)")
    ax[1].set_title(f"Speed estimate: median error {err.median() * 100:.1f} cm/s")
    fig.tight_layout(rect=style.rect(fig))
    fig.savefig(FIG / "fig4_doppler.png", dpi=150)
    plt.close(fig)
    return {"speed_error_median_cm_s": round(float(err.median() * 100), 2),
            "speed_error_p95_cm_s": round(float(err.quantile(0.95) * 100), 2)}


# ------------------------------------------------------------------ 4. real hydrophone noise

def run_real():
    rows = []
    for noise in ("wenz", "real"):
        for s in [6, 9, 12, 15, 18, 21]:
            for k in range(N_PKT):
                r, _ = link_packet(900 + 31 * k + s, s, "dfe", noise=noise)
                rows.append({"noise": noise, "snr_db": s, "seed": k, "ber": r.ber, "mse_db": r.mse_db})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "real_noise_ber.csv", index=False)
    # detection: matched-filter peaks on noise only, with and without pre-whitening
    det = []
    rng = np.random.default_rng(3)
    n = int(0.6 * phy.FS)
    for kind in ("wenz", "real"):
        for k in range(150 if not QUICK else 30):
            w = channel.wenz_noise(n, rng) if kind == "wenz" else channel.real_noise(n, rng)
            for wh in (False, True):
                c = phy.matched_filter(phy.whiten(w) if wh else w, phy.chirp())
                det.append({"noise": kind, "whiten": wh, "max_corr": float(c.max())})
    det = pd.DataFrame(det)
    det.to_csv(OUT / "noise_only_detection.csv", index=False)
    return df, det


def band_stats():
    rec = channel.real_recording()
    bb = channel.to_baseband(rec)
    x = bb.real
    kurt = float(np.mean((x - x.mean()) ** 4) / x.var() ** 2)
    g = channel.wenz_noise(len(bb), 1).real
    return {"recording_seconds": round(len(rec) / phy.FS, 1), "band_kurtosis_real": round(kurt, 2),
            "band_kurtosis_gaussian": round(float(np.mean((g - g.mean()) ** 4) / g.var() ** 2), 2)}


def fig_real(df, det):
    rec = channel.real_recording()
    fig = plt.figure(figsize=(13, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1])
    style.title(fig, "Real hydrophone noise: heavier tails, same link performance at equal SNR",
                "Glacier Bay recording (US National Park Service, public domain), 60 s, humpback whale song and "
                "ambient noise, injected into the simulated link at the same in-band SNR")
    a0 = fig.add_subplot(gs[0])
    f, t, S = spectrogram(rec, phy.FS, nperseg=1024, noverlap=768)
    L = 10 * np.log10(S + 1e-20)
    a0.pcolormesh(t, f / 1000, L, shading="auto", cmap="magma", vmin=np.percentile(L, 40), vmax=np.percentile(L, 99.7))
    a0.axhspan(4, 8, fill=False, ec=style.TEAL, lw=1.5)
    a0.text(1, 8.3, "modem band", color=style.TEAL, fontsize=8)
    a0.set_ylim(0, 11), a0.set_xlabel("time (s)"), a0.set_ylabel("kHz"), a0.set_title("Spectrogram of the recording")
    a0.grid(False)
    a1 = fig.add_subplot(gs[1])
    s = df.groupby(["noise", "snr_db"])["ber"].apply(lambda b: 100 * np.mean(b < 1e-2)).reset_index()
    for kind, colr, lab in [("wenz", style.CYAN, "Gaussian, Wenz-shaped"), ("real", style.AMBER, "real recording")]:
        g = s[s["noise"] == kind]
        a1.plot(g["snr_db"], g["ber"], "o-", color=colr, label=lab)
    a1.set_xlabel("in-band SNR (dB)"), a1.set_ylabel("packets with BER < 1 % (%)"), a1.set_ylim(-3, 103)
    a1.set_title("Link success, same SNR"), a1.legend()
    a2 = fig.add_subplot(gs[2])
    bins = np.linspace(0, 0.5, 60)
    for (kind, wh), colr, lab in [(("wenz", False), style.CYAN, "Gaussian"), (("real", False), style.AMBER, "real"),
                                  (("real", True), style.TEAL, "real, pre-whitened")]:
        v = det[(det["noise"] == kind) & (det["whiten"] == wh)]["max_corr"]
        a2.hist(v, bins=bins, color=colr, alpha=0.55, label=lab)
    a2.axvline(0.3, color=style.CORAL, ls="--", lw=1.2)
    a2.text(0.305, a2.get_ylim()[1] * 0.85, "detection\nthreshold 0.3", color=style.CORAL, fontsize=7.5)
    a2.set_xlabel("max normalised correlation, noise only"), a2.set_ylabel("trials")
    a2.set_title("False alarms of the chirp detector"), a2.legend()
    fig.tight_layout(rect=style.rect(fig))
    fig.savefig(FIG / "fig5_real_noise.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ 5. ranging

def run_ranging():
    from sal.environment import Arrival
    rows = []
    rng = np.random.default_rng(11)
    # (a) single path: estimator against the Cramer-Rao bound
    for s in [-6, -3, 0, 3, 6, 10, 15, 20]:
        for k in range(60 if not QUICK else 15):
            arr = [Arrival(0.4, 1e-3 * np.exp(1j * rng.uniform(0, 2 * np.pi)), 0, 0, 0, 0)]
            y, lead = channel.propagate(phy.chirp(), arr, c=WG.c, lead=0.02, tail=0.02)
            p = np.abs(y) ** 2
            y = channel.add_noise(y, s, "wenz", rng, ref_power=p[p > 0.05 * p.max()].mean())
            t = ranging.estimate_toa(y, lead)
            rows.append({"exp": "crlb", "snr_db": s, "err_s": t.t_first})
    crlb = pd.DataFrame(rows)
    # (b) multipath: first-path vs strongest-peak estimators over range
    rr = []
    for h in [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800]:
        for k in range(30 if not QUICK else 8):
            p = ranging.twtt_ping(WG, h, 30, 49, radial_speed=0.0, rng=1000 * h + k, sl=165, spread_hz=0.5)
            rr.append({"horiz": h, "err_first": p["r_first"] - p["slant"], "err_max": p["r_max"] - p["slant"],
                       "snr_db": p["snr_db"]})
    mp = pd.DataFrame(rr)
    # (c) Doppler bias and its correction from the DVL speed
    dd = []
    for v in [-2, -1, 0, 1, 2]:
        for corr in (False, True):
            for k in range(15 if not QUICK else 5):
                p = ranging.twtt_ping(WG, 600, 30, 49, radial_speed=v, rng=77 + k, sl=165,
                                      speed_known=(v if corr else None))
                dd.append({"speed": v, "corrected": corr, "err": p["r_first"] - p["slant"]})
    dop = pd.DataFrame(dd)
    crlb.to_csv(OUT / "ranging_crlb.csv", index=False)
    mp.to_csv(OUT / "ranging_multipath.csv", index=False)
    dop.to_csv(OUT / "ranging_doppler.csv", index=False)
    return crlb, mp, dop


def fig_ranging(crlb, mp, dop):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    style.title(fig, "Ranging: as good as the theory allows, if the first path is the one you pick",
                "Two-way travel time with a 50 ms LFM chirp, 3.8 kHz swept; beacon on the seabed, AUV at 30 m")
    g = crlb.groupby("snr_db")["err_s"]
    std = g.apply(lambda e: np.sqrt(np.mean((e - e.mean()) ** 2))) * WG.c * 100
    snr = np.array(sorted(crlb["snr_db"].unique()), float)
    ax[0].semilogy(snr, std.values, "o-", color=style.CYAN, label="measured (one leg)")
    ax[0].semilogy(snr, ranging.crlb_toa_std(snr) * WG.c * 100, "--", color=style.AMBER, label="Cramér-Rao bound")
    ax[0].set_xlabel("in-band SNR (dB)"), ax[0].set_ylabel("range standard deviation (cm)")
    ax[0].set_title("Time-of-arrival estimator vs the bound"), ax[0].legend()
    s = mp.groupby("horiz").agg(first=("err_first", lambda e: np.mean(np.abs(e) > 0.5) * 100),
                                strongest=("err_max", lambda e: np.mean(np.abs(e) > 0.5) * 100))
    ax[1].plot(s.index / 1000, s["strongest"], "o-", color=style.CORAL, label="strongest peak")
    ax[1].plot(s.index / 1000, s["first"], "o-", color=style.CYAN, label="first path (> 50 % of max)")
    ax[1].set_xlabel("horizontal range (km)"), ax[1].set_ylabel("ranges off by more than 0.5 m (%)")
    ax[1].set_ylim(-3, 103), ax[1].set_title("Multipath outliers"), ax[1].legend()
    for corr, colr, lab in [(False, style.CORAL, "raw"), (True, style.TEAL, "corrected with DVL speed")]:
        d = dop[dop["corrected"] == corr].groupby("speed")["err"].mean() * 100
        ax[2].plot(d.index, d.values, "o-", color=colr, label=lab)
    ax[2].axhline(0, color=style.MUTED, lw=0.8)
    ax[2].set_xlabel("closing speed (m/s)"), ax[2].set_ylabel("mean range error (cm)")
    ax[2].set_title("Range-Doppler coupling of the chirp"), ax[2].legend()
    fig.tight_layout(rect=style.rect(fig))
    fig.savefig(FIG / "fig6_ranging.png", dpi=150)
    plt.close(fig)
    hi = crlb[crlb["snr_db"] >= 10].groupby("snr_db")["err_s"].std()
    return {"toa_std_cm_at_20db": round(float(std.loc[20]), 2),
            "crlb_cm_at_20db": round(float(ranging.crlb_toa_std(20) * WG.c * 100), 2),
            "efficiency_at_20db": round(float(ranging.crlb_toa_std(20) * WG.c * 100 / std.loc[20]), 2),
            "outliers_first_pct": round(float((mp["err_first"].abs() > 0.5).mean() * 100), 1),
            "outliers_strongest_pct": round(float((mp["err_max"].abs() > 0.5).mean() * 100), 1),
            "doppler_bias_cm_at_2ms_raw": round(float(dop[(dop.speed == 2) & ~dop.corrected]["err"].mean() * 100), 1),
            "doppler_bias_cm_at_2ms_corrected": round(float(dop[(dop.speed == 2) & dop.corrected]["err"].mean() * 100),
                                                      1)}


# ------------------------------------------------------------------ 6. ML outlier detection

def ping_dataset(n, seed, noise_mix=True):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n):
        h = rng.uniform(200, 1900)
        v = rng.uniform(-2, 2)
        kind = rng.choice(["wenz", "real"]) if noise_mix else "wenz"
        p = ranging.twtt_ping(WG, h, 30, 49, radial_speed=v, rng=int(rng.integers(1e9)), sl=rng.uniform(158, 172),
                              spread_hz=rng.uniform(0.2, 1.0), noise=kind, speed_known=v)
        rows.append({**p, "noise": kind, "speed": v})
    return pd.DataFrame(rows)


def run_ml():
    from sklearn.metrics import average_precision_score, roc_auc_score
    n = 300 if QUICK else 1600
    train = ping_dataset(n, 1)
    test = ping_dataset(n // 2, 2)
    train.to_csv(OUT / "pings_train.csv", index=False)
    test.to_csv(OUT / "pings_test.csv", index=False)
    model = outliers.train(train)
    p = model.predict_proba(test[outliers.FEATURES])[:, 1]
    y = outliers.label(test)
    imp = outliers.importance(model)
    imp.to_csv(OUT / "outlier_feature_importance.csv", index=False)
    pred = p > 0.5
    res = {"train_pings": len(train), "test_pings": len(test), "outlier_rate_test_pct": round(100 * y.mean(), 1),
           "auc": round(float(roc_auc_score(y, p)), 3), "average_precision": round(float(average_precision_score(y, p)), 3),
           "recall_at_0_5": round(float((pred & (y == 1)).sum() / max(1, y.sum())), 3),
           "precision_at_0_5": round(float((pred & (y == 1)).sum() / max(1, pred.sum())), 3)}
    return model, res, imp


# ------------------------------------------------------------------ 7. LBL navigation

def run_navigation(model):
    track = pos.lawnmower()
    vel = pos.dvl_velocity(track, rng=5)
    dr = pos.dead_reckoning(track[0, 1:3], vel)
    every = 10
    rng = np.random.default_rng(21)
    pings = []
    for i in range(0, len(track), every):
        p_true = track[i, 1:3]
        for b_id, b in enumerate(pos.BEACONS):
            horiz = float(np.hypot(*(p_true - b[:2])))
            u = (b[:2] - p_true) / max(horiz, 1e-6)
            radial = float(track[i, 4:6] @ u)                      # closing speed towards the beacon
            radial_meas = float(vel[i] @ u)
            pg = ranging.twtt_ping(WG, horiz, pos.DEPTH, b[2], radial_speed=radial, rng=int(rng.integers(1e9)),
                                   sl=165, spread_hz=0.5, speed_known=radial_meas)
            pings.append({"i": i, "beacon": b_id, **pg})
        if i % 600 == 0:
            log(f"  navigation ping epoch {i // every}/{len(track) // every}")
    pg = pd.DataFrame(pings)
    pg["p_outlier"] = model.predict_proba(pg[outliers.FEATURES])[:, 1]
    pg["true_outlier"] = (pg["r_first"] - pg["slant"]).abs() > outliers.THRESHOLD_M
    pg.to_csv(OUT / "navigation_pings.csv", index=False)
    by_i = {i: g.sort_values("beacon") for i, g in pg.groupby("i")}
    est = {"dead reckoning": dr}
    ls = np.full_like(dr, np.nan)
    guess = track[0, 1:3]
    for i, g in by_i.items():
        ls[i] = guess = pos.least_squares(g["r_first"].to_numpy(), guess=guess)
    est["least squares per epoch"] = ls
    stats = {}
    for name, gate, ml, bias in [("EKF, position only", False, False, False),
                                 ("EKF, position only + NIS gate", True, False, False),
                                 ("EKF + DVL calibration", False, False, True),
                                 ("EKF + calibration + NIS gate", True, False, True),
                                 ("EKF + calibration + gate + ML", True, True, True)]:
        f = pos.Ekf(track[0, 1:3].copy(), estimate_bias=bias)
        xs = np.zeros_like(dr)
        rejected, rejected_true, ml_dropped = 0, 0, 0
        for i in range(len(track)):
            if i > 0:
                f.predict(vel[i - 1])
            if i in by_i:
                for _, row in by_i[i].iterrows():
                    if ml and row["p_outlier"] > 0.5:
                        ml_dropped += 1
                        continue
                    ok, _ = f.update(row["r_first"], pos.BEACONS[int(row["beacon"])], use_gate=gate)
                    if not ok:
                        rejected += 1
                        rejected_true += int(row["true_outlier"])
            xs[i] = f.xy
        est[name] = xs
        stats[name] = {"gate_rejections": rejected, "gate_rejections_true_outliers": rejected_true,
                       "ml_dropped": ml_dropped, "heading_bias_est_deg": round(float(np.degrees(f.x[2])), 2),
                       "scale_err_est_pct": round(float(f.x[3] * 100), 2)}
    rows = []
    for name, xs in est.items():
        e = np.hypot(*(xs - track[:, 1:3]).T)
        e = e[~np.isnan(e)]
        rows.append({"method": name, "rmse_m": round(float(np.sqrt(np.mean(e ** 2))), 3),
                     "p95_m": round(float(np.percentile(e, 95)), 3), "max_m": round(float(e.max()), 2),
                     "final_m": round(float(e[-1]), 3), **stats.get(name, {})})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "navigation_results.csv", index=False)
    np.savez_compressed(OUT / "navigation_tracks.npz", truth=track, **{k.replace(" ", "_").replace("+", "p"): v
                                                                       for k, v in est.items()})
    return track, est, res, pg


def fig_navigation(track, est, res, pg):
    fig = plt.figure(figsize=(13, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35])
    style.title(fig, "LBL navigation: from metres of drift to centimetres",
                f"Lawnmower survey, {track[-1, 0] / 3600:.1f} h at 1.5 m/s, four seabed beacons, a ping to each every "
                "10 s; DVL with 0.3 % scale error and a 1° compass bias")
    a0 = fig.add_subplot(gs[0])
    a0.plot(track[:, 1], track[:, 2], color=style.WHITE, lw=1.0, label="truth")
    a0.plot(*est["dead reckoning"].T, color=style.CORAL, lw=1.2, label="dead reckoning")
    a0.plot(*est["EKF + calibration + gate + ML"].T, color=style.CYAN, lw=1.0, ls="--", label="EKF + calibration + gate + ML")
    a0.scatter(pos.BEACONS[:, 0], pos.BEACONS[:, 1], marker="^", s=90, color=style.AMBER, zorder=5, label="beacons")
    bad = pg[(pg["r_first"] - pg["slant"]).abs() > 0.5]
    if len(bad):
        xy = track[bad["i"].to_numpy(), 1:3]
        a0.scatter(xy[:, 0], xy[:, 1], s=7, color=style.VIOLET, alpha=0.45, lw=0, label="multipath outlier in a ping")
    a0.set_aspect("equal"), a0.set_xlabel("east (m)"), a0.set_ylabel("north (m)")
    a0.set_xlim(-120, 1620), a0.set_ylim(-120, 1620)
    a0.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=7.8)
    a0.set_title("Track")
    a1 = fig.add_subplot(gs[1])
    t = track[:, 0] / 60
    for (name, colr) in [("dead reckoning", style.CORAL), ("EKF, position only + NIS gate", style.MUTED),
                         ("least squares per epoch", style.VIOLET), ("EKF + DVL calibration", style.AMBER),
                         ("EKF + calibration + gate + ML", style.CYAN)]:
        e = np.hypot(*(est[name] - track[:, 1:3]).T)
        e = pd.Series(e).interpolate(limit_direction="both").to_numpy()
        sm = np.sqrt(pd.Series(e ** 2).rolling(120, center=True, min_periods=1).mean().to_numpy())
        rm = res.set_index("method").loc[name, "rmse_m"]
        a1.semilogy(t, np.maximum(sm, 1e-3), color=colr, lw=1.8, label=f"{name}: {rm * 100:.1f} cm" if rm < 1
                    else f"{name}: {rm:.1f} m")
    a1.set_xlabel("time (min)"), a1.set_ylabel("horizontal error, 2-min RMS (m)")
    a1.legend(fontsize=7.8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    a1.set_title("Position error over the survey (RMSE in the legend)")
    fig.tight_layout(rect=style.rect(fig))
    fig.savefig(FIG / "fig7_navigation.png", dpi=150)
    plt.close(fig)


def to_py(o):
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    return o.item() if hasattr(o, "item") else o


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    summ = {"config": {"fs_hz": phy.FS, "fc_hz": phy.FC, "symbol_rate_baud": phy.RS, "roll_off": phy.BETA,
                       "chirp_ms": phy.CHIRP_T * 1000, "chirp_bw_hz": phy.CHIRP_B, "packets_per_point": N_PKT,
                       "link": LINK}}
    log("environment"), summ.update({"environment": fig_environment()})
    log("BER"), (ber := run_ber())
    s = fig_ber(ber)
    summ["ber"] = s.to_dict(orient="records")
    log("constellations"), summ.update({"constellation_example": fig_constellation()})
    log("Doppler"), (dop := run_doppler())
    summ["doppler"] = fig_doppler(dop)
    summ["doppler"]["packet_ok_pct"] = (dop.groupby(["speed", "compensation"])["ber"]
                                        .apply(lambda b: round(100 * np.mean(b < 1e-2), 1)).reset_index()
                                        .to_dict(orient="records"))
    log("real noise"), (real := run_real())
    fig_real(*real)
    det = real[1]
    summ["real_noise"] = {**band_stats(),
                          "packet_ok_pct": (real[0].groupby(["noise", "snr_db"])["ber"]
                                            .apply(lambda b: round(100 * np.mean(b < 1e-2), 1)).reset_index()
                                            .to_dict(orient="records")),
                          "false_alarm_pct_at_0_2": (det.groupby(["noise", "whiten"])["max_corr"]
                                                     .apply(lambda v: round(100 * np.mean(v > 0.2), 1))
                                                     .reset_index().to_dict(orient="records")),
                          "false_alarm_pct_at_0_3": (det.groupby(["noise", "whiten"])["max_corr"]
                                                     .apply(lambda v: round(100 * np.mean(v > 0.3), 1)).reset_index()
                                                     .to_dict(orient="records")),
                          "noise_only_max_corr_p99": (det.groupby(["noise", "whiten"])["max_corr"]
                                                      .quantile(0.99).round(3).reset_index().to_dict(orient="records"))}
    log("ranging"), (rg := run_ranging())
    summ["ranging"] = fig_ranging(*rg)
    log("ML outlier model"), (ml := run_ml())
    model, summ["ml_outliers"], imp = ml
    summ["ml_outliers"]["importance"] = imp.round(1).to_dict(orient="records")
    summ["navigation_outlier_rate_pct"] = None
    log("navigation"), (nav := run_navigation(model))
    fig_navigation(*nav)
    summ["navigation"] = nav[2].to_dict(orient="records")
    summ["navigation_outlier_rate_pct"] = round(float(nav[3]["true_outlier"].mean() * 100), 1)
    summ["navigation_pings"] = int(len(nav[3]))
    summ["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "summary.json").write_text(json.dumps(to_py(summ), indent=1))
    log(f"done in {summ['runtime_s']} s")


if __name__ == "__main__":
    main()
