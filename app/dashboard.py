"""Subsea Acoustic Link: interactive lab. Run with: streamlit run app/dashboard.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from sal import channel, phy, positioning as pos, ranging, receiver, style  # noqa: E402
from sal.environment import Waveguide, transmission_loss_db  # noqa: E402

RES = ROOT / "results"
style.apply()
st.set_page_config(page_title="Subsea Acoustic Link", page_icon="🌊", layout="wide")

st.markdown("""
<style>
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #0d3557 0%, #061726 55%) fixed; }
section.main > div { padding-top: 1.2rem; }
h1, h2, h3, h4, p, li, label, span, div { color: #e6f1ff; }
.hero { padding: 26px 30px; border-radius: 18px; margin-bottom: 18px;
        background: linear-gradient(120deg, rgba(63,208,255,.16), rgba(46,230,166,.08) 60%, rgba(255,181,71,.06));
        border: 1px solid rgba(63,208,255,.25); }
.hero h1 { font-size: 2.3rem; margin: 0 0 6px 0; letter-spacing: .3px; }
.hero p { color: #8fb3d1; font-size: 1.02rem; margin: 0; max-width: 980px; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 4px 0 14px; }
.kpi { background: rgba(11,34,54,.85); border: 1px solid #1d3a55; border-radius: 14px; padding: 14px 16px; }
.kpi b { display: block; font-size: 1.65rem; color: #3fd0ff; line-height: 1.15; }
.kpi span { color: #8fb3d1; font-size: .84rem; }
.find { background: rgba(11,34,54,.7); border-left: 3px solid #2ee6a6; border-radius: 8px; padding: 10px 14px;
        margin: 6px 0; color: #e6f1ff; }
.small { color: #8fb3d1; font-size: .85rem; }
div[data-testid="stTabs"] button p { font-size: 1rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def summary():
    p = RES / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


S = summary()


def kpis(items):
    st.markdown('<div class="kpis">' + "".join(f'<div class="kpi"><b>{v}</b><span>{k}</span></div>' for v, k in items)
                + "</div>", unsafe_allow_html=True)


def show(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


st.markdown("""<div class="hero"><h1>🌊 Subsea Acoustic Link</h1>
<p>Communication and positioning for underwater robots, where there is no GPS and sound is the only link: a
shallow-water channel with multipath, Doppler and real hydrophone noise; a single-carrier modem with Doppler
compensation and an adaptive equaliser; chirp ranging at the Cramér-Rao bound; LBL navigation of an AUV with
self-calibrating Kalman filtering and machine-learning outlier screening.</p></div>""", unsafe_allow_html=True)

tabs = st.tabs(["Overview", "Live link", "Channel", "Ranging", "Navigation", "Real hydrophone data", "Method"])

# ------------------------------------------------------------------ overview
with tabs[0]:
    if S is None:
        st.warning("Run `python experiments/run_all.py` first to produce the results.")
    else:
        nav = {r["method"]: r for r in S["navigation"]}
        best = nav["EKF + calibration + gate + ML"]
        ber = pd.DataFrame(S["ber"])
        dfe18 = ber[(ber["mode"] == "dfe") & (ber["snr_db"] == 18)]["packet_ok_pct"].iloc[0]
        kpis([(f"{dfe18:.0f} %", "packets decoded from 18 dB, 1 km, 1 m/s (RLS-DFE)"),
              (f"{S['doppler']['speed_error_median_cm_s']} cm/s", "median error of the Doppler speed estimate"),
              (f"{S['ranging']['toa_std_cm_at_20db']} / {S['ranging']['crlb_cm_at_20db']} cm",
               "range std vs Cramér-Rao bound at 20 dB"),
              (f"{best['rmse_m'] * 100:.1f} cm", "LBL navigation RMSE over the survey"),
              (f"{S['ml_outliers']['auc']:.2f}", "AUC of the multipath-outlier classifier")])
        c1, c2 = st.columns(2)
        with c1:
            st.image(str(RES / "figures" / "fig2_ber.png"), use_container_width=True)
        with c2:
            st.image(str(RES / "figures" / "fig7_navigation.png"), use_container_width=True)
        st.markdown("#### What the study shows")
        dr = nav["dead reckoning"]["rmse_m"]
        lock = nav["EKF, position only + NIS gate"]["rmse_m"]
        for t in [
            f"The channel at 1 km has {S['environment']['n_paths_1km']} significant paths spread over "
            f"{S['environment']['delay_spread_ms_1km']} ms, i.e. {S['environment']['delay_spread_symbols']:.0f} symbols: "
            "a single-tap receiver cannot decode, a linear equaliser stalls on an error floor, the RLS decision-feedback "
            "equaliser with a PLL decodes every packet from 18 dB (86 % at 15 dB).",
            "Without time-scale compensation, packets fail from 1.5 m/s; estimating the Doppler from the shift of the "
            "multipath signature between preamble and postamble keeps every packet up to 3 m/s.",
            f"The first-path TOA estimator reaches the Cramér-Rao bound at high SNR; picking the strongest peak instead "
            f"puts {S['ranging']['outliers_strongest_pct']} % of ranges off by more than 0.5 m.",
            f"Dead reckoning drifts to {dr} m RMSE. A gated filter that ignores the DVL calibration errors locks itself "
            f"out ({lock} m); adding heading-bias and scale states brings the error to "
            f"{best['rmse_m'] * 100:.1f} cm and recovers the 1° compass bias.",
        ]:
            st.markdown(f'<div class="find">{t}</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ live link
with tabs[1]:
    st.markdown("Send one packet through the simulated ocean and decode it. Everything below is computed live.")
    c = st.columns(4)
    rng_m = c[0].slider("Range (m)", 200, 2500, 1000, 100)
    speed = c[1].slider("Closing speed (m/s)", 0.0, 3.0, 1.0, 0.25)
    snr = c[2].slider("In-band SNR (dB)", 0, 30, 18)
    spread = c[3].slider("Surface Doppler spread (Hz)", 0.0, 2.0, 0.5, 0.1)
    c = st.columns(4)
    mode = c[0].selectbox("Receiver", list(receiver.MODES), index=2)
    noise = c[1].selectbox("Noise", ["Gaussian (Wenz-shaped)", "Real hydrophone recording"])
    comp = c[2].toggle("Doppler compensation", value=True)
    seed = c[3].number_input("Seed", 0, 9999, 7)

    @st.cache_data(show_spinner="Propagating and decoding...")
    def live(rng_m, speed, snr, spread, mode, noise, comp, seed):
        wg = Waveguide()
        arr = wg.arrivals(rng_m, 10, 30, phy.FC)
        pkt = phy.make_packet(rng=seed)
        y, _ = channel.propagate(pkt.signal, arr, speed=speed, c=wg.c, spread_hz=spread, rng=seed)
        y = channel.add_noise(y, snr, "wenz" if noise.startswith("Gaussian") else "real", rng=seed + 1)
        r = receiver.receive(y, pkt, mode, doppler=comp)
        cc = phy.matched_filter(y, phy.chirp())
        return arr, y, r, pkt, cc, wg.c

    arr, y, r, pkt, cc, cw = live(rng_m, speed, snr, spread, mode, noise, comp, int(seed))
    if not r.detected:
        kpis([("not detected", "the chirp correlation stayed below the threshold")])
    else:
        kpis([(f"{r.ber:.4f}", "bit error rate (uncoded, 2,048 bits)"),
              (f"{r.mse_db:.1f} dB", "equaliser output MSE"),
              (f"{r.a_hat * cw:.2f} m/s", f"estimated closing speed (true {speed:.2f})"),
              (f"{len(arr)}", "eigenrays above -30 dB")])
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    t0 = arr[0].delay
    for a in arr:
        db = 20 * np.log10(abs(a.gain) / max(abs(b.gain) for b in arr))
        ax[0].vlines((a.delay - t0) * 1000, -32, db, color=style.CYAN if a.n_surface == 0 else style.AMBER, lw=2.2)
    ax[0].set_ylim(-32, 3), ax[0].set_xlabel("delay (ms)"), ax[0].set_ylabel("dB"), ax[0].set_title(
        "Channel impulse response (amber: surface bounces)")
    tt = np.arange(len(cc)) / phy.FS
    ax[1].plot(tt, cc, color=style.CYAN, lw=0.8)
    ax[1].set_xlabel("time (s)"), ax[1].set_ylabel("normalised correlation"), ax[1].set_title(
        "Chirp matched filter: preamble and postamble")
    if r.detected:
        z = r.eq_out[pkt.n_train:]
        ax[2].scatter(z.real, z.imag, s=4, alpha=.6, color=style.TEAL, lw=0)
    ax[2].scatter(phy.QPSK.real, phy.QPSK.imag, marker="+", s=90, color="white")
    ax[2].set_xlim(-2.2, 2.2), ax[2].set_ylim(-2.2, 2.2), ax[2].set_aspect("equal"), ax[2].set_title(
        "Equalised data symbols")
    fig.tight_layout()
    show(fig)
    if r.detected and len(r.eq_out):
        e = np.abs(r.eq_out - pkt.symbols) ** 2
        k = 32
        fig, ax = plt.subplots(figsize=(15, 2.6))
        sm = np.convolve(e, np.ones(k) / k, mode="valid")
        ax.plot(np.arange(len(sm)), 10 * np.log10(sm + 1e-6), color=style.AMBER)
        ax.axvspan(0, pkt.n_train, color=style.CYAN, alpha=.12)
        ax.text(5, ax.get_ylim()[1] - 2, "training", color=style.CYAN, fontsize=8)
        ax.set_xlabel("symbol"), ax.set_ylabel("squared error (dB)"), ax.set_title("Equaliser learning curve")
        fig.tight_layout()
        show(fig)

# ------------------------------------------------------------------ channel
with tabs[2]:
    c = st.columns(5)
    depth = c[0].slider("Water depth (m)", 20, 200, 50, 5)
    temp = c[1].slider("Temperature (°C)", 2, 28, 10)
    sal = c[2].slider("Salinity (ppt)", 25, 40, 35)
    wave = c[3].slider("Surface roughness rms (m)", 0.0, 1.0, 0.2, 0.05)
    rr = c[4].slider("Range (m)", 100, 5000, 1000, 100)
    wg = Waveguide(depth=depth, temperature=temp, salinity=sal, wave_rms=wave)
    arr = wg.arrivals(rr, min(10, depth - 1), min(30, depth - 1), phy.FC)
    kpis([(f"{wg.c:.1f} m/s", "sound speed (Mackenzie)"), (f"{len(arr)}", "eigenrays above -30 dB"),
          (f"{(arr[-1].delay - arr[0].delay) * 1000:.1f} ms", "delay spread"),
          (f"{transmission_loss_db(rr, 6.0, 2.0):.1f} dB", "transmission loss at 6 kHz")])
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    for a in arr:
        db = 20 * np.log10(abs(a.gain) / max(abs(b.gain) for b in arr))
        ax[0].vlines((a.delay - arr[0].delay) * 1000, -32, db, color=style.SERIES[min(a.n_bottom, 4)], lw=2.4)
        ax[0].annotate(f"S{a.n_surface}B{a.n_bottom}", ((a.delay - arr[0].delay) * 1000, db), xytext=(2, 3),
                       textcoords="offset points", fontsize=7, color=style.MUTED)
    ax[0].set_ylim(-32, 4), ax[0].set_xlabel("delay (ms)"), ax[0].set_ylabel("dB vs strongest"), ax[0].set_title(
        "Eigenrays (colour = bottom bounces)")
    zt, zr_ = min(10, depth - 1), min(30, depth - 1)
    xs = np.linspace(0, rr, 400)
    for a in arr:
        dz = a.length * np.sin(np.radians(a.grazing_deg))
        for sgn in (1, -1):
            u = zt + sgn * xs * dz / rr                      # unfolded depth along the straight image path
            n_surf = int(np.sum(np.diff(np.floor(u / (2 * depth))) != 0))
            n_bot = int(np.sum(np.diff(np.floor((u - depth) / (2 * depth))) != 0))
            zf = np.abs(((u + depth) % (2 * depth)) - depth)
            if abs(zf[-1] - zr_) < 0.5 and n_surf == a.n_surface and n_bot == a.n_bottom:
                ax[1].plot(xs, zf, color=style.SERIES[min(a.n_bottom, 4)], lw=1.1, alpha=.85)
                break
    ax[1].plot([0], [zt], "o", color="white"), ax[1].plot([rr], [zr_], "s", color="white")
    ax[1].invert_yaxis(), ax[1].set_xlabel("range (m)"), ax[1].set_ylabel("depth (m)"), ax[1].set_title(
        "Ray geometry (image method, unfolded rays drawn folded)")
    fig.tight_layout()
    show(fig)
    st.image(str(RES / "figures" / "fig1_environment.png"), use_container_width=True)

# ------------------------------------------------------------------ ranging
with tabs[3]:
    if S:
        rgs = S["ranging"]
        kpis([(f"{rgs['toa_std_cm_at_20db']} cm", "range std at 20 dB (one leg)"),
              (f"{rgs['crlb_cm_at_20db']} cm", "Cramér-Rao bound at 20 dB"),
              (f"{rgs['outliers_strongest_pct']} %", "outliers with the strongest-peak estimator"),
              (f"{rgs['outliers_first_pct']} %", "outliers with the first-path estimator"),
              (f"{rgs['doppler_bias_cm_at_2ms_raw']} → {rgs['doppler_bias_cm_at_2ms_corrected']} cm",
               "Doppler bias at 2 m/s, raw → corrected")])
        st.image(str(RES / "figures" / "fig6_ranging.png"), use_container_width=True)
    st.markdown("##### One two-way ping, live")
    c = st.columns(3)
    h = c[0].slider("Horizontal range to the beacon (m)", 100, 2000, 1500, 50)
    sl = c[1].slider("Source level (dB re 1 µPa @ 1 m)", 150, 185, 165)
    sd = c[2].number_input("Ping seed", 0, 9999, 3)
    p = ranging.twtt_ping(Waveguide(), h, 30, 49, rng=int(sd), sl=sl, spread_hz=0.5)
    kpis([(f"{p['slant']:.2f} m", "true slant range"), (f"{p['r_first'] - p['slant']:+.3f} m", "first-path error"),
          (f"{p['r_max'] - p['slant']:+.3f} m", "strongest-peak error"), (f"{p['snr_db']:.1f} dB", "SNR (sonar eq.)")])

# ------------------------------------------------------------------ navigation
with tabs[4]:
    if S:
        df = pd.DataFrame(S["navigation"])
        best = df.set_index("method").loc["EKF + calibration + gate + ML"]
        kpis([(f"{df.set_index('method').loc['dead reckoning', 'rmse_m']:.1f} m", "dead reckoning RMSE"),
              (f"{best['rmse_m'] * 100:.1f} cm", "best filter RMSE"),
              (f"{best['heading_bias_est_deg']}°", "compass bias recovered (true 1°)"),
              (f"{S['navigation_outlier_rate_pct']} %", f"multipath outliers among {S['navigation_pings']} pings")])
        st.image(str(RES / "figures" / "fig7_navigation.png"), use_container_width=True)
        show_cols = ["method", "rmse_m", "p95_m", "max_m", "gate_rejections", "gate_rejections_true_outliers",
                     "ml_dropped"]
        st.dataframe(df[[c for c in show_cols if c in df]], hide_index=True, use_container_width=True)
        st.markdown("##### Machine learning on the matched-filter output")
        m = S["ml_outliers"]
        kpis([(f"{m['auc']}", "ROC AUC on unseen pings"), (f"{m['average_precision']}", "average precision"),
              (f"{m['outlier_rate_test_pct']} %", "outlier rate in the test pings")])
        imp = pd.DataFrame(m["importance"])
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.barh(imp["feature"][::-1], imp["gain_pct"][::-1], color=style.CYAN)
        ax.set_xlabel("share of LightGBM gain (%)"), ax.set_title("What tells a corrupted range apart")
        fig.tight_layout()
        show(fig)

# ------------------------------------------------------------------ real data
with tabs[5]:
    st.markdown("A 60-second recording from a hydrophone in Glacier Bay, Alaska (US National Park Service, public "
                "domain): humpback whale song over ambient noise. The study injects it into the simulated link, at the "
                "same in-band SNR as the Gaussian noise, to see what the textbook assumptions miss.")
    st.audio(str(channel.REAL))
    if S:
        rn = S["real_noise"]
        fa = {(d["noise"], d["whiten"]): d["max_corr"] for d in rn["false_alarm_pct_at_0_3"]}
        kpis([(f"{rn['band_kurtosis_real']}", "kurtosis in the modem band (Gaussian: 3)"),
              (f"{fa.get(('real', False), 0)} %", "detector false alarms on real noise"),
              (f"{fa.get(('wenz', False), 0)} %", "false alarms on Gaussian noise")])
        st.image(str(RES / "figures" / "fig5_real_noise.png"), use_container_width=True)

# ------------------------------------------------------------------ method
with tabs[6]:
    st.markdown((ROOT / "docs" / "method.md").read_text() if (ROOT / "docs" / "method.md").exists() else "")
