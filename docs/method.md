## Method

**Channel.** Isovelocity shallow-water waveguide (50 m, sound speed from Mackenzie 1981). Eigenrays by the image
method; each ray carries spherical spreading, Thorp absorption, a −1 surface reflection with Rayleigh roughness loss
and the Rayleigh reflection coefficient of a sand bottom (1,700 m/s, 1.9 g/cm³). Rays that touched the moving surface
get a random phase process (Gaussian Doppler spectrum). The whole waveform is time-scaled by `a = v/c` and rotated by
the carrier Doppler `fc·a`, which is what a closing speed `v` does to a wideband acoustic signal.

**Noise.** Either Gaussian noise shaped by the Coates fit of the Wenz curves (shipping, wind), or a real hydrophone
recording (Glacier Bay, public domain), both taken through the receiver front end and scaled to the same in-band SNR.

**Modem.** Complex baseband at 24 kHz, carrier 6 kHz, QPSK at 3 kBd with a root-raised-cosine pulse (roll-off 0.25,
3.75 kHz occupied in the 4 to 8 kHz band). Packet: 50 ms LFM chirp, guard, 256 training and 1,024 data symbols, guard,
postamble chirp.

**Receiver.**
1. Normalised chirp matched filter; detection threshold 0.3 (noise-only false alarms measured on both noise types).
2. Doppler: the preamble-to-postamble interval gives `1 + a`. The interval is measured by cross-correlating the two
   correlation envelopes over ±15 ms, so the whole multipath signature is aligned, not a single peak that fading can
   swap between paths.
3. Resampling by `1/(1 + â)` and removal of the residual carrier rotation.
4. Timing on the first significant path (first local maximum above 50 % of the strongest, within 15 ms before it).
5. Fractionally spaced (T/2) equaliser adapted by RLS (λ = 0.995) with a second-order PLL inside the loop
   (Stojanovic, Catipovic and Proakis): single tap, linear (48 taps) or decision feedback (48 + 32 taps).

**Ranging.** Two-way travel time with the 50 ms chirp, known turnaround. Time of arrival from the first-path peak with
parabolic interpolation. The Cramér-Rao bound uses the chirp's RMS bandwidth: `var(τ) ≥ 1 / (8π² β² E/N0)`. The
range-Doppler coupling of an LFM up-chirp biases each leg by `−a (fc T/B + T/2)`; the correction uses the DVL closing
speed.

**Outlier model.** LightGBM on eight matched-filter features (peak, first-to-strongest ratio and delay, number of peaks
above half the maximum, second peak, first-peak width, peak-to-median, SNR), trained on 1,600 simulated pings and tested
on 800 others with different seeds, speeds, source levels and noise types. Label: range error above 0.5 m.

**Navigation.** Four seabed beacons on a 1.5 km square, AUV at 30 m on a lawnmower survey at 1.5 m/s, pings every 10 s.
DVL with 0.3 % scale error and 2 cm/s noise, compass with 1° bias. Estimators: dead reckoning; Gauss-Newton
least squares per epoch; EKF on position only; EKF with heading-bias and scale states (made observable by the turns);
χ² gating of the innovations (99 %) with a lock-out guard; optional ML screening before the gate.

## Limits

* **Simulation first.** The only real data is the hydrophone recording used as noise. Validating the receiver on sea
  trial recordings (with a known transmitted packet) is the next step, and the job of the internship.
* The waveguide is isovelocity with a flat bottom; a real sound-speed profile bends rays (refraction, shadow zones) and
  would need a ray tracer such as Bellhop.
* No channel coding: BER is uncoded; a code would trade rate for a much lower packet error rate.
* The surface Doppler model is a simple Gaussian phase process, not a full time-varying surface scattering model.
* The navigation scenario assumes beacon positions and the sound speed are known; real deployments calibrate both.
