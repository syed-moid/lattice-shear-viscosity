# VSR 1982 Fig. 6 — cubic BaTiO₃, $\Omega_0^2(T)$ and Debye relaxation rate $1/\tau(T)$

Paper: H. Vogt, J. A. Sanjurjo, and G. Rossbroich, Phys. Rev. B 26, 5904
(1982) [references/PhysRevB.26.5904.pdf; crop references/VSR_Fig6.jpg].

Figure/panels: Fig. 6. Bottom panel: $\Omega_0^2$ vs T. Top panel:
$1/\tau$ vs T, with the panel's own axis label
"$1/\tau = \Omega_0^2/\gamma$ ($\mathrm{cm}^{-1}$)" — i.e. the paper's
relaxation rate IS $\Omega_0^2/\gamma$ with NO additional convention
factor (this resolves consistency check (c) of the merge script exactly).

Axis units AS PLOTTED (verified against the figure image 2026-07-24):
T in KELVIN (300-800 K); $\Omega_0^2$ in $10^3\ \mathrm{cm}^{-2}$
(self-checked: the Fig. 3 maximum $\Omega_0$ = 92.1 $\mathrm{cm}^{-1}$
squares to $8.48 \times 10^3\ \mathrm{cm}^{-2}$, matching this file's
maximum 8.43); $1/\tau$ in $\mathrm{cm}^{-1}$. No conversions applied to
the raw files.

Files:
- omega0sq_vsr1982_fit.csv (14 pts) = closed circles, DIRECT spectral
  fits — PRIMARY series.
- omega0sq_vsr1982_intensity.csv (10 pts) = crosses — derived from the
  hyper-Raman INTENSITY via the paper's Eq. (4), calibrated to
  $\Omega_0$ = 31 $\mathrm{cm}^{-1}$ at 473 K, and assuming a
  T-independent hyper-Raman polarizability — CORROBORATION ONLY, not
  merged as primary data.
- invtau_vsr1982.csv (12 pts) = open circles, Debye-relaxator rate.

Deliberate omissions: the two straight lines in the bottom panel were
NOT digitized — they are Curie-Weiss + LST theory curves for
$C = 1.5 \times 10^5$ K and $C = 1.8 \times 10^5$ K (values printed on
the figure); they are reconstructible from these constants and are
recorded here instead.

Digitization: WebPlotDigitizer, by Syed Moid, received 2026-07-24.
