# Python-Oppenheimer-Snyder-collapse-simulation-with-finite-differences-in-RGPS-gauge
Project for the "Estança d'Investigació" of the Master's in Advanced Physics - Theoretical Physics Specialization.

Based on Jose V. Romero, J. M. (1996). A new spherically symmetric general relativistic hydrodynamical code. The Astrophysical Journal, 839-855. (https://ui.adsabs.harvard.edu/abs/10.1086/177198).

This code implements a numerical simulation of the 1D Oppenheimer-Sndyer collapse by using the finite volume approach.

The simulation evolves the relativistic hydrodynamic equations in a spherically symmetric, dynamic spacetime and benchmarks the numerical solution against the exact analytical OS solution.

Features
- Evolves conservative variables ($\mathcal{D}, \mathcal{S}, \tau$) coupled to the spacetime metric.
- Computes the metric components (the lapse function $\alpha$ and the spatial metric component $X$) at every timestep using a cumulative trapezoidal integration of the mass-energy profile.
- Employs Minmod slope limiting for second-order spatial reconstruction at cell interfaces to prevent unphysical oscillations near the stellar boundary.
- Evaluates numerical fluxes across cell boundaries alongside a localized vacuum atmospheric treatment ($\rho_{\text{atm}} = 10^{-10} \rho_0$).
- Time Integration: Uses 2nd-order Runge-Kutta (RK2) time-stepping scheme.

Dependencies:
Ensure you have a standard Python environment with numpy, scipy and matplotlib.

Output:
The code generates a three-panel snapshot comparison (matplotlib) tracking the evolution over time ($t = 0$ to $t = 12$) normalized by the Schwarzschild radius ($R/2M$) of the Density Evolution ($\rho/\rho_i$), the Velocity Profile ($v$), the Lapse Function ($\alpha$)


Functions:
phys_2_cons / cons_2_phys: Conversions between physical primitives and conserved hydro variables.

obtener_analitica_temporal: Exact mathematical solution parsing the internal/external metric boundary conditions.

reconstruct_primitive: Piecewise linear reconstruction using the Minmod limiter.

get_metrica: Integrates the dynamic spacetime profile based on mass enclosures.

rhs: Formulates the flux derivatives and geometric source terms for the solver loop.


