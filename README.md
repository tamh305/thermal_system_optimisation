# Computational Methods and Modelling, Coursework Project- Heap Pump Simulation
Nonlinear Dynamic Simulation & Optimisation of Coupled Thermal Storage

This repository models the transient dynamics of coupled heat-pump and energy storage etworks under dynamic environmental boundary conditions, formulated as non-linear ordinary differential equations (ODEs) and solved via implicit backward differentiation. The optimisation uses a parallel Differential Evolution engine design to minimise daily energy consumption across a constrained 6-dimensional parameter space

---

## Mathematical Formulation

### 1. Transient Energy Balance ODE
The temperature evolution of the thermal energy storage reservoir over continuous time $t$ is modelled via the first-order dynamic energy balance:

$$\frac{dT_{\text{tank}}}{dt} = \frac{\dot{Q}_{\text{transfer}}(t) - \dot{Q}_{\text{load}}(t) - \dot{Q}_{\text{loss}}(t)}{C_t}$$

Where:
* $C_t = m \cdot c_p$ represents the total lumped thermal capacitance of the reservoir ($J \cdot K^{-1}$).
* $\dot{Q}_{\text{transfer}}(t) = U_c A_c (T_{\text{cond}} - T_{\text{tank}}(t)) \cdot \sigma(t)$ is the heat transferred across the heat exchanger, governed by the binary compresser operational state $\sigma(t) \in \{0, 1\}$.
* $\dot{Q}_{\text{load}}(t) = -\left(A_w U_w + A_r U_r\right)(T_{\text{ambient}}(t) - T_{\text{setpoint}})$ represents building envelope conduction losses driven by external ambient temperatures fetched via meteorological time-series APIs.
* $\dot{Q}_{\text{loss}}(t) = U_{\text{tank}} A_{\text{tank}} (T_{\text{tank}}(t) - T_{\text{ambient}}(t))$ accounts for ambient loss across the vessel boundary.

### 2. Numerical Stiff-System Integration (BDF vs RK45)
The compressor state $\sigma(t)$ operates via discrete hysteresis threshold switching:

$$\sigma(t^+) = \begin{cases}  1 & \text{if } T_{\text{tank}}(t) \le T_{\text{threshold, on}} \\ 0 & \text{if } T_{\text{tank}}(t) \ge T_{\text{threshold, off}} \\ \sigma(t^-) & \text{otherwise} \end{cases}$$

This discontinuous on/off hysteresis switching induces mathematical stiffness along boundary interfaces, creating severe eigenvalues in the system Jacobian. Standard explicit Runge-Kutta integration (`RK45`) encounters numerical instability or forces infinitesimal step sizes. 

To ensure numerical stability and computational efficiency, the system implements an implicit **Backward Differentiation Formula (BDF)** multi-step solver via `scipy.integrate.solve_ivp`, allowing robust integration over $24$-hour horizons.

### 3. Continuous Quadratures & Empirical Regressions
* **Empirical COP Model:** Compressor performance is calibrated against empirical manufacturer data by solving a non-linear least-squares optimisation:

$$\text{COP}(T_{\text{ambient}}) = a + \frac{b}{\vert{}T_{\text{cond}} - (T_{\text{ambient}} + 273.15)\vert{}}$$

  Parameters $a$ and $b$ are estimated using curve fitting (`scipy.optimize.curve_fit`).
* **Energy Quadratures:** Continuous power consumption $P(t) = \frac{\dot{Q}_{\text{transfer}}(t)}{\text{COP}(t)}$ is integrated over time $\mathcal{T}$ using trapezoidal quadratures:

$$E_{\text{total}} = \int_{0}^{\mathcal{T}} P(t) \, dt \approx \sum_{k=1}^{N-1} \frac{P(t_k) + P(t_{k+1})}{2} \Delta t_k$$

---

## Optimisation

To minimise total energy consumption while maintaining indoor thermal comfort, the design formulates a global optimisation problem across a continuous 6-dimensional parameter vector $\mathbf{x} \in \mathbb{R}^6$:

$$\mathbf{x} = \begin{bmatrix} U_w & U_r & T_{\text{setpoint}} & T_{\text{initial}} & U_{\text{tank}} & m \end{bmatrix}^T$$

$$\min_{\mathbf{x}} \quad \mathcal{J}(\mathbf{x}) = E_{\text{total}}(\mathbf{x})$$

$$\text{subject to} \quad \mathbf{x}_{\text{lower}} \le \mathbf{x} \le \mathbf{x}_{\text{upper}}$$

### Differential Evolution Strategy
Because the forward evaluation $\mathcal{J}(\mathbf{x})$ requires solving an implicit ODE initial-value problem, the loss landscape contains discontinuities and non-convex regions. We implement a population-based **Differential Evolution (DE)** stochastic search (`scipy.optimize.differential_evolution`) with deferred vector mutation and crossover updates:
* **Population size:** 15 candidates per dimension ($N_p = 90$)
* **Parallel Processing:** Multi-core parallel evaluation (`workers=-1`) distributes individual ODE simulation evaluations across all available CPU threads.
* **Robust Error Handling:** Divergent simulations or physical violations return a penalty barrier ($\mathcal{J} = 10^6$) to steer the population away from unphysical regimes.

---

## Results & Benchmarks

Optimisation evaluations conducted across structural archetypes demonstrated substantial reductions in energy consumption:

| Structural Archetype | Baseline Demand (kWh) | Optimized Demand (kWh) | Energy Reduction (%) | Primary Driving Parameters |
| :--- | :--- | :--- | :--- | :--- |
| **Well-Insulated Building** | 11.96 | 5.29 | **-55.8%** | $U_w$ (-48.7%), $T_{\text{sp}}$ (-24.9%) |
| **Poorly Insulated Building** | 35.96 | 17.12 | **-52.4%** | $U_w$ (-87.7%), $U_r$ (-78.4%) |
| **Large Modern Building** | 13.20 | 9.78 | **-25.9%** | $U_w$ (-30.7%), $U_{\text{tank}}$ (-30.0%) |

---

## Project Structure

```text
Thermal_system_optimisation/
├── data/
│   └── heat_pump_cop_synthetic_full.yaml  # Empirical COP calibration data
├── src/
│   ├── __init__.py                        # Package namespace
│   ├── model.py                           # Dynamic system ODEs & BDF solver
│   ├── optimizer.py                       # Differential Evolution engine
│   └── auxiliary.py                       # Multi-source auxiliary hybrid extension
├── .gitignore
├── README.md
└── requirements.txt