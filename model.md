User: Please explain me the climate model. What run's in which order and what function is used to calculate (write steps mathematically)

GitHub Copilot: Read [](file:///home/baris/NumTerra/fields/earth/climate/climate.py)

Read [](file:///home/baris/NumTerra/fields/earth/climate/climate.md)

Read [](file:///home/baris/NumTerra/fields/earth/climate/calib.md)

Read [](file:///home/baris/NumTerra/fields/earth/climate/sun/sun.py)

Read [](file:///home/baris/NumTerra/fields/earth/climate/thermal/thermal.py)

Read [](file:///home/baris/NumTerra/fields/earth/climate/pressure/pressure.py)

Read [](file:///home/baris/NumTerra/fields/earth/climate/hydro/hydro.py)

Read [](file:///home/baris/NumTerra/fields/earth/climate/wind/wind.py)

Read [](file:///home/baris/NumTerra/fields/earth/climate/hydro/water.py)

Read [](file:///home/baris/NumTerra/fields/earth/climate/hydro/numba.py)

Read [](file:///home/baris/NumTerra/fields/earth/climate/thermal/numba.py)

Read [](file:///home/baris/NumTerra/fields/earth/climate/wind/numba.py)

Read [](file:///home/baris/NumTerra/fields/earth/climate/sun/numba.py)

Searching for regex `climate.step|Climate(` (`**/*.py`)

Searched for regex `def step|timeRegister|advance` (`**/*.py`), 14 results

Read [](file:///home/baris/NumTerra/fields/earth/climate/wind/oscilator.py)

Read memory

Read [](file:///home/baris/NumTerra/fields/engine/world.py)

Read [](file:///home/baris/NumTerra/fields/time/time.py#1-1), lines 1 to 180

Read [](file:///home/baris/NumTerra/fields/time/timeRegister.py)

Read [](file:///home/baris/NumTerra/fields/earth/__init__.py)

Read [](file:///home/baris/NumTerra/fields/earth/climate/__init__.py)

Read [](file:///home/baris/NumTerra/fields/baseModel.py)

Read [](file:///home/baris/NumTerra/fields/earth/terrain/terrain.py)

## Overview

The climate model is a 2-D gridded model. Each map is an array over the terrain grid. The main prognostic state is:

- $T_a$: air temperature
- $T_s$: land-surface temperature
- $T_w$: water temperature
- $W_a$: atmospheric water vapor
- $W_c$: cloud water
- $W_s$: surface water
- $\mathbf V=(u,v)$: wind velocity

The main climate loop is implemented in climate.py.

---

# 1. Global execution order

The world initializes its models in this order:

1. `Time`
2. `Terrain`
3. `Climate`
4. Renderer and observer

This is defined in world.py and __init__.py.

For every world step:

```text
Time.step()
Terrain.step()
Climate.step()
Renderer.step(), if enabled
```

The time step is currently:

$$
\Delta t = 2\ \text{hours}
$$

The time model advances before the climate model, so climate calculations use the new time and solar position. This is defined in time.py.

Within one climate step, the order is:

```text
1. Solar radiation
2. Atmospheric water capacity and water-cycle fluxes
3. Thermal energy update
4. Water-mass conservation
5. Atmospheric pressure
6. Wind and advection, repeated twice
7. Sea-surface water correction
8. Save maps
```

---

# 2. Initialization

Climate initialization happens in climate.py.

## 2.1 Terrain

The terrain model first creates:

- normalized terrain height $H$
- sea mask $M_{\text{sea}}$
- sea level

The terrain height is converted to meters using:

$$
H_m = H \cdot H_{\max}
$$

where:

$$
H_{\max}=1000\ \text{m}
$$

The terrain model is implemented in terrain.py.

## 2.2 Initial temperature

The initial temperatures are calculated by `Thermal.init()` in thermal.py.

Latitude controls the mean temperature:

$$
L=\cos(\phi)
$$

$$
T_{\text{mean}} = 30L - 15(1-L)
$$

Seasonal variation:

$$
T_{\text{season}} =
15(1-L)\cdot \text{season\_phase}
$$

Diurnal variation:

$$
h=\frac{h_{\text{hour}}-14}{24}\cdot 2\pi
$$

$$
T_{\text{diurnal}}=5\cos(h)
$$

The base temperature is:

$$
T_{\text{base}}
=
T_{\text{mean}}
+
T_{\text{season}}
+
T_{\text{diurnal}}
$$

Air temperature decreases with terrain height:

$$
T_a = T_{\text{base}}-\Gamma H_m
$$

where:

$$
\Gamma=0.0065\ \text{K/m}
$$

Initially:

$$
T_s=T_a
$$

For water cells:

$$
T_w=T_{\text{base}}
$$

For land cells:

$$
T_w=T_a
$$

## 2.3 Initial atmospheric water

After calculating the initial moisture capacity $W_{a,\max}$:

$$
W_a=0.7W_{a,\max}
$$

Cloud water starts at zero:

$$
W_c=0
$$

Land surface water starts at:

$$
W_s=50\ \text{mm}
$$

For sea cells, surface water is set from the terrain depth:

$$
W_s =
\max(0,\text{sea\_level}-H)
\cdot H_{\max}\cdot 1000
$$

The initialization logic is in [fields/earth/climate/climate.py](fields/earth/climate/climate.py#L229-L250).

---

# 3. Step 1: Solar radiation

The solar model is called first:

```python
Sun, Shadow, Sun_atm = self.sun(...)
```

See [fields/earth/climate/climate.py](fields/earth/climate/climate.py#L165-L167).

## 3.1 Solar direction

The time model computes the solar vector:

$$
\mathbf s=(s_x,s_y,s_z)
$$

where $s_z$ is the upward component. If:

$$
s_z\leq 0
$$

then it is nighttime and:

$$
\text{Sun}=0
$$

The solar vector calculation is in time.py.

## 3.2 Terrain normal

For land, the terrain normal is approximately:

$$
\mathbf n=
\frac{(-H_y,-H_x,1)}
{\sqrt{H_x^2+H_y^2+1}}
$$

For sea cells, the surface is treated as flat:

$$
\mathbf n=(0,0,1)
$$

## 3.3 Solar incidence and shadows

The direct incidence is:

$$
I_{\text{direct}}
=
\max(0,\mathbf n\cdot\mathbf s)
\cdot \text{Shadow}
\cdot (1-f_d)
$$

The diffuse component is:

$$
I_{\text{diffuse}}=f_d s_z
$$

where:

$$
f_d=0.25
$$

Therefore:

$$
I_{\text{effective}}
=
I_{\text{direct}}+I_{\text{diffuse}}
$$

The solar energy before atmospheric absorption is:

$$
S_{\text{toa}}
=
S_0I_{\text{effective}}
$$

where:

$$
S_0=1361\ \text{W/m}^2
$$

## 3.4 Atmospheric transmission

Cloud transmission:

$$
\tau_c=e^{-k_cW_c}
$$

Water-vapor transmission:

$$
\tau_v=e^{-k_vW_a}
$$

Total transmission:

$$
\tau
=
\tau_0\tau_v\tau_c
$$

where:

$$
\tau_0=0.85
$$

Surface solar radiation:

$$
S=S_{\text{toa}}\tau
$$

Energy absorbed by the atmosphere:

$$
S_{\text{atm}}
=
S_{\text{toa}}-S
$$

The implementation is in sun.py.

---

# 4. Step 2: Atmospheric moisture capacity and water-cycle fluxes

The hydrological model is called next:

```python
Wa_max, Evap, Condensation, Precip = self.hydro(...)
```

See climate.py.

The main implementation is the Numba function in numba.py.

## 4.1 Saturation vapor pressure

For each atmospheric layer, saturation vapor pressure is calculated using the Magnus formula:

$$
e_s(T)
=
611.2
\exp\left(
\frac{17.67T}{T+243.5}
\right)
$$

where $T$ is in Celsius and $e_s$ is in pascals.

It is limited to:

$$
e_s\leq0.99P
$$

## 4.2 Saturation specific humidity

The saturation specific humidity is:

$$
q_s
=
\frac{\epsilon e_s}
{P-(1-\epsilon)e_s}
$$

where:

$$
\epsilon=0.622
$$

## 4.3 Maximum atmospheric water

The model integrates the moisture capacity over three atmospheric layers:

$$
W_{a,\max}
=
\sum_{k=0}^{N-1}q_{s,k}\frac{\Delta P}{g}
$$

with:

$$
N=3
$$

$$
\Delta P=15000\ \text{Pa}
$$

For each layer:

$$
P_{k+1}=P_k-\Delta P
$$

$$
T_{k+1}=T_k-\lambda_P\Delta P
$$

where:

$$
\lambda_P=0.0008\ \text{K/Pa}
$$

## 4.4 Evaporation

The moisture deficit is:

$$
D=\max(0,W_{a,\max}^{\text{surface}}-W_a)
$$

Wind-dependent evaporation potential over water is:

$$
E_{\text{water}}
=
C_{e,w}(V+0.1)D\cdot3600
$$

Land evaporation is:

$$
E_{\text{land}}
=
C_{e,l}(V+0.1)D\cdot3600
$$

Land evaporation cannot exceed the available surface water:

$$
E_{\text{land}}
=
\min\left(E_{\text{land}},\frac{W_s}{\Delta t}\right)
$$

The final evaporation is:

$$
E=
M_{\text{sea}}E_{\text{water}}
+
(1-M_{\text{sea}})E_{\text{land}}
$$

Land cells with:

$$
W_s>20\ \text{mm}
$$

are treated as inland lakes and use the water evaporation coefficient.

## 4.5 Condensation

Relative humidity is approximated by:

$$
RH=\frac{W_a}{W_{a,\max}}
$$

Condensation starts above:

$$
RH_c=0.85
$$

The relaxation factor is:

$$
\alpha=1-e^{-\Delta t/\tau_c}
$$

where:

$$
\tau_c=3\ \text{hours}
$$

The condensation rate is:

$$
C
=
\frac{\alpha}{\Delta t}
\max(0,W_a-RH_cW_{a,\max})
$$

## 4.6 Precipitation

Cloud water converts into precipitation according to:

$$
r=1-f_{\text{delay}}
$$

$$
f_{\text{removed}}
=
1-e^{-r\Delta t}
$$

$$
R
=
\frac{W_c f_{\text{removed}}}{\Delta t}
$$

With the default configuration:

$$
f_{\text{delay}}=0.5
$$

---

# 5. Step 3: Thermal update

The thermal model is called using the fluxes calculated in the previous step:

```python
dTa, dTs, dTw = self.thermal(...)
Ta, Ts, Tw = Ta + dTa, Ts + dTs, Tw + dTw
```

See climate.py.

The complete numerical implementation is in numba.py.

The model calculates energy changes for:

- air
- land surface
- water surface

## 5.1 Sensible heat exchange

The heat-transfer coefficient is:

$$
h=ChV
$$

where:

$$
C=5
$$

The sensible heat flux from surface to air is:

$$
F_{\text{sensible}}
=
h(T_{\text{surface}}-T_a)
$$

The resulting temperature changes are:

$$
\Delta T_a^{\text{sensible}}
=
\frac{F_{\text{sensible}}}{C_a}
$$

$$
\Delta T_{\text{surface}}^{\text{sensible}}
=
-\frac{F_{\text{sensible}}}{C_{\text{surface}}}
$$

The heat capacities are:

$$
C_a=1.03\times10^6\ \text{J/m}^2\text{/K}
$$

$$
C_{\text{land}}=2.0\times10^6\ \text{J/m}^2\text{/K}
$$

$$
C_{\text{water}}=5.184\times10^7\ \text{J/m}^2\text{/K}
$$

## 5.2 Solar heating and evaporative cooling

Evaporative energy loss is:

$$
F_{\text{evap}}
=
\frac{E}{3600}L_v
$$

where:

$$
L_v=2.5\times10^6\ \text{J/kg}
$$

For land:

$$
\Delta T_s^{\text{solar}}
=
\frac{S(1-\alpha_l)-F_{\text{evap}}}
{C_{\text{land}}}
$$

For water:

$$
\Delta T_w^{\text{solar}}
=
\frac{S(1-\alpha_w)-F_{\text{evap}}}
{C_{\text{water}}}
$$

with:

$$
\alpha_l=0.25
$$

$$
\alpha_w=0.06
$$

Atmospheric shortwave heating is:

$$
\Delta T_a^{\text{solar}}
=
\frac{S_{\text{atm}}}{C_a}
$$

## 5.3 Latent heat from condensation

Condensation releases heat:

$$
\Delta T_a^{\text{latent}}
=
\frac{(C/3600)L_v}{C_a}
$$

## 5.4 Longwave radiation and greenhouse effect

Surface outgoing longwave radiation is:

$$
F_{\text{out}}
=
\sigma T_s^4
$$

where temperature is converted to Kelvin.

Atmospheric emissivity is calculated from the product of several transmissions:

$$
\tau_{\text{base}}=1-\epsilon_{\text{base}}
$$

$$
\tau_{\text{wv}}
=
1-m_{\text{wv}}
\left(1-e^{-k_{\text{wv}}W_a}\right)
$$

$$
\tau_{\text{cloud}}
=
1-m_c
\left(1-e^{-k_cW_c}\right)
$$

Then:

$$
\tau_{\text{total}}
=
\tau_{\text{base}}
\tau_{\text{wv}}
\tau_{\text{cloud}}
$$

$$
\epsilon_a=1-\tau_{\text{total}}
$$

Atmospheric emission is:

$$
F_{\text{atm}}
=
\epsilon_a\sigma T_a^4
$$

Surface longwave balance:

$$
F_{\text{surface,LW}}
=
F_{\text{atm}}-F_{\text{out}}
$$

Atmospheric longwave balance:

$$
F_{\text{air,LW}}
=
\epsilon_aF_{\text{out}}-2F_{\text{atm}}
$$

## 5.5 Final thermal equations

For land cells:

$$
\frac{dT_s}{dt}
=
\frac{
S(1-\alpha_l)-F_{\text{evap}}
}{C_{\text{land}}}
+
\frac{F_{\text{surface,LW}}}{C_{\text{land}}}
-
\frac{F_{\text{sensible}}}{C_{\text{land}}}
$$

For water cells:

$$
\frac{dT_w}{dt}
=
\frac{
S(1-\alpha_w)-F_{\text{evap}}
}{C_{\text{water}}}
+
\frac{F_{\text{surface,LW}}}{C_{\text{water}}}
-
\frac{F_{\text{sensible}}}{C_{\text{water}}}
$$

Air temperature:

$$
\frac{dT_a}{dt}
=
\frac{F_{\text{sensible}}}{C_a}
+
\frac{F_{\text{air,LW}}}{C_a}
+
\frac{(C/3600)L_v}{C_a}
+
\frac{S_{\text{atm}}}{C_a}
$$

The actual update is:

$$
T^{n+1}=T^n+\Delta t\frac{dT}{dt}
$$

---

# 6. Step 4: Water-mass conservation

After the thermal update, the model updates water stores:

```python
Ta, Wa, Wc, Ws, Condensation = self.hydro.apply_mass_balance(...)
```

See climate.py.

The water equations are:

$$
W_a^{n+1}
=
W_a^n+(E-C)\Delta t
$$

$$
W_c^{n+1}
=
W_c^n+(C-R)\Delta t
$$

$$
W_s^{n+1}
=
W_s^n+(R-E)\Delta t
$$

where:

- $E$: evaporation
- $C$: condensation
- $R$: precipitation

Then the model applies:

$$
W_a,W_c,W_s\geq0
$$

If atmospheric water exceeds capacity:

$$
X=\max(0,W_a-W_{a,\max})
$$

then:

$$
W_a\leftarrow W_a-X
$$

$$
W_c\leftarrow W_c+X
$$

and latent heat is released:

$$
T_a\leftarrow T_a+\frac{XL_v}{C_a}
$$

This is implemented in numba.py.

---

# 7. Step 5: Atmospheric pressure

Pressure is recalculated using the updated $T_a$ and $W_a$:

```python
P = self.pressure(H, Ta, Wa)
```

See climate.py.

## 7.1 Temperature-driven sea-level pressure

The mean air temperature is:

$$
\overline T=\text{mean}(T_a+273.15)
$$

The temperature anomaly is used to perturb sea-level pressure:

$$
P_{\text{sl}}
=
P_0
\left[
1-0.03
\frac{T_k-\overline T}{\overline T}
\right]
$$

Warm regions therefore receive lower pressure and cold regions higher pressure.

## 7.2 Virtual temperature

Atmospheric water content is converted to a mixing ratio approximation:

$$
q=
\frac{W_a}{P_0/g}
$$

and clipped to:

$$
0\leq q\leq0.04
$$

Virtual temperature is:

$$
T_v=T_k(1+0.61q)
$$

## 7.3 Hydrostatic pressure decrease

At terrain height $H_m$:

$$
P
=
P_{\text{sl}}
\exp\left(
-\frac{gH_m}{RT_v}
\right)
$$

The pressure implementation is in [fields/earth/climate/pressure/pressure.py](fields/earth/climate/pressure/pressure.py#L5-L42).

---

# 8. Step 6: Wind and atmospheric advection

The climate model calls `_advect()`:

```python
H, Ta, Wa, Wc, V = self._advect(...)
```

See [fields/earth/climate/climate.py](fields/earth/climate/climate.py#L184-L186).

The default configuration uses:

```python
adv_sub_steps = 2
advection_scheme = "semi_lagrangian"
```

Therefore each climate step is divided into two wind/advection substeps:

$$
\Delta t_{\text{sub}}=\frac{\Delta t}{2}
$$

The wind model is called twice from [fields/earth/climate/climate.py](fields/earth/climate/climate.py#L216-L223).

## 8.1 Pressure-gradient acceleration

Pressure is scaled:

$$
P_s=0.02P
$$

The pressure-gradient force is:

$$
\frac{du}{dt}
=
-\frac{1}{\rho_0}\frac{\partial P_s}{\partial x}
$$

$$
\frac{dv}{dt}
=
-\frac{1}{\rho_0}\frac{\partial P_s}{\partial y}
$$

where:

$$
\rho_0=1.225\ \text{kg/m}^3
$$

## 8.2 Coriolis rotation

The Coriolis parameter is:

$$
f=2\Omega\sin(\phi)
$$

The velocity is rotated by:

$$
\theta=f\Delta t
$$

approximately:

$$
u' = u\cos\theta-v\sin\theta
$$

$$
v' = u\sin\theta+v\cos\theta
$$

The implementation is in [fields/earth/climate/wind/numba.py](fields/earth/climate/wind/numba.py#L5-L40).

## 8.3 Wind friction

Wind friction is applied as:

$$
u\leftarrow\frac{u}{1+\mu\Delta t}
$$

$$
v\leftarrow\frac{v}{1+\mu\Delta t}
$$

where:

$$
\mu=0.001
$$

## 8.4 Pressure projection

The model calculates velocity divergence:

$$
\nabla\cdot\mathbf V
=
\frac{\partial u}{\partial x}
+
\frac{\partial v}{\partial y}
$$

It then solves approximately:

$$
\nabla^2p=\nabla\cdot\mathbf V
$$

using 10 Jacobi iterations.

The velocity is projected toward a divergence-free field:

$$
u\leftarrow u-\frac{\partial p}{\partial x}
$$

$$
v\leftarrow v-\frac{\partial p}{\partial y}
$$

This is also implemented in [fields/earth/climate/wind/numba.py](fields/earth/climate/wind/numba.py#L42-L91).

## 8.5 Prevailing-wind nudging

A stochastic prevailing wind is generated with an Ornstein–Uhlenbeck process:

$$
V_{t+\Delta t}
=
V_t
-
\frac{1}{\tau_v}(V_t-V_{\text{target}})\Delta t
+
\sigma_v\sqrt{\Delta t}\,\xi
$$

The direction follows a similar process:

$$
\theta_{t+\Delta t}
=
\theta_t
+
\frac{\theta_{\text{target}}-\theta_t}{\tau_\theta}\Delta t
+
\sigma_\theta\sqrt{\Delta t}\,\eta
$$

The current velocity is nudged toward that macro wind:

$$
\mathbf V
\leftarrow
\mathbf V
+
\alpha(\mathbf V_{\text{macro}}-\mathbf V)
$$

where:

$$
\alpha=\min\left(\frac{\Delta t}{6},1\right)
$$

The oscillator is implemented in [fields/earth/climate/wind/oscilator.py](fields/earth/climate/wind/oscilator.py#L4-L54).

## 8.6 Semi-Lagrangian advection

The model advects:

- $T_a$
- $W_a$
- $W_c$

For each grid cell, it traces backward along the wind:

$$
i_s=i-\frac{v\Delta t}{\Delta y}
$$

$$
j_s=j-\frac{u\Delta t}{\Delta x}
$$

The value at the departure point is calculated with bilinear interpolation:

$$
F^{n+1}_{i,j}
=
w_{00}F_{i_0,j_0}
+w_{10}F_{i_1,j_0}
+w_{01}F_{i_0,j_1}
+w_{11}F_{i_1,j_1}
$$

This is implemented in numba.py.

## 8.7 Orographic cooling

The terrain slopes are:

$$
\frac{\partial H}{\partial x},
\qquad
\frac{\partial H}{\partial y}
$$

The vertical-motion proxy is:

$$
w=
-\left(
u\frac{\partial H}{\partial x}
+
v\frac{\partial H}{\partial y}
\right)
$$

Only rising air is cooled:

$$
\frac{dT_a}{dt}
=
\begin{cases}
-\Gamma w,&w>0\\
0,&w\leq0
\end{cases}
$$

The final applied tendency is additionally scaled by the pressure-gradient factor:

$$
T_a\leftarrow
T_a+
\left(-\Gamma w\right)
\Delta t
\cdot0.02
$$

The implementation is in numba.py.

---

# 9. Step 7: Sea-water correction

After advection, sea-cell surface water is reset from the terrain depth:

$$
W_s=
\max(0,\text{sea\_level}-H)
\cdot H_{\max}\cdot1000
$$

This prevents atmospheric advection from changing the ocean depth. It is performed in climate.py.

---

# 10. Final state update

The following maps are saved back to the world:

$$
\{H,\text{Sun},\text{Shadow},T_a,T_s,T_w,P,
W_a,W_{a,\max},W_c,W_s,E,C,R\}
$$

This is performed in climate.py.

---

## Complete one-step algorithm

Mathematically, one climate timestep is:

$$
(\text{Sun},\text{Shadow},S_{\text{atm}})
=
\mathcal S(H,\nabla H,M_{\text{sea}},W_a,W_c,t)
$$

$$
(W_{a,\max},E,C,R)
=
\mathcal H(P,T_a,T_s,T_w,W_a,W_c,M_{\text{sea}},W_s,V)
$$

$$
(\Delta T_a,\Delta T_s,\Delta T_w)
=
\mathcal T(
M_{\text{sea}},
\text{Sun},
S_{\text{atm}},
T_a,T_s,T_w,
|V|,E,C,R,W_a,W_c
)
$$

$$
(T_a,T_s,T_w)
\leftarrow
(T_a+\Delta T_a,T_s+\Delta T_s,T_w+\Delta T_w)
$$

$$
(W_a,W_c,W_s)
\leftarrow
\mathcal B(W_a,W_c,W_s,E,C,R,W_{a,\max})
$$

$$
P\leftarrow\mathcal P(H,T_a,W_a)
$$

For each of two advection substeps:

$$
\mathbf V\leftarrow
\mathcal W(P,\mathbf V)
$$

$$
(T_a,W_a,W_c)
\leftarrow
\mathcal A(T_a,W_a,W_c,\mathbf V)
$$

$$
T_a\leftarrow T_a+\text{orographic cooling}
$$

Finally:

$$
W_s\leftarrow
W_s^{\text{sea-depth}}
\quad\text{on sea cells}
$$

A notable implementation detail is that the active `Water` runoff-routing model and `Erosion` model are currently disabled in `_advect()`; both calls are commented out in climate.py. Therefore, surface water changes only through precipitation and evaporation during the climate step, except for the sea-depth reset.