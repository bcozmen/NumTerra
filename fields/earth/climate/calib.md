# Istanbul Climate Calibration Reference Dataset
This document provides localized baseline target values for a **200km × 200km mesoscale PDE atmospheric model** centered on Istanbul ($41^\circ\text{N}$, $29^\circ\text{E}$). 

Values represent typical **Day** (midday peak/average daytime) and **Night** (nocturnal minimum/average nighttime) conditions for **January, April, June, and September** to ensure your boundary conditions, source terms, and conservation equations calibrate properly across seasons.

---

## 1. Temperature Variables (`Ta`, `Ts`, `Tw`)

Istanbul has a strong maritime climate signature. Notice how water temperature (`Tw`) acts as a massive thermal flywheel lagging behind air temperature (`Ta`), while bare/urban land surface temperature (`Ts`) swings aggressively under high solar loads.

| Variable | Time | January | April | June | September |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`Ta` (Air Temp)**<br>Unit: `°C`<br>*Description: Air temperature* | **Day**<br>**Night** | 9.0 °C<br>4.0 °C | 17.0 °C<br>9.0 °C | 27.0 °C<br>18.0 °C | 26.0 °C<br>17.0 °C |
| **`Ts` (Land Temp)**<br>Unit: `°C`<br>*Description: Surface temperature* | **Day**<br>**Night** | 11.0 °C<br>2.0 °C | 22.0 °C<br>7.0 °C | 35.0 °C<br>16.0 °C | 31.0 °C<br>15.0 °C |
| **`Tw` (Water Temp)**<br>Unit: `°C`<br>*Description: Water/Ocean temp* | **Day/Night** | 8.0 °C *(Stable)* | 11.0 °C *(Stable)* | 22.0 °C *(Stable)* | 23.5 °C *(Stable)* |

### 💡 PDE Calibration Notes:
* **Thermal Inversions:** In winter (`January`), night-time radiative cooling often drops `Ts` below `Ta`, creating a stable nocturnal boundary layer.
* **Urban Heat Island & Topography:** In summer (`June`/`September`), your grid cells representing central Istanbul (highly urbanized) will easily push `Ts` several degrees higher than surrounding rural Thrace or Anatolian forested cells.

---

## 2. Atmospheric & Moisture Capacity (`P`, `Wa`, `Wa_max`)

Atmospheric pressure (`P`) trends higher in winter due to the continental Siberian high-pressure margins pushing dense, cold air over the region. `Wa_max` scales non-linearly with temperature via the Clausius-Clapeyron relation.

| Variable | Time | January | April | June | September |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`P` (Atm. Pressure)**<br>Unit: `Pa`<br>*Description: Atmospheric pressure* | **Day**<br>**Night** | 101,800 Pa<br>101,850 Pa | 101,400 Pa<br>101,450 Pa | 101,100 Pa<br>101,150 Pa | 101,400 Pa<br>101,450 Pa |
| **`Wa` (Water Content)**<br>Unit: `kg/m²`<br>*Description: Atm. water content* | **Day/Night** | 10.0 to 12.0 | 16.0 to 18.0 | 28.0 to 32.0 | 26.0 to 30.0 |
| **`Wa_max` (Max Capacity)**<br>Unit: `kg/m²`<br>*Description: Max water capacity* | **Day**<br>**Night** | 14.0 kg/m²<br>11.0 kg/m² | 24.0 kg/m²<br>15.0 kg/m² | 44.0 kg/m²<br>27.0 kg/m² | 41.0 kg/m²<br>25.0 kg/m² |

### 💡 PDE Calibration Notes:
* **Relative Humidity Calculation:** $RH = \frac{Wa}{Wa\_max}$. Real-world relative humidity in Istanbul remains consistently high year-round ($65\% - 80\%$), meaning your moisture advection terms must maintain high ambient background vapor.
* **Pressure Gradients:** Set up slight nocturnal pressure increases as air cools and contracts over land mass regions in your 200km domain.

---

## 3. Water Cycle Fluxes (`Wc`, `Ws`, `Evap`, `Condensation`, `Precip`)

Istanbul exhibits high precipitation in winter and convective, sporadic episodes during late summer. Evaporation peaks intensely over land during summer days but remains active over sea surfaces almost uniformly.

| Variable | Time | January | April | June | September |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`Wc` (Cloud Water)**<br>Unit: `kg/m²`<br>*Description: Cloud liquid water* | **Day**<br>**Night** | 0.25 kg/m²<br>0.35 kg/m² | 0.12 kg/m²<br>0.18 kg/m² | 0.04 kg/m²<br>0.06 kg/m² | 0.08 kg/m²<br>0.10 kg/m² |
| **`Ws` (Surface Water)**<br>Unit: `mm`<br>*Description: Soil moisture/bodies* | **Day/Night** | 45.0 mm *(Saturated)* | 30.0 mm *(Drying)* | 12.0 mm *(Dry soil)* | 15.0 mm *(Parched)* |
| **`Evap` (Evaporation Rate)**<br>Unit: `mm/hr`<br>*Description: Evaporation flux* | **Day**<br>**Night** | 0.02 mm/hr<br>0.00 mm/hr | 0.15 mm/hr<br>0.01 mm/hr | 0.40 mm/hr<br>0.03 mm/hr | 0.30 mm/hr<br>0.02 mm/hr |
| **`Condensation`**<br>Unit: `mm/hr`<br>*Description: Vapor $\rightarrow$ Cloud flux* | **Day**<br>**Night** | 0.05 mm/hr<br>0.08 mm/hr | 0.03 mm/hr<br>0.05 mm/hr | 0.01 mm/hr<br>0.02 mm/hr | 0.02 mm/hr<br>0.04 mm/hr |
| **`Precip` (Precipitation)**<br>Unit: `mm/hr`<br>*Description: Cloud $\rightarrow$ Surf flux* | **Monthly Avg** | ~0.10 mm/hr *(72mm total)* | ~0.07 mm/hr *(51mm total)* | ~0.04 mm/hr *(31mm total)* | ~0.06 mm/hr *(45mm total)* |

### 💡 PDE Calibration Notes:
* **The Evaporation-Condensation Loop:** Use the night-time drop in air temperature to trigger the condensation term ($Condensation > 0$) as $Wa \rightarrow Wa\_max$.
* **Precipitation Spikes:** Treat `Precip` as an episodic sink term rather than a continuous flat drizzle. Summer rain (June/Sept) should trigger via microphysics thresholds when $Wc$ exceeds a critical density, simulating brief convective thunderstorms (e.g., spikes up to $10.0\text{ mm/hr}$).

---

## 4. Solar Dynamics & Wind (`Sun`, `Shadow`, `V`)

Wind regimes in Istanbul are legendary and deeply structural. The region is swept by the **Etesian (Meltemi)** winds—strong, cool, steady north-northeasterly flows coming over the Black Sea, particularly sharp in summer.

| Variable | Time | January | April | June | September |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`Sun` (Solar Input)**<br>Unit: `W/m²`<br>*Description: Surface solar flux* | **Noon Peak**<br>**Night** | ~250 W/m²<br>0 W/m² | ~600 W/m²<br>0 W/m² | ~850 W/m²<br>0 W/m² | ~550 W/m²<br>0 W/m² |
| **`Shadow` (Sunlit Fraction)**<br>Unit: `float`<br>*Description: 0.0=Shadow, 1.0=Clear* | **Day**<br>**Night** | 0.4 *(High winter cloud)*<br>0.0 | 0.7 *(Intermittent)*<br>0.0 | 0.85 *(Clear skies)*<br>0.0 | 0.80 *(Clear skies)*<br>0.0 |
| **`V` (Wind Vector)**<br>Unit: `m/s`<br>*Description: Vector field* | **Speed**<br>**Direction** | 5.0 to 8.0 m/s<br>Highly variable | 4.0 to 6.0 m/s<br>Predominantly NE | 5.5 to 8.5 m/s<br>Strong, steady NE | 4.5 to 7.0 m/s<br>Steady NE |

### 💡 PDE Calibration Notes:
* **Sea Breeze Convergence Zones (Mesoscale Effect):** On your 200km grid during June/September days, the land area heats up rapidly (`Ts` up to 35°C) creating low localized pressure, while the Black Sea (North) and Marmara Sea (South) remain cold (`Tw` ~22°C). Your momentum PDE equations should naturally generate convergent sea-breeze vectors flowing inward from both coastlines during mid-afternoon!
* **Solar Forcing Integration:** The `Sun` input acts as the primary driver for your system's energy equation ($dT_s/dt$). It should vary cleanly using a sine function based on time-of-day, scaled by latitude ($41^\circ\text{N}$) and the month's specific solar inclination angle.
