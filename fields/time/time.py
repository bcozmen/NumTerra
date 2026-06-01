from dataclasses import dataclass, field
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

from fields import BaseModel
#x → East
#y → North

@dataclass
class TimeConfig:
    start_year: int = 2000
    start_month: int = 3
    start_day: int = 1
    start_hour: int = 6

    dt : int = 4 #time step in hours
class Time(BaseModel):
    info = {
        'name':'time',
        'map_info' : {}
    }
    def __init__(self, world, config = None):
        super().__init__(world, config)
        self.__dict__.update(TimeConfig().__dict__) # set default config values
        # start at 1st of January 2000 at 00:00
        self.tick = datetime(self.start_year, self.start_month, self.start_day, self.start_hour)
    
    def __str__(self):
        return self.tick.strftime("%d/%m/%Y %H:%M")

    def step(self):
        self.tick += relativedelta(hours=self.dt)



    # ── basic accessors ───────────────────────────────────────────────────────
    @property
    def date(self):
        return self.tick.date()
    @property
    def hour(self):
        return self.tick.hour

    @property
    def day(self):
        return self.tick.day

    @property
    def month(self):
        return self.tick.month

    @property
    def year(self):
        return self.tick.year

    @property
    def day_of_year(self):
        return self.tick.timetuple().tm_yday

    @property
    def is_leap_year(self):
        year = self.tick.year
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    @property
    def day_of_year_normalized(self):
        return (self.day_of_year - 1) / 365.2422
    @property
    def fractional_hour(self):
        """Continuous hour value so the sun moves smoothly within each hour."""
        return (
            self.tick.hour
            + self.tick.minute / 60.0
            + self.tick.second / 3600.0
        )

    @property
    def timezone_offset(self):
        return int((self.world.longitude + 7.5) // 15.0)  # round to nearest timezone

    # ── Derived properties ────────────────────────────────────────────────────────
    @property
    def weekday(self):
        return self.tick.weekday()  # Monday is 0, Sunday is 6

    @property
    def is_first_hour_of_day(self):
        return self.tick.hour == 0

    @property
    def is_first_hour_of_week(self):
        return self.tick.hour == 0 and self.tick.weekday() == 0

    @property
    def is_first_hour_of_month(self):
        return self.tick.hour == 0 and self.tick.day == 1

    def is_first_hour_of_year(self):
        return self.tick.hour == 0 and self.tick.month == 1 and self.tick.day == 1


    # ── seasonal / orbital properties ────────────────────────────────────────

    @property
    def season_phase(self):
        """−1 (winter solstice) → +1 (summer solstice)."""
        return np.cos(2 * np.pi * (self.day_of_year_normalized - 0.4692))

    @property
    def declination(self):
        """Solar declination in radians (standard low-precision formula)."""
        return np.radians(
            23.44 * np.sin(2 * np.pi * (self.day_of_year_normalized - 0.2192))
        )


    # ── solar geometry ────────────────────────────────────────────────────────
    @property
    def solar_vectors(self):
        """Return the solar direction unit-vector `(sx, sy, sz)`.

        Interpretation:
        - `sx` : East component (positive toward +x / east)
        - `sy` : North component (positive toward +y / north)
        - `sz` : Up component (positive upward)

        The returned vector is unit-length (or very close to it); `sz` is
        equal to `sin(solar_altitude)` so `sz>0` when the sun is above the
        geometric horizon. Azimuth is measured from North toward East
        (0° = North, 90° = East). Use the dot-product `dot(surface_normal, s)`
        (clipped to [0,1]) as a slope incidence factor when computing
        surface insolation for sloped terrain.
        """

        lat  = np.radians(self.world.latitude)
        dec  = self.declination

        # FIX 1+2+3: use continuous, longitude- and EoT-corrected solar time
        hour_angle = np.radians(15.0 * (self._apparent_solar_time - 12.0))

        solar_altitude = np.arcsin(
            np.sin(lat) * np.sin(dec)
            + np.cos(lat) * np.cos(dec) * np.cos(hour_angle)
        )
        solar_azimuth = np.arctan2(
            -np.cos(dec) * np.sin(hour_angle),
             np.cos(lat) * np.sin(dec) - np.sin(lat) * np.cos(dec) * np.cos(hour_angle)
        )

        sx = np.cos(solar_altitude) * np.sin(solar_azimuth)   # East
        sy = np.cos(solar_altitude) * np.cos(solar_azimuth)   # North
        sz = np.sin(solar_altitude)                            # Up

        return sx, sy, sz

    @property
    def _apparent_solar_time(self):
        lon             = self.world.longitude          # observer longitude (°E)
        ref_lon         = self.timezone_offset * 15.0  # reference meridian (°)
        longitude_corr  = (lon - ref_lon) / 15.0              # hours
        ast = self.fractional_hour + longitude_corr + self._equation_of_time
        return ast % 24.0  # wrap around 24 hours

    @property
    def _equation_of_time(self):
        """
        Equation of Time in fractional hours.
        Accounts for orbital eccentricity and axial tilt; corrects solar noon
        by up to ~16 minutes.  Uses the standard double-angle approximation.
        """
        B = 2 * np.pi * (self.day_of_year - 81) / 365.0   # proxy angle
        eot_minutes = (
            9.87 * np.sin(2 * B)
            - 7.53 * np.cos(B)
            - 1.5  * np.sin(B)
        )
        return eot_minutes / 60.0   # convert to hours

    # ── FIX 4: convenience night guard ───────────────────────────────────────

    @property
    def is_daytime(self):
        """True when the sun's centre is above the geometric horizon (sz > 0)."""
        _, _, sz = self.solar_vectors
        return sz > np.sin(np.radians(-0.833))  # account for refraction and sun radius


