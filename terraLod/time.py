import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
class Time:
    def __init__(self, worldConfig):
        # start at 1st of January 0001 at 00:00
        self.tick = datetime(1, 1, 1)
        self.worldConfig = worldConfig

    def __call__(self, **kwargs):
        self.step(**kwargs)

    def __str__(self):
        return self.tick.strftime("%d/%m/%Y %H:%M")

    def step(self, **kwargs):
        self.tick += relativedelta(**kwargs)

    # ── basic accessors ───────────────────────────────────────────────────────

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
        lat  = np.radians(self.worldConfig.latitude)
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
        lon             = self.worldConfig.longitude          # observer longitude (°E)
        ref_lon         = self.worldConfig.timezone_offset * 15.0  # reference meridian (°)
        longitude_corr  = (lon - ref_lon) / 15.0              # hours
        return self.fractional_hour + longitude_corr + self._equation_of_time

    @property
    def _equation_of_time(self):
        """
        Equation of Time in fractional hours.
        Accounts for orbital eccentricity and axial tilt; corrects solar noon
        by up to ~16 minutes.  Uses the standard double-angle approximation.
        """
        B = 2 * np.pi * (self.day_of_year - 81) / 364.0   # proxy angle
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
        return sz > 0.0