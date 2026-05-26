Is this realistic? 

frofrom datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

"""
Coordinate system:
+X = east
+Y = north

ij version (array / index convention):
+i (axis 0, rows) = north (+Y)
+j (axis 1, cols) = east (+X)
"""


class Time:
    def __init__(self, worldConfig):
        #start at 1st of January 0000 at 00:00
        self.tick = datetime(1, 1, 1)
        self.worldConfig = worldConfig

    def __call__(self, **kwargs):
        self.step(**kwargs)

    def step(self, **kwargs):
        self.tick += relativedelta(**kwargs)

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
    def season_phase(self):
        # Map day of year to a phase between -1 and 1, peaking at summer solstice
        return - np.cos(2 * np.pi * (self.day_of_year_normalized - 0.4692))  
    @property
    def declination(self):
        return np.radians(23.44 * np.sin(2 * np.pi * (self.day_of_year_normalized - 0.2192)))

    @property
    def solar_vectors(self):
        lat = np.radians(self.worldConfig.latitude)

        # solar noon at 12:00
        hour_angle = np.radians(15.0 * (self.hour - 12.0))

        dec = self.declination

        solar_altitude = np.arcsin(np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(hour_angle))

        solar_azimuth = np.arctan2(
            -np.cos(dec) * np.sin(hour_angle),
            np.cos(lat) * np.sin(dec)
            - np.sin(lat) * np.cos(dec) * np.cos(hour_angle)
        )

        sx = np.cos(solar_altitude) * np.sin(solar_azimuth)
        sy = np.cos(solar_altitude) * np.cos(solar_azimuth)
        sz = np.sin(solar_altitude)

        return sx, sy, sz