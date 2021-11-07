import json
import math
from scipy import interpolate

from . import tools as qttools


class RateCurve:
    def __init__(self, zc_dates, zc_values):
        zc_dates_valid, zc_values_valid = qttools.get_curve_data_with_extrapolation_pillars(zc_dates, zc_values)
        self._zc_dates = zc_dates_valid
        self._zc_values = zc_values_valid
        self.rate_curve = interpolate.interp1d(self._zc_dates, self._zc_values, fill_value="extrapolate")

    def __repr__(self):
        return json.dumps({"type": "RateCurve", "dates": self._zc_dates, "rates": self._zc_values})

    def rate(self, date):
        return self.rate_curve(date)

    def discount(self, date):
        return math.exp(-self.rate(date) * date)

    def forward_rate(self, start_date, end_date):
        return (self.discount(start_date) / self.discount(end_date) - 1) / (end_date - start_date)
