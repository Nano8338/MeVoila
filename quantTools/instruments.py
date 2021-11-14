import enum
import json
import numpy as np
from abc import ABCMeta, abstractmethod


class RateType(enum.Enum):
    FORWARD_LOOKING = "forward-looking rate"
    BACKWARD_LOOKING = "backward-looking rate"
    SWAP_RATE = "swap rate"


def build_rate(start_date, underlying_length, rate_type):
    if rate_type == RateType.FORWARD_LOOKING:
        return ForwardRate(start_date, start_date + underlying_length)
    elif rate_type is RateType.BACKWARD_LOOKING:
        return BackwardRate(start_date, start_date + underlying_length)
    else:  # rate_type == RateType.SWAP_RATE:
        return SwapRate(start_date, underlying_length)


class Underlying(metaclass=ABCMeta):
    @abstractmethod
    def compute_forward(self, model):
        pass

    @abstractmethod
    def compute_numeraire(self, model):
        pass


class RateIndex(Underlying):
    def __init__(self, start_date, end_date, is_forward_looking=True):
        self.start_date = start_date
        self.end_date = end_date
        self.tenor = end_date - start_date
        self.is_forward_looking = is_forward_looking

    @property
    def pay_date(self):
        return self.end_date

    @property
    def expiry(self):
        return self.start_date if self.is_forward_looking is True else self.end_date

    def __repr__(self):
        return json.dumps(
            {
                "type": "ForwardRate" if self.is_forward_looking else "BackwardRate",
                "start_date": self.start_date,
                "end_date": self.end_date,
                "tenor": self.tenor,
            }
        )

    def compute_forward(self, model):
        return model.rate_curve.forward_rate(self.start_date, self.end_date)

    def compute_numeraire(self, model):
        return (self.end_date - self.start_date) * model.rate_curve.discount(self.pay_date)


class BackwardRate(RateIndex):
    def __init__(self, start_date, end_date):
        super(BackwardRate, self).__init__(start_date, end_date, is_forward_looking=False)


class ForwardRate(RateIndex):
    def __init__(self, start_date, end_date):
        super(ForwardRate, self).__init__(start_date, end_date, is_forward_looking=True)


class SwapRate(Underlying):
    def __init__(self, start_date, underlying_length):
        self.start_date = start_date
        self.underlying_length = underlying_length
        self._fixed_leg_pay_dates = np.flipud(np.arange(start_date + underlying_length, start_date, step=-1.0))
        self._basis = np.diff(np.concatenate([[self.start_date], self._fixed_leg_pay_dates]))

    @property
    def end_date(self):
        return self.start_date + self.underlying_length

    @property
    def expiry(self):
        return self.start_date

    def __repr__(self):
        return json.dumps(
            {
                "type": "SwapRate",
                "start_date": self.start_date,
                "underlying_length": self.underlying_length,
            }
        )

    def compute_annuity(self, model):
        return np.sum(
            [
                basis * model.rate_curve.discount(pay_date)
                for basis, pay_date in zip(self._basis, self._fixed_leg_pay_dates)
            ]
        )

    def compute_forward(self, model):
        return (
            model.rate_curve.discount(self.start_date) - model.rate_curve.discount(self.end_date)
        ) / self.compute_annuity(model)

    def compute_numeraire(self, model):
        return self.compute_annuity(model)

    def get_swap_dates_flows(self, strike, call=True):
        dates = np.concatenate([[self.start_date], self._fixed_leg_pay_dates])
        flows = np.concatenate([[1.0], -strike * self._basis])
        flows[-1] -= 1.0
        if not call:
            flows = -flows
        return dates, flows
