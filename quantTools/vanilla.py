import math
import numpy as np
import enum
import scipy
from scipy import stats

from . import tools as qttools


class VolType(enum.Enum):
    NORMAL = "Normal"
    LOG_NORMAL = "LogNormal"


# region Option value


def compute_option_normal(forward, strike, expiry, vol, discount, call):
    epsilon = 1.0 if call else -1.0
    intrinsic_value = np.maximum(epsilon * (forward - strike), 0.0)
    option_otm = 0.0

    if vol > 0.0 and expiry > 0.0:
        epsilon_otm = np.where(intrinsic_value > 0.0, -epsilon, epsilon)
        vracT = vol * math.sqrt(expiry)
        d1 = epsilon_otm * (forward - strike) / vracT
        option_otm = epsilon_otm * (forward - strike) * stats.norm.cdf(d1) + vracT * stats.norm.pdf(d1)

    return (intrinsic_value + option_otm) * discount


def compute_option_ln(forward, strike, expiry, vol, discount, call):
    epsilon = 1.0 if call else -1.0
    intrinsic_value = np.maximum(epsilon * (forward - strike), 0.0)
    option_otm = 0.0

    if vol > 0.0 and expiry > 0.0 and forward > 0.0 and np.all(strike > 0.0):
        epsilon_otm = np.where(intrinsic_value > 0.0, -epsilon, epsilon)
        vracT = vol * math.sqrt(expiry)
        d1 = np.log(forward / strike) / vracT + 0.5 * vracT
        d2 = d1 - vracT
        option_otm = epsilon_otm * (
            forward * stats.norm.cdf(epsilon_otm * d1) - strike * stats.norm.cdf(epsilon_otm * d2)
        )

    return (intrinsic_value + option_otm) * discount


# endregion

# region Implied volatilities


def __compute_implied_vol(premiums, discount, forward, strikes, expiry, call, f_option_value):
    strikes = qttools.to_array(strikes)
    ndisc_premiums = np.divide(qttools.to_array(premiums), discount)

    vracT = []
    for ndisc_premium, strike in zip(ndisc_premiums, strikes):

        def f_target(x):
            return f_option_value(forward, strike, 1.0, x, 1.0, call) - ndisc_premium

        vracT.append(scipy.optimize.brentq(f_target, 0, 10.0))

    return np.divide(vracT, math.sqrt(expiry))


def compute_implied_vol_ln(premiums, discount, forward, strike, expiry, call):
    if forward > 0.0 and np.alltrue(strike > 0.0) and expiry > 0.0:
        return __compute_implied_vol(premiums, discount, forward, strike, expiry, call, compute_option_ln)
    else:
        return np.full_like(premiums, np.nan, dtype="float")


def compute_implied_vol_normal(premiums, discount, forward, strike, expiry, call):
    if expiry > 0.0:
        return __compute_implied_vol(premiums, discount, forward, strike, expiry, call, compute_option_normal)
    else:
        return np.full_like(premiums, np.nan, dtype="float")


def compute_implied_vol(premiums, strikes, discount, forward, expiry, call, vol_type):
    return (
        compute_implied_vol_normal(premiums, discount, forward, strikes, expiry, call)
        if vol_type == VolType.NORMAL
        else compute_implied_vol_ln(premiums, discount, forward, strikes, expiry, call)
    )


# endregion

# region ATM implied volatilities


def _compute_implied_atm_vol_normal(atm_option, discount, expiry):
    return atm_option / discount * math.sqrt(2.0 * math.pi / expiry) if expiry > 0.0 else 0.0


def _compute_implied_atm_vol_SLN(atm_option, discount, expiry, forward):
    return (
        2.0 / math.sqrt(expiry) * stats.norm.ppf(0.5 * (1 + atm_option / (discount * forward)))
        if expiry > 0.0 and forward > 0.0
        else 0.0
    )


def compute_implied_atm_vol(atm_option, discount, expiry, forward, vol_type):
    return (
        _compute_implied_atm_vol_normal(atm_option, discount, expiry)
        if vol_type == VolType.NORMAL
        else _compute_implied_atm_vol_SLN(atm_option, discount, expiry, forward)
    )


def compute_implied_atm_volatilities(option_values, discounts, expiries, forwards, vol_type):
    return [
        compute_implied_atm_vol(atm_option, discount, expiry, forward, vol_type)
        for atm_option, discount, expiry, forward in zip(option_values, discounts, expiries, forwards)
    ]


# endregion

# region Tools


def build_range_strikes(forward, atm_vol, expiry, nb_std_dev, vol_type):
    vracT = nb_std_dev * atm_vol * math.sqrt(expiry)
    nb_strikes = 101
    if vracT > 0.0:
        moneyness = np.linspace(-vracT, vracT, nb_strikes)
        return forward + moneyness if vol_type == VolType.NORMAL else forward * np.exp(moneyness)
    else:
        return [forward]


# endregion
