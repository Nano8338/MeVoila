import json
import copy
import math
import numpy as np
import pandas as pd
from scipy import stats, interpolate

from .market_data import RateCurve
from .instruments import ForwardRate, BackwardRate, SwapRate
from .monte_carlo import MonteCarloGenerator

from . import tools as qttools
from . import vanilla as qtvanilla
from . import instruments as qtinstruments

label_dates = "T"
label_caplets = "Caplet price"
label_rate = "Rate"
label_expiry = "T expiry"
label_mean_rev = "mean rev"
label_vol = "volatility"
label_type = "type"
label_error_vol = "vol error (rel)"
label_correl = "correlation"


# Math tools

# (1-e^(-ax))/a
def hw_b(a, x):
    return x if abs(a) < 1.0e-10 else (1.0 - math.exp(-a * x)) / a


# I{ e^(ax)*dx } from x_inf to x_sup
def integrate_exp(x_inf, x_sup, a):
    return x_sup - x_inf if abs(a) < 1.0e-10 else math.exp(a * x_sup) * hw_b(a, x_sup - x_inf)


# I{ f(x)*e^(ax)*dx } from x_inf to x_sup
def integrate_exp_piecewise(a, piece_wise_curve, x_inf, x_sup):
    l_x = qttools.get_sorted_list(piece_wise_curve.x, x_inf, x_sup)
    integral = 0
    for x_prev, x_next in zip(l_x[:-1], l_x[1:]):
        integral += piece_wise_curve(0.5 * (x_prev + x_next)) * integrate_exp(x_prev, x_next, a)
    return integral


# I{ (1-e^(-a(t-x)))/a*dx } from x_inf to x_sup
def integrate_hw_b(x_inf, x_sup, a, t):
    if abs(a) < 1.0e-10:
        return (x_sup - x_inf) * (t - 0.5 * (x_inf + x_sup))
    else:
        return ((x_sup - x_inf) - math.exp(-a * (t - x_sup)) * hw_b(a, x_sup - x_inf)) / a


def integrate_hw_b_square(x_inf, x_sup, a, t):
    if abs(a) < 1.0e-10:
        integral_value = (((t - x_inf) ** 3.0) - ((t - x_sup) ** 3.0)) / 3.0
    else:
        factor = math.exp(-a * t)
        integral_value = (
            (x_sup - x_inf)
            - 2.0 * factor * integrate_exp(x_inf, x_sup, a)
            + factor * factor * integrate_exp(x_inf, x_sup, 2.0 * a)
        ) / (a * a)
    return max(integral_value, 0.0)


def integrate_hw_b_square_piecewise(a, piece_wise_curve, x_inf, x_sup, T):
    l_x = qttools.get_sorted_list(piece_wise_curve.x, x_inf, x_sup)
    integral = 0
    for x_prev, x_next in zip(l_x[:-1], l_x[1:]):
        integral += piece_wise_curve(0.5 * (x_prev + x_next)) * integrate_hw_b_square(x_prev, x_next, a, T)
    return integral


def integrate_hw_b_piecewise(a, piece_wise_curve, x_inf, x_sup, T):
    l_x = qttools.get_sorted_list(piece_wise_curve.x, x_inf, x_sup)
    integral = 0
    for x_prev, x_next in zip(l_x[:-1], l_x[1:]):
        integral += piece_wise_curve(0.5 * (x_prev + x_next)) * integrate_hw_b(x_prev, x_next, a, T)
    return integral


class ModelHullWhite:
    def __init__(self, mean_rev, vol_dates, vol_values, rate_curve):
        vol_dates_hw, vol_values_hw = qttools.get_curve_data_with_extrapolation_pillars(vol_dates, vol_values)

        self.mean_rev = mean_rev
        self._vol_dates = vol_dates_hw
        self._vol_values = vol_values_hw

        self.vol_curve = interpolate.interp1d(self._vol_dates, self._vol_values, kind="next", fill_value="extrapolate")
        self.rate_curve = rate_curve
        self._vol_square_curve = qttools.to_squared_curve(self.vol_curve)

    def __repr__(self):
        return json.dumps(
            {
                "type": "ModelHullWhite",
                "meanRev": self.mean_rev,
                "volCurve": {
                    "dates": self._vol_dates,
                    "volatilities": self._vol_values,
                },
                "rateCurve": repr(self.rate_curve),
            }
        )

    @classmethod
    def build_default(cls, mean_rev, hw_flat_vol=0.001):
        return cls(mean_rev, [1.0], [hw_flat_vol], RateCurve([0.0], [0.01]))

    def vol(self, date):
        return self.vol_curve(date)

    def inst_vol_bond(self, t, T_bond):
        return self.vol_curve(t) * hw_b(self.mean_rev, T_bond - t) if t < T_bond else 0.0

    def inst_vol_bond_ratio(self, t, T_bond_num, T_bond_denom):
        return self.inst_vol_bond(t, max(T_bond_num, T_bond_denom)) - self.inst_vol_bond(
            t, min(T_bond_num, T_bond_denom)
        )

    # integral(t_inf, t_sup) of variance of P(s,t_bond)
    def compute_integrated_variance_extended_bond(self, t_inf, t_sup, t_bond):

        t_integral_inf = max(0.0, t_inf)
        t_integral_sup = min(t_sup, t_bond)

        if t_integral_inf < t_integral_sup and t_integral_inf < t_bond:
            return max(
                integrate_hw_b_square_piecewise(
                    self.mean_rev,
                    self._vol_square_curve,
                    t_integral_inf,
                    t_integral_sup,
                    t_bond,
                ),
                0.0,
            )
        else:
            return 0.0

    # integral(0, t) of variance of P(s,t1)/P(s,t2)
    def compute_integrated_variance_extended_bond_ratio(self, t, t1, t2):
        if t > 0:
            t_inf = min(t1, t2)
            t_sup = max(t1, t2)

            if t_sup > 0.0:
                t_bond_ratio_end = max(0.0, min(t_inf, t))
                t_extended_bond_end = max(0.0, min(t, t_sup))

                factor_ratio_start = hw_b(self.mean_rev, t_sup - t_inf) * math.exp(-self.mean_rev * t_inf)
                variance_bond_ratio = (
                    factor_ratio_start
                    * factor_ratio_start
                    * integrate_exp_piecewise(
                        2.0 * self.mean_rev,
                        self._vol_square_curve,
                        0.0,
                        t_bond_ratio_end,
                    )
                )
                variance_extended_bond = self.compute_integrated_variance_extended_bond(
                    t_bond_ratio_end, t_extended_bond_end, t_sup
                )
                return variance_bond_ratio + variance_extended_bond
            else:
                return 0.0
        else:
            return 0

    #  computes integrated vol of P(s,t1)/P(s,t2) = 1 + delta * F(s,t1,t2)
    def compute_integrated_vol_extended_bond_ratio(self, t, t1, t2):
        return math.sqrt(self.compute_integrated_variance_extended_bond_ratio(t, t1, t2))

    # Call payoff is: basis * max(Libor - K;0)
    def compute_caplet(self, rate_index, strike, call):
        discount = self.rate_curve.discount(rate_index.pay_date)
        forward = self.rate_curve.forward_rate(rate_index.start_date, rate_index.end_date)
        basis = rate_index.tenor
        integrated_variance = self.compute_integrated_variance_extended_bond_ratio(
            rate_index.expiry, rate_index.start_date, rate_index.end_date
        )
        integrated_vol = math.sqrt(integrated_variance)  # equivalent to vol * sqrt(T)

        return (
            qtvanilla.compute_option_ln(
                1.0 + basis * forward,
                1.0 + basis * strike,
                1.0,
                integrated_vol,
                discount,
                call,
            )
        )

    def compute_swaption_variance_approx_normal(self, swap_rate):
        mean_rev = max(1.0e-10, self.mean_rev)
        fwd = swap_rate.compute_forward(self)
        annuity = swap_rate.compute_annuity(self)
        dates, flows = swap_rate.get_swap_dates_flows(fwd)
        coeff_i = [
            self.rate_curve.discount(date) * flow * math.exp(-mean_rev * date) for date, flow in zip(dates, flows)
        ]
        coeff = np.sum(coeff_i) / (mean_rev * annuity)
        integral_variance = integrate_exp_piecewise(2.0 * mean_rev, self._vol_square_curve, 0.0, swap_rate.expiry)

        return integral_variance * coeff * coeff

    def compute_swaption(self, swap_rate, strike, call):
        forward = swap_rate.compute_forward(self)
        annuity = swap_rate.compute_annuity(self)
        integrated_variance = self.compute_swaption_variance_approx_normal(swap_rate)

        return qtvanilla.compute_option_normal(
            forward, strike, 1.0, math.sqrt(integrated_variance), annuity, call=call
        )

    def compute_option_value(self, underlying, strike, call):
        return (
            self.compute_swaption(underlying, strike, call)
            if isinstance(underlying, SwapRate)
            else self.compute_caplet(underlying, strike, call)
        )

    def compute_caplet_quanto(self, strike, rate_index, call, correl, fx_vol):
        discount = self.rate_curve.discount(rate_index.pay_date)
        forward = self.rate_curve.forward_rate(rate_index.start_date, rate_index.end_date)
        basis = rate_index.tenor
        integrated_variance = self.compute_integrated_variance_extended_bond_ratio(
            rate_index.expiry, rate_index.start_date, rate_index.end_date
        )
        integrated_vol = math.sqrt(integrated_variance)  # equivalent to vol * sqrt(T)

        drift = (
            -correl
            * fx_vol
            * self.compute_integrated_vol_extended_bond_ratio(
                rate_index.expiry, rate_index.start_date, rate_index.end_date
            )
        )

        return (
            1.0
            / basis
            * qtvanilla.compute_option_ln(
                (1.0 + basis * forward) * math.exp(drift),
                1.0 + basis * strike,
                1.0,
                integrated_vol,
                discount,
                call,
            )
        )


def compute_model_atm_swaption_vol(model, swap_rate, vol_type):
    fwd = swap_rate.compute_forward(model)
    option_value = model.compute_swaption(swap_rate, fwd, call=True)
    return qtvanilla.compute_implied_atm_vol(
        option_value, swap_rate.compute_annuity(model), swap_rate.expiry, fwd, vol_type
    )


def compute_implied_atm_variance(rate_index, caplet_price, rate_curve):
    discount = rate_curve.discount(rate_index.pay_date)
    tenor = rate_index.tenor
    cumul_half_var = 0.5 * (
        1.0
        + caplet_price
        * tenor
        / ((1.0 + tenor * rate_curve.forward_rate(rate_index.start_date, rate_index.end_date)) * discount)
    )
    return (2.0 * stats.norm.ppf(cumul_half_var)) ** 2


def compute_hw_implied_atm_vol(hw_model, rate_indices, normal_vol):
    fwds = [
        hw_model.rate_curve.forward_rate(rate_index.start_date, rate_index.end_date) for rate_index in rate_indices
    ]
    discounts = [hw_model.rate_curve.discount(rate_index.pay_date) for rate_index in rate_indices]
    expiries = [rate_index.expiry for rate_index in rate_indices]
    caplets = [hw_model.compute_caplet(rate_index, fwd, call=True) for rate_index, fwd in zip(rate_indices, fwds)]
    return [
        qtvanilla.compute_implied_atm_vol(caplet, discount, expiry, forward, normal_vol)
        for caplet, discount, expiry, forward in zip(caplets, discounts, expiries, fwds)
    ]


def get_valid_quotations(l_rate_indices, l_hw_atm_caplet_prices):
    index_start = 0
    for i in range(len(l_rate_indices)):
        index_start = i
        if l_rate_indices[i].expiry > 0.0001:
            break
    return copy.deepcopy(l_rate_indices[index_start:]), copy.deepcopy(l_hw_atm_caplet_prices[index_start:])


def build_implied_model_atm_vol_normal(hw_model, rate_indices, label_key, df_vol_ref=None):
    expiries = [rate_index.expiry for rate_index in rate_indices]
    implied_vol = compute_hw_implied_atm_vol(hw_model, rate_indices, normal_vol=True)
    first_index = 1 if expiries[0] < 0.0000001 else 0
    df_impled_vol = pd.DataFrame(
        {
            label_dates: expiries[first_index:],
            label_vol: implied_vol[first_index:],
            label_type: label_key,
        }
    )
    if df_vol_ref is not None:
        df_impled_vol[label_error_vol] = np.divide(
            df_impled_vol[label_vol] - df_vol_ref[label_vol], df_vol_ref[label_vol]
        )
    return df_impled_vol
