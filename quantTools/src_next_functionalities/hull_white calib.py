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


# region Calibration


def compute_caplet_analytic_atm_prices(rate_model, underlyings, call=True):
    atm_strikes = [
        rate_model.rate_curve.forward_rate(underlying.start_date, underlying.end_date) for underlying in underlyings
    ]
    return pd.DataFrame(
        {
            label_dates: [underlying.start_date for underlying in underlyings],
            label_caplets: [
                rate_model.compute_caplet(underlying, strike, call)
                for underlying, strike in zip(underlyings, atm_strikes)
            ],
        }
    )


def calibrate_HW_vol(l_rate_indices, l_hw_atm_caplet_prices, mean_rev, rate_curve):
    hw_vol_term_structure = []
    expiry_dates = []

    prev_expiry_date = 0

    for rate_index, caplet_price in zip(l_rate_indices, l_hw_atm_caplet_prices):
        target_variance = compute_implied_atm_variance(rate_index, caplet_price, rate_curve)
        expiry_date = rate_index.expiry
        expiry_dates.append(expiry_date)
        hw_vol_term_structure.append(1.0)
        hw_model = ModelHullWhite(mean_rev, expiry_dates, hw_vol_term_structure, rate_curve)
        model_variance_accumulated = hw_model.compute_integrated_variance_extended_bond_ratio(
            prev_expiry_date, rate_index.start_date, rate_index.end_date
        )
        model_variance_proxy = hw_model.compute_integrated_variance_extended_bond_ratio(
            expiry_date, rate_index.start_date, rate_index.end_date
        )

        hw_var = max(target_variance - model_variance_accumulated, 0.0) / (
            model_variance_proxy - model_variance_accumulated
        )
        hw_vol_term_structure[-1] = math.sqrt(hw_var)
        prev_expiry_date = expiry_date

    return pd.DataFrame({"T": expiry_dates, "Vol HW": hw_vol_term_structure})


def calibrate_HW_vol_next_RFR(l_rate_indices, l_hw_atm_caplet_prices, mean_rev, rate_curve):
    hw_vol_term_structure = []
    expiry_dates = []

    prev_expiry_date = 0
    true_prev_expiry_date = 0

    for rate_index, caplet_price in zip(l_rate_indices, l_hw_atm_caplet_prices):
        target_variance = compute_implied_atm_variance(rate_index, caplet_price, rate_curve)
        expiry_date = rate_index.expiry
        expiry_dates.append(expiry_date)
        hw_vol_term_structure.append(1.0)
        hw_model = ModelHullWhite(mean_rev, expiry_dates, hw_vol_term_structure, rate_curve)
        model_variance_accumulated = hw_model.compute_integrated_variance_extended_bond_ratio(
            true_prev_expiry_date, rate_index.start_date, rate_index.end_date
        )
        model_variance_proxy = hw_model.compute_integrated_variance_extended_bond_ratio(
            expiry_date, rate_index.start_date, rate_index.end_date
        )

        hw_var = max(target_variance - model_variance_accumulated, 0.0) / (
            model_variance_proxy - model_variance_accumulated
        )
        hw_vol_term_structure[-1] = math.sqrt(hw_var)
        expiry_dates[-1] = prev_expiry_date
        true_prev_expiry_date = prev_expiry_date
        prev_expiry_date = expiry_date

    return pd.DataFrame({"T": expiry_dates, "Vol HW": hw_vol_term_structure})


def calibrate_HW_vol_generic(
    l_rate_indices,
    l_hw_atm_caplet_prices,
    mean_rev,
    rate_curve,
    f_transform_target,
    f_add_calib_expiry_date,
    f_get_prev_expiry_date,
):
    hw_vol_term_structure = []
    expiry_dates = []
    expiry_dates_calib = []

    prev_expiry_date_calib = 0

    l_rate_indices_valid, l_hw_atm_caplet_prices_valid = get_valid_quotations(l_rate_indices, l_hw_atm_caplet_prices)
    (
        l_rate_calib_indices,
        l_rate_target_indices,
        l_hw_atm_calib_prices,
    ) = f_transform_target(l_rate_indices_valid, l_hw_atm_caplet_prices_valid, mean_rev, rate_curve)

    for rate_calib_index, rate_index, caplet_price in zip(
        l_rate_calib_indices, l_rate_target_indices, l_hw_atm_calib_prices
    ):
        target_variance = compute_implied_atm_variance(rate_calib_index, caplet_price, rate_curve)
        expiry_date = rate_index.expiry
        expiry_date_calib = rate_calib_index.expiry

        expiry_dates.append(expiry_date)
        f_add_calib_expiry_date(expiry_dates_calib, expiry_date_calib, expiry_dates)
        hw_vol_term_structure.append(1.0)
        hw_model = ModelHullWhite(mean_rev, expiry_dates_calib, hw_vol_term_structure, rate_curve)
        model_variance_accumulated = hw_model.compute_integrated_variance_extended_bond_ratio(
            prev_expiry_date_calib,
            rate_calib_index.start_date,
            rate_calib_index.end_date,
        )
        model_variance_proxy = hw_model.compute_integrated_variance_extended_bond_ratio(
            expiry_date_calib, rate_calib_index.start_date, rate_calib_index.end_date
        )

        hw_var = (
            max(target_variance - model_variance_accumulated, 0.0)
            / (model_variance_proxy - model_variance_accumulated)
            if model_variance_proxy > model_variance_accumulated
            else 0.0
        )
        hw_vol_term_structure[-1] = math.sqrt(max(0.0, hw_var))
        prev_expiry_date_calib = expiry_dates_calib[-1]

    return ModelHullWhite(mean_rev, expiry_dates, hw_vol_term_structure, rate_curve)


def calibrate_HW_vol_bootstrap_backward(l_rate_indices, l_hw_atm_caplet_prices, mean_rev, rate_curve):
    def f_transform_to_clone(l_rate_indices, l_hw_atm_caplet_prices, mean_rev, rate_curve):
        return (
            copy.deepcopy(l_rate_indices),
            copy.deepcopy(l_rate_indices),
            copy.deepcopy(l_hw_atm_caplet_prices),
        )

    def f_append_calib_expiry_date(expiry_dates_calib, expiry_date_calib, expiry_dates):
        return expiry_dates_calib.append(expiry_date_calib)

    def f_get_last_calib_expiry_date(expiry_dates_calib, expiry_dates):
        return expiry_dates_calib[-1]

    return calibrate_HW_vol_generic(
        l_rate_indices,
        l_hw_atm_caplet_prices,
        mean_rev,
        rate_curve,
        f_transform_target=f_transform_to_clone,
        f_add_calib_expiry_date=f_append_calib_expiry_date,
        f_get_prev_expiry_date=f_get_last_calib_expiry_date,
    )


def calibrate_HW_vol_shifted_bootstrap_backward(l_rate_indices, l_hw_atm_caplet_prices, mean_rev, rate_curve):
    def f_transform_to_shifted_targets(l_rate_indices, l_hw_atm_caplet_prices, mean_rev, rate_curve):
        return (
            copy.deepcopy(l_rate_indices[1:]),
            copy.deepcopy(l_rate_indices[:-1]),
            copy.deepcopy(l_hw_atm_caplet_prices[1:]),
        )

    def f_append_shifted_expiry_date(expiry_dates_calib, expiry_date_calib, expiry_dates):
        return expiry_dates_calib.append(expiry_dates[-1])

    def f_get_last_effective_expiry_date(expiry_dates_calib, expiry_dates):
        return expiry_dates[-1]

    return calibrate_HW_vol_generic(
        l_rate_indices,
        l_hw_atm_caplet_prices,
        mean_rev,
        rate_curve,
        f_transform_target=f_transform_to_shifted_targets,
        f_add_calib_expiry_date=f_append_shifted_expiry_date,
        f_get_prev_expiry_date=f_get_last_effective_expiry_date,
    )


def calibrate_HW_vol_bootstrap_eq_next_forward(l_rate_indices, l_hw_atm_caplet_prices, mean_rev, rate_curve):
    def f_transform_to_eq_next_fwd(l_bwd_rate_indices, l_hw_atm_caplet_prices, mean_rev, rate_curve):
        l_fwd_rate_indices = [
            ForwardRate(rate_index.end_date, rate_index.end_date + rate_index.tenor)
            for rate_index in l_bwd_rate_indices
        ]
        l_hw_atm_eq_prices = []
        for fwd_rate_index, bwd_rate_index, bwd_atm_caplet_price in zip(
            l_fwd_rate_indices, l_bwd_rate_indices, l_hw_atm_caplet_prices
        ):
            hw_flat_model = calibrate_HW_vol_bootstrap_backward(
                [bwd_rate_index], [bwd_atm_caplet_price], mean_rev, rate_curve
            )
            df_atm_price = compute_caplet_analytic_atm_prices(hw_flat_model, [fwd_rate_index])
            l_hw_atm_eq_prices.append(df_atm_price[label_caplets][0])

        return l_fwd_rate_indices, copy.deepcopy(l_bwd_rate_indices), l_hw_atm_eq_prices

    def f_append_calib_expiry_date(expiry_dates_calib, expiry_date_calib, expiry_dates):
        return expiry_dates_calib.append(expiry_date_calib)

    def f_get_last_calib_expiry_date(expiry_dates_calib, expiry_dates):
        return expiry_dates_calib[-1]

    return calibrate_HW_vol_generic(
        l_rate_indices,
        l_hw_atm_caplet_prices,
        mean_rev,
        rate_curve,
        f_transform_target=f_transform_to_eq_next_fwd,
        f_add_calib_expiry_date=f_append_calib_expiry_date,
        f_get_prev_expiry_date=f_get_last_calib_expiry_date,
    )


def calibrate_HW_vol_bootstrap_eq_forward(l_rate_indices, l_hw_atm_caplet_prices, mean_rev, rate_curve):
    def f_transform_to_eq_next_fwd(l_bwd_rate_indices, l_hw_atm_caplet_prices, mean_rev, rate_curve):
        l_fwd_rate_indices = []
        l_hw_atm_eq_prices = []
        for bwd_rate_index, bwd_atm_caplet_price in zip(l_bwd_rate_indices, l_hw_atm_caplet_prices):
            fwd_eq_start = compute_equivalent_fwd_expiry(bwd_rate_index.start_date, bwd_rate_index.tenor, mean_rev)
            l_fwd_rate_indices.append(ForwardRate(fwd_eq_start, fwd_eq_start + bwd_rate_index.tenor))
            hw_flat_model = calibrate_HW_vol_bootstrap_backward(
                [bwd_rate_index], [bwd_atm_caplet_price], mean_rev, rate_curve
            )
            df_atm_price = compute_caplet_analytic_atm_prices(hw_flat_model, [l_fwd_rate_indices[-1]])
            l_hw_atm_eq_prices.append(df_atm_price[label_caplets][0])

        return l_fwd_rate_indices, copy.deepcopy(l_bwd_rate_indices), l_hw_atm_eq_prices

    def f_append_calib_expiry_date(expiry_dates_calib, expiry_date_calib, expiry_dates):
        return expiry_dates_calib.append(expiry_date_calib)

    def f_get_last_calib_expiry_date(expiry_dates_calib, expiry_dates):
        return expiry_dates_calib[-1]

    return calibrate_HW_vol_generic(
        l_rate_indices,
        l_hw_atm_caplet_prices,
        mean_rev,
        rate_curve,
        f_transform_target=f_transform_to_eq_next_fwd,
        f_add_calib_expiry_date=f_append_calib_expiry_date,
        f_get_prev_expiry_date=f_get_last_calib_expiry_date,
    )


def compute_eq_hw_maturity(mean_rev, rate_index):
    start_date = rate_index.start_date
    tenor = rate_index.tenor
    eq_hw_maturity = (hw_b(mean_rev, tenor) ** 2) * hw_b(2.0 * mean_rev, start_date)

    if not rate_index.is_forward_looking:
        extra_maturity = (
            1
            / (mean_rev ** 2)
            * (
                tenor
                + 2.0 / mean_rev * math.exp(-mean_rev * tenor)
                - 1 / (2.0 * mean_rev) * math.exp(-2.0 * mean_rev * tenor)
                - 3.0 / (2.0 * mean_rev)
            )
        )
        eq_hw_maturity += extra_maturity
    return eq_hw_maturity


def compute_equivalent_fwd_expiry(start_date, tenor, mean_rev):
    bwd_maturity = compute_eq_hw_maturity(mean_rev, BackwardRate(start_date, start_date + tenor))
    return -math.log(1.0 - 2.0 * mean_rev * bwd_maturity / (hw_b(mean_rev, tenor) ** 2)) / (2.0 * mean_rev)


def calibrate_HW_vol_bootstrap_eq_forward_with_update(l_rate_indices, l_hw_atm_caplet_prices, mean_rev, rate_curve):
    def f_transform_to_eq_next_fwd(l_bwd_rate_indices, l_hw_atm_caplet_prices, mean_rev, rate_curve):
        l_fwd_rate_indices = []
        l_hw_atm_eq_prices = []
        for bwd_rate_index, bwd_atm_caplet_price in zip(l_bwd_rate_indices, l_hw_atm_caplet_prices):
            fwd_eq_start = compute_equivalent_fwd_expiry(bwd_rate_index.start_date, bwd_rate_index.tenor, mean_rev)
            l_fwd_rate_indices.append(ForwardRate(fwd_eq_start, fwd_eq_start + bwd_rate_index.tenor))
            hw_flat_model = calibrate_HW_vol_bootstrap_backward(
                [bwd_rate_index], [bwd_atm_caplet_price], mean_rev, rate_curve
            )
            df_atm_price = compute_caplet_analytic_atm_prices(hw_flat_model, [l_fwd_rate_indices[-1]])
            l_hw_atm_eq_prices.append(df_atm_price[label_caplets][0])

        return l_fwd_rate_indices, copy.deepcopy(l_bwd_rate_indices), l_hw_atm_eq_prices

    def f_append_calib_expiry_date(expiry_dates_calib, expiry_date_calib, expiry_dates):

        if len(expiry_dates_calib) > 1:
            expiry_dates_calib[-2] = expiry_dates[-3]

        return expiry_dates_calib.append(expiry_date_calib)

    def f_get_last_calib_expiry_date(expiry_dates_calib, expiry_dates):
        return expiry_dates_calib[-1]

    return calibrate_HW_vol_generic(
        l_rate_indices,
        l_hw_atm_caplet_prices,
        mean_rev,
        rate_curve,
        f_transform_target=f_transform_to_eq_next_fwd,
        f_add_calib_expiry_date=f_append_calib_expiry_date,
        f_get_prev_expiry_date=f_get_last_calib_expiry_date,
    )


# endregion


# region Code from notebook


# data & perturbed data


hw_model_calib = copy.deepcopy(hw_model)

label_target = "target"
df_model_vol_target = build_implied_model_atm_vol_normal(hw_model, bwd_rates, label_target)

target_bwd_prices = df_caplets_bwd_analytic["Caplet price"].copy()
target_bwd_prices[2] = df_caplets_bwd_mc_mean_1M["Caplet price"][2]


# different calibrations

label_calib_rfr = "calib RFR"
hw_model_calib_rfr = calibrate_HW_vol_bootstrap_backward(bwd_rates, target_bwd_prices, mean_rev, flat_rate_curve)
df_model_vol_calib_rfr = build_implied_model_atm_vol_normal(
    hw_model_calib_rfr, bwd_rates, label_calib_rfr, df_model_vol_target
)

label_calib_next_rfr = "calib next RFR"
hw_model_calib_next_rfr = calibrate_HW_vol_shifted_bootstrap_backward(
    bwd_rates, target_bwd_prices, mean_rev, flat_rate_curve
)
df_model_vol_calib_next_rfr = build_implied_model_atm_vol_normal(
    hw_model_calib_next_rfr, bwd_rates, label_calib_next_rfr, df_model_vol_target
)

label_calib_eq_fwd_rate = "calib eq fwd rate"
hw_model_calib_eq_fwd_rate = calibrate_HW_vol_bootstrap_eq_forward(
    bwd_rates, target_bwd_prices, mean_rev, flat_rate_curve
)
df_model_vol_calib_eq_fwd_rate = build_implied_model_atm_vol_normal(
    hw_model_calib_eq_fwd_rate, bwd_rates, label_calib_eq_fwd_rate, df_model_vol_target
)

label_calib_eq_fwd_with_update = "calib eq fwd rate (update pillars)"
hw_model_calib_eq_fwd_with_update = calibrate_HW_vol_bootstrap_eq_forward_with_update(
    bwd_rates, target_bwd_prices, mean_rev, flat_rate_curve
)
df_model_vol_calib_eq_fwd_with_update = build_implied_model_atm_vol_normal(
    hw_model_calib_eq_fwd_with_update, bwd_rates, label_calib_eq_fwd_with_update, df_model_vol_target
)


# plot implied vols


df_model_implied_vol = pd.concat(
    [
        df_model_vol_target,
        df_model_vol_calib_rfr,
        df_model_vol_calib_next_rfr,
        df_model_vol_calib_next_eq_fwd_rate,
        df_model_vol_calib_eq_fwd_rate,
        df_model_vol_calib_eq_fwd_with_update,
    ]
)

fig_model_implied_vol = px.line(df_model_implied_vol, x=label_dates, y=label_vol, color=label_type)
fig_model_implied_vol.show()


# plot implied vols errors

fig_model_implied_vol_error = px.line(df_model_implied_vol, x=label_dates, y=label_error_vol, color=label_type)
fig_model_implied_vol_error.show()


# plot model vols

plot_model_vol(
    {
        label_target: hw_model,
        label_calib_rfr: hw_model_calib_rfr,
        label_calib_next_rfr: hw_model_calib_next_rfr,
        label_calib_next_eq_fwd_rate: hw_model_calib_next_eq_fwd_rate,
        label_calib_eq_fwd_rate: hw_model_calib_eq_fwd_rate,
        label_calib_eq_fwd_with_update: hw_model_calib_eq_fwd_with_update,
    }
)

# endregion
