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


# region Monte carlo simulation


def simulate_caplet_analytic_prices_forward_rate(hw_model, dates, tenor, simulations_N):
    mc_generator = MonteCarloGenerator()
    norm_rand_nb = mc_generator.simulate_normal(simulations_N)

    l_caplet_price_by_date = []

    for start_date in dates:
        end_date = start_date + tenor
        atm_strike = hw_model.rate_curve.forward_rate(start_date, end_date)
        equivalent_strike = 1.0 + tenor * atm_strike
        discount_start = hw_model.rate_curve.discount(start_date)
        discount_end = hw_model.rate_curve.discount(end_date)
        integrated_vol = hw_model.compute_integrated_vol_extended_bond_ratio(start_date, start_date, end_date)
        factor_discount_ratio = discount_start / discount_end * math.exp(-0.5 * integrated_vol * integrated_vol)
        simulated_discount_ratio = factor_discount_ratio * np.exp(integrated_vol * norm_rand_nb)
        l_caplet_price_by_date.append(
            discount_end / tenor * np.maximum(simulated_discount_ratio - equivalent_strike, 0)
        )

    return pd.DataFrame(
        {
            label_dates: np.repeat(dates, simulations_N),
            label_caplets: np.concatenate(l_caplet_price_by_date),
        }
    )


def simulate_caplet_analytic_prices_backward_rate(hw_model, start_dates, tenor, simulations_N):
    mc_generator = MonteCarloGenerator()
    dim_N = 2
    payoff_dates_N = 2
    norm_rand_nb = mc_generator.simulate_normal(simulations_N * dim_N * payoff_dates_N).reshape(
        dim_N * payoff_dates_N, simulations_N
    )
    mean_rev = hw_model.mean_rev

    l_caplet_price_by_date = []

    for start_date in start_dates:
        end_date = start_date + tenor
        atm_strike = hw_model.rate_curve.forward_rate(start_date, end_date)
        equivalent_strike = 1.0 + tenor * atm_strike

        prev_x_t = 0
        prev_t = 0

        discount_compounded = np.zeros((2, simulations_N))
        int_x_t = 0

        for i, t in enumerate([start_date, end_date]):
            t_discount = t
            t = max(t, 0)
            var_x_t = math.exp(-2.0 * mean_rev * t) * integrate_exp_piecewise(
                2.0 * mean_rev, hw_model._vol_square_curve, prev_t, t
            )
            cov_var_x_t = math.exp(-mean_rev * t) * integrate_exp_piecewise(
                mean_rev, hw_model._vol_square_curve, prev_t, t
            )
            var_int_x_t = integrate_hw_b_square_piecewise(mean_rev, hw_model._vol_square_curve, prev_t, t, t)
            cov_x_t_int_x_t = (cov_var_x_t - var_x_t) / mean_rev
            correl_t = cov_x_t_int_x_t / math.sqrt(var_int_x_t * var_x_t) if var_int_x_t > 0 and var_x_t > 0 else 0.0

            x_t = (
                math.sqrt(var_x_t)
                * norm_rand_nb[
                    2 * i,
                ]
                + prev_x_t * math.exp(-mean_rev * (t - prev_t))
            )
            int_x_t += (
                math.sqrt(var_int_x_t)
                * (
                    correl_t
                    * norm_rand_nb[
                        2 * i,
                    ]
                    + math.sqrt(1 - correl_t ** 2)
                    * norm_rand_nb[
                        2 * i + 1,
                    ]
                )
                + prev_x_t * hw_b(mean_rev, t - prev_t)
            )

            prev_x_t = x_t
            prev_t = t
            var_discount = integrate_hw_b_square_piecewise(hw_model.mean_rev, hw_model._vol_square_curve, 0, t, t)
            discount_compounded[i,] = (
                hw_model.rate_curve.discount(t_discount) * math.exp(-0.5 * var_discount) * np.exp(-int_x_t)
            )

        discount_start = discount_compounded[
            0,
        ]
        discount_end = discount_compounded[
            1,
        ]
        l_caplet_price_by_date.append(1 / tenor * np.maximum(discount_start - equivalent_strike * discount_end, 0))

    return pd.DataFrame(
        {
            label_dates: np.repeat(start_dates, simulations_N),
            label_caplets: np.concatenate(l_caplet_price_by_date),
        }
    )


def simulate_caplet_fwd_atm_prices(hw_model, l_rate_indices, simulations_N, mean=True):
    mc_generator = MonteCarloGenerator()
    norm_rand_nb = mc_generator.simulate_normal(simulations_N)

    l_caplet_price_by_date = []

    for rate_index in l_rate_indices:
        start_date = rate_index.start_date
        end_date = rate_index.end_date
        tenor = rate_index.tenor
        atm_strike = hw_model.rate_curve.forward_rate(start_date, end_date)
        equivalent_strike = 1.0 + tenor * atm_strike
        discount_start = hw_model.rate_curve.discount(start_date)
        discount_end = hw_model.rate_curve.discount(end_date)
        integrated_vol = hw_model.compute_integrated_vol_extended_bond_ratio(start_date, start_date, end_date)
        factor_discount_ratio = discount_start / discount_end * math.exp(-0.5 * integrated_vol * integrated_vol)
        simulated_discount_ratio = factor_discount_ratio * np.exp(integrated_vol * norm_rand_nb)
        l_caplet_price_by_date.append(
            discount_end / tenor * np.maximum(simulated_discount_ratio - equivalent_strike, 0)
        )

    start_dates = [rate_index.start_date for rate_index in l_rate_indices]

    df_caplets = pd.DataFrame(
        {
            label_dates: np.repeat(start_dates, simulations_N),
            label_caplets: np.concatenate(l_caplet_price_by_date),
        }
    )

    return df_caplets.groupby("T").mean().reset_index() if mean is True else df_caplets


def simulate_caplet_bwd_atm_prices(hw_model, l_rate_indices, simulations_N, mean=True):
    mc_generator = MonteCarloGenerator()
    dim_N = 2
    payoff_dates_N = 2
    norm_rand_nb = mc_generator.simulate_normal(simulations_N * dim_N * payoff_dates_N).reshape(
        dim_N * payoff_dates_N, simulations_N
    )
    mean_rev = hw_model.mean_rev

    l_caplet_price_by_date = []

    for rate_index in l_rate_indices:
        start_date = rate_index.start_date
        end_date = rate_index.end_date
        tenor = rate_index.tenor
        atm_strike = hw_model.rate_curve.forward_rate(start_date, end_date)
        equivalent_strike = 1.0 + tenor * atm_strike

        prev_x_t = 0
        prev_t = 0

        discount_compounded = np.zeros((2, simulations_N))
        int_x_t = 0

        for i, t in enumerate([start_date, end_date]):
            t_discount = t
            t = max(t, 0)
            var_x_t = math.exp(-2.0 * mean_rev * t) * integrate_exp_piecewise(
                2.0 * mean_rev, hw_model._vol_square_curve, prev_t, t
            )
            cov_var_x_t = math.exp(-mean_rev * t) * integrate_exp_piecewise(
                mean_rev, hw_model._vol_square_curve, prev_t, t
            )
            var_int_x_t = integrate_hw_b_square_piecewise(mean_rev, hw_model._vol_square_curve, prev_t, t, t)
            cov_x_t_int_x_t = (cov_var_x_t - var_x_t) / mean_rev
            correl_t = cov_x_t_int_x_t / math.sqrt(var_int_x_t * var_x_t) if var_int_x_t > 0 and var_x_t > 0 else 0.0

            x_t = (
                math.sqrt(var_x_t)
                * norm_rand_nb[
                    2 * i,
                ]
                + prev_x_t * math.exp(-mean_rev * (t - prev_t))
            )
            int_x_t += (
                math.sqrt(var_int_x_t)
                * (
                    correl_t
                    * norm_rand_nb[
                        2 * i,
                    ]
                    + math.sqrt(1 - correl_t ** 2)
                    * norm_rand_nb[
                        2 * i + 1,
                    ]
                )
                + prev_x_t * hw_b(mean_rev, t - prev_t)
            )

            prev_x_t = x_t
            prev_t = t
            var_discount = integrate_hw_b_square_piecewise(hw_model.mean_rev, hw_model._vol_square_curve, 0, t, t)
            discount_compounded[i,] = (
                hw_model.rate_curve.discount(t_discount) * math.exp(-0.5 * var_discount) * np.exp(-int_x_t)
            )

        discount_start = discount_compounded[
            0,
        ]
        discount_end = discount_compounded[
            1,
        ]
        l_caplet_price_by_date.append(1 / tenor * np.maximum(discount_start - equivalent_strike * discount_end, 0))

    start_dates = [rate_index.start_date for rate_index in l_rate_indices]

    df_caplets = pd.DataFrame(
        {
            label_dates: np.repeat(start_dates, simulations_N),
            label_caplets: np.concatenate(l_caplet_price_by_date),
        }
    )

    return df_caplets.groupby("T").mean().reset_index() if mean is True else df_caplets


def simulate_caplet_prices(hw_model, l_rate_indices, simulations_N, mean=True):
    is_fwd_rate = len(l_rate_indices) > 0 and l_rate_indices[0].is_forward_looking
    f_simulate_caplet_atm_prices = simulate_caplet_fwd_atm_prices if is_fwd_rate else simulate_caplet_bwd_atm_prices

    return f_simulate_caplet_atm_prices(hw_model, l_rate_indices, simulations_N, mean)


def simulate_fixed_tenor_caplet_prices(
    hw_model, tenor, l_rate_start_dates, backward_looking=False, simulations_N=10000, mean=True
):
    l_rate_indices = [
        qtinstruments.RateIndex(start_date, start_date + tenor, backward_looking) for start_date in l_rate_start_dates
    ]
    return simulate_caplet_prices(hw_model, l_rate_indices, simulations_N, mean)


# endregion


# region Code from notebook


simulations_N_1K = 1000
simulations_N_10K = 10000
simulations_N_100K = 100000
simulations_N_1M = 1000000


label_dates = "T"
label_caplets = "Caplet price"
label_pricing = "Pricing"
label_rate = "Rate"
label_expiry = "T expiry"
label_mean_rev = "mean rev"
label_vol = "volatility"
label_tenor = "tenor"
label_type = "type"
label_error_vol = "vol error (rel)"
label_correl = "correlation"


# region Caplet prices

flat_rate_curve = RateCurve(zc_dates=[0.0, 10.0], zc_values=[0.03, 0.03])
mean_rev = 0.02
hw_model = ModelHullWhite(
    mean_rev=mean_rev, vol_dates=[0.0, 10.0], vol_values=[0.02, 0.02], rate_curve=flat_rate_curve
)

tenor_6M = 0.5
last_rate_start_date = 2.0

fwd_rates = [ForwardRate(end_date - tenor_caplet, end_date) for end_date in end_dates]
bwd_rates = [BackwardRate(end_date - tenor_caplet, end_date) for end_date in end_dates]

df_caplets_fwd_analytic = compute_fixed_tenor_caplet_analytic_atm_prices(
    hw_model, tenor_6M, start_dates, backward_looking=False
)
df_caplets_bwd_analytic = compute_fixed_tenor_caplet_analytic_atm_prices(
    hw_model, tenor_6M, start_dates, backward_looking=True
)


simulations_N = simulations_N_1K
df_caplets_fwd_mc = simulate_fixed_tenor_caplet_prices(
    hw_model, tenor_6M, start_dates, backward_looking=False, simulations_N=simulations_N, mean=False
)
df_caplets_bwd_mc = simulate_fixed_tenor_caplet_prices(
    hw_model, tenor_6M, start_dates, backward_looking=True, simulations_N=simulations_N, mean=False
)


f, axes = plt.subplots(1, 2, figsize=(10, 5))

f.tight_layout()

sns.lineplot(data=df_caplets_fwd_mc, x=label_dates, y=label_caplets, ax=axes[0])
sns.lineplot(data=df_caplets_fwd_analytic, x=label_dates, y=label_caplets, ax=axes[0])

sns.lineplot(data=df_caplets_bwd_mc, x=label_dates, y=label_caplets, ax=axes[1])
sns.lineplot(data=df_caplets_bwd_analytic, x=label_dates, y=label_caplets, ax=axes[1])

# endregion


# endregion
