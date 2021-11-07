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


# region Quanto


def compute_equivalent_libor_expiry(t_start, t_end, target, is_libor):
    tenor = t_end - t_start
    t_expiry_start = max(t_start, 0.0)
    t_expiry_end = t_expiry_start if is_libor else max(t_end, 0.0)
    if target == "vol":
        return t_expiry_start + ((t_expiry_end - t_expiry_start) ** 3) / (3.0 * (tenor ** 2))
    else:
        return t_expiry_start + ((t_expiry_end - t_expiry_start) ** 2) / (2.0 * (tenor ** 2))


def compute_proxy_quanto_caplets(hw_model, start_dates, tenor, sample_mean_rev, sample_correl, fx_vol):
    hw_model_mean_rev = copy.deepcopy(hw_model)
    bwd_rates = [BackwardRate(start_date, start_date + tenor) for start_date in start_dates]
    fwd_rates = [ForwardRate(start_date, start_date + tenor) for start_date in start_dates]

    forwards = [hw_model_mean_rev.rate_curve.forward_rate(rate.start_date, rate.end_date) for rate in fwd_rates]
    discounts = [hw_model_mean_rev.rate_curve.discount(rate.pay_date) for rate in fwd_rates]

    l_df = []
    for is_libor, rates in dict({True: fwd_rates, False: bwd_rates}).items():
        t_equivalent_expiries = [
            compute_equivalent_libor_expiry(rate.start_date, rate.end_date, "vol", is_libor) for rate in rates
        ]
        T_equivalent_adj = [
            compute_equivalent_libor_expiry(rate.start_date, rate.end_date, "fwd", is_libor) for rate in rates
        ]

        for mean_rev in sample_mean_rev:

            hw_model_mean_rev.mean_rev = mean_rev
            options_hw = [
                hw_model_mean_rev.compute_caplet(rate_index, fwd, call=True) / discount
                for rate_index, fwd, discount in zip(rates, forwards, discounts)
            ]

            for correl in sample_correl:
                options_hw_quanto = [
                    hw_model_mean_rev.compute_caplet_quanto(fwd, rate_index, True, correl, fx_vol) / discount
                    for rate_index, fwd, discount in zip(rates, forwards, discounts)
                ]
                l_df.append(
                    pd.DataFrame(
                        {
                            label_dates: start_dates,
                            label_caplets: options_hw,
                            label_mean_rev: mean_rev,
                            label_rate: "Libor" if is_libor else "RFR",
                            label_correl: correl,
                            label_type: "HW (no-quanto adj)",
                        }
                    )
                )
                l_df.append(
                    pd.DataFrame(
                        {
                            label_dates: start_dates,
                            label_caplets: options_hw_quanto,
                            label_mean_rev: mean_rev,
                            label_rate: "Libor" if is_libor else "RFR",
                            label_correl: correl,
                            label_type: "HW",
                        }
                    )
                )

            for normal_vol in [True, False]:

                atm_volatilities_libor = compute_hw_implied_atm_vol(hw_model_mean_rev, fwd_rates, normal_vol)

                for correl in sample_correl:
                    quanto_adjs = [
                        (-correl * fx_vol * atm_vol * math.sqrt(T))
                        for atm_vol, T in zip(atm_volatilities_libor, T_equivalent_adj)
                    ]
                    quanto_fwds = [
                        (fwd + quanto_adj if normal_vol else fwd * math.exp(quanto_adj))
                        for fwd, quanto_adj in zip(forwards, quanto_adjs)
                    ]
                    if normal_vol:
                        options = [
                            qtvanilla.compute_option_normal(quanto_fwd, fwd, t_equivalent_expiry, atm_vol, 1.0, True)
                            for quanto_fwd, fwd, t_equivalent_expiry, atm_vol in zip(
                                quanto_fwds,
                                forwards,
                                t_equivalent_expiries,
                                atm_volatilities_libor,
                            )
                        ]
                    else:
                        options = [
                            qtvanilla.compute_option_ln(quanto_fwd, fwd, t_equivalent_expiry, atm_vol, 1.0, True)
                            for quanto_fwd, fwd, t_equivalent_expiry, atm_vol in zip(
                                quanto_fwds,
                                forwards,
                                t_equivalent_expiries,
                                atm_volatilities_libor,
                            )
                        ]

                    l_df.append(
                        pd.DataFrame(
                            {
                                label_dates: start_dates,
                                label_caplets: options,
                                label_mean_rev: mean_rev,
                                label_rate: "Libor" if is_libor else "RFR",
                                label_correl: correl,
                                label_type: "Normal approx" if normal_vol else "SLN approx",
                            }
                        )
                    )
    return pd.concat(l_df)


# endregion


# region Code from notebook


# region Code which should be here


def concat_model_vol_data(dict_hw_model):

    fig_model_vol = go.Figure()

    for label, hw_model in dict_hw_model.items():
        fig_model_vol.add_trace(
            go.Scatter(x=hw_model.vol_curve.x, y=hw_model.vol_curve.y, line_shape="vh", name=label)
        )

    fig_model_vol.show()


def build_model_vol(dict_hw_model):

    l_df_model = []

    for label, hw_model in dict_hw_model.items():
        l_df_model.append(
            pd.DataFrame({label_dates: hw_model.vol_curve.x, label_vol: hw_model.vol_curve.y, label_type: label})
        )

    return pd.concat(l_df_model)


def plot_model_vol(dict_hw_model):

    fig_model_vol = go.Figure()

    for label, hw_model in dict_hw_model.items():
        fig_model_vol.add_trace(
            go.Scatter(x=hw_model.vol_curve.x, y=hw_model.vol_curve.y, line_shape="vh", name=label)
        )

    fig_model_vol.show()


def get_sub_df(df_data, label_col, col_value):
    return df_data[df_data[label_col] == col_value]


def plot_quanto_rate(df_quanto_data, rate_type):

    df_quanto_data_libor = get_sub_df(df_quanto_data, label_rate, rate_type)

    for correl in df_quanto_data_libor[label_correl].unique():
        print(f"correl = {correl}")
        df_quanto_correl = get_sub_df(df_quanto_data_libor, label_correl, correl)
        # fig_quanto = px.line(data_frame=df_quanto_correl, x=label_dates, y=label_caplets, color = label_type, facet_col=label_mean_rev)
        # fig_quanto.show()
        sns.relplot(
            data=df_quanto_correl,
            x=label_dates,
            y=label_caplets,
            hue=label_type,
            style=label_type,
            col=label_mean_rev,
            kind="line",
        )
        plt.show()


# endregion


sample_mean_rev = [0.0, 1.0e-4, 0.01, 0.1, 1.0]
sample_correl = [-0.95, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 0.95]
fx_vol = 0.20

hw_model = ModelHullWhite(
    mean_rev=mean_rev, vol_dates=[0.0, 10.0], vol_values=[0.02, 0.02], rate_curve=flat_rate_curve
)

df_proxy_quanto_hw_caplets = compute_proxy_quanto_caplets(
    hw_model, start_dates, tenor_caplet, sample_mean_rev, [-0.95, -0.5, 0.0, 0.5, 0.95], fx_vol
)

plot_quanto_rate(df_proxy_quanto_hw_caplets, "Libor")

plot_quanto_rate(df_proxy_quanto_hw_caplets, "RFR")

# endregion
