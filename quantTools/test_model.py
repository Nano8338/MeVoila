import seaborn as sns
from matplotlib import pyplot as plt
from scipy import integrate
import pandas as pd

from .model import *


def test_vanilla_option_call_put_parity():
    forward = 0.02
    vol_LN = 0.20
    vol_N = 0.0020
    T = 2.0
    r = 0.01
    discount = math.exp(-r * T)

    strikes = np.array([forward * 0.9, forward, forward * 1.1])

    call_value_ln = compute_option_ln(forward, strikes, T, vol_LN, discount, call=True)
    put_value_ln = compute_option_ln(forward, strikes, T, vol_LN, discount, call=False)

    call_value_ln_intrinsic = compute_option_ln(forward, strikes, T, 0.0, discount, call=True)
    put_value_ln_intrinsic = compute_option_ln(forward, strikes, T, 0.0, discount, call=False)

    call_value_normal = compute_option_normal(forward, strikes, T, vol_N, discount, call=True)
    put_value_normal = compute_option_normal(forward, strikes, T, vol_N, discount, call=False)

    call_value_normal_intrinsic = compute_option_normal(forward, strikes, T, 0.0, discount, call=True)
    put_value_normal_intrinsic = compute_option_normal(forward, strikes, T, 0.0, discount, call=False)

    df_call_put_parity = pd.DataFrame(
        {
            "forward - strike": (forward - strikes) * discount,
            "call - put (normal)": call_value_normal - put_value_normal,
            "call - put (normal intrinsic)": call_value_normal_intrinsic - put_value_normal_intrinsic,
            "call - put (ln)": call_value_ln - put_value_ln,
            "call - put (ln intrinsic)": call_value_ln_intrinsic - put_value_ln_intrinsic,
            "call (normal)": call_value_normal,
            "call (normal intrinsic)": call_value_normal_intrinsic,
            "call (ln)": call_value_ln,
            "call (ln intrinsic)": call_value_ln_intrinsic,
        },
        index=strikes,
    )

    display(df_call_put_parity)


def test_implied_vol_formula():
    forward = 0.02
    vol_LN = 0.20
    vol_N = 0.0020
    T = 2.0
    r = 0.01
    discount = math.exp(-r * T)

    strikes = np.array([forward * 0.9, forward, forward * 1.1])

    call_value_ln = compute_option_ln(forward, strikes, T, vol_LN, discount, call=True)
    put_value_ln = compute_option_ln(forward, strikes, T, vol_LN, discount, call=False)

    call_value_normal = compute_option_normal(forward, strikes, T, vol_N, discount, call=True)
    put_value_normal = compute_option_normal(forward, strikes, T, vol_N, discount, call=False)

    call_implied_vol_normal = compute_implied_vol_normal(call_value_normal, discount, forward, strikes, T, call=True)
    put_implied_vol_normal = compute_implied_vol_normal(put_value_normal, discount, forward, strikes, T, call=False)

    call_implied_vol_ln = compute_implied_vol_ln(call_value_ln, discount, forward, strikes, T, call=True)
    put_implied_vol_ln = compute_implied_vol_ln(put_value_ln, discount, forward, strikes, T, call=False)

    df_implied_vol = pd.DataFrame(
        {
            "implied vol normal (call)": call_implied_vol_normal,
            "implied vol normal (put)": put_implied_vol_normal,
            "implied vol log-normal (call)": call_implied_vol_ln,
            "implied vol log-normal (put)": put_implied_vol_ln,
            "option normal (call)": call_value_normal,
            "option normal (put)": put_value_normal,
            "option log-normal (call)": call_value_ln,
            "option log-normal (put)": put_value_ln,
        },
        index=strikes,
    )

    display(df_implied_vol)


def test_rate_curve_linear_interpolation():
    rate_curve = RateCurve([1.0, 2.0], [1.0, 2.0])
    x = np.linspace(0.0, 5.0, 501)
    sns.lineplot(x=x, y=rate_curve.rate(x))
    plt.show()


def test_hw_vol_flat_interpolation():
    rate_curve = RateCurve([1.0, 2.0], [1.0, 2.0])
    hull_white = ModelHullWhite(
        1.0,
        vol_dates=[1.0, 2.0, 3.0, 4.0],
        vol_values=[0.01, 0.02, 0.01, 0.03],
        rate_curve=rate_curve,
    )
    x = np.linspace(0.0, 5.0, 501)
    sns.lineplot(x=x, y=hull_white.vol(x))
    plt.show()


def validate_integral_exp_ax(sample_a, bound_inf, bound_sup):
    def f_exp_ax(a, x):
        return math.exp(a * x)

    integral_analytic_exp = [integrate_exp(bound_inf, bound_sup, a) for a in sample_a]
    integral_approx_exp = [integrate.quad(lambda x: f_exp_ax(a, x), bound_inf, bound_sup)[0] for a in sample_a]

    data_integral_exp = pd.DataFrame(
        {
            "meanRev": sample_a,
            "analytic": integral_analytic_exp,
            "approx": integral_approx_exp,
        }
    ).set_index("meanRev")
    sns.lineplot(data=data_integral_exp)
    plt.show()


def validate_integral_piecewise_x_exp_ax(sample_a, piecewise_curve, bound_inf, bound_sup):
    def f_piecewise_x_exp_ax(curve, a, x):
        return curve(x) * math.exp(a * x)

    integral_piecewise_approx_exp = [
        integrate.quad(lambda x: f_piecewise_x_exp_ax(piecewise_curve, a, x), bound_inf, bound_sup)[0]
        for a in sample_a
    ]
    integral_piecewise_analytic_exp = [
        integrate_exp_piecewise(a, piecewise_curve, bound_inf, bound_sup) for a in sample_a
    ]

    df_integral_piecewise_wide = pd.DataFrame(
        {
            "meanRev": sample_a,
            "approx": integral_piecewise_approx_exp,
            "analytic": integral_piecewise_analytic_exp,
        }
    ).set_index("meanRev")
    sns.lineplot(data=df_integral_piecewise_wide)
    plt.show()


def validate_integral_piecewise_x_hw_b(sample_a, piecewise_curve, T, bound_inf, bound_sup):
    def f_piecewise_hw_b(curve, a, x):
        return curve(x) * hw_b(a, T - x) * hw_b(a, T - x)

    integral_piecewise_approx_hw_b = [
        integrate.quad(lambda x: f_piecewise_hw_b(piecewise_curve, a, x), bound_inf, bound_sup)[0] for a in sample_a
    ]
    integral_piecewise_analytic_hw_b = [
        integrate_hw_b_square_piecewise(a, piecewise_curve, bound_inf, bound_sup, T) for a in sample_a
    ]

    df_integral_piecewise_hw_b = pd.DataFrame(
        {
            "meanRev": sample_a,
            "approx": integral_piecewise_approx_hw_b,
            "analytic": integral_piecewise_analytic_hw_b,
        }
    ).set_index("meanRev")
    sns.lineplot(data=df_integral_piecewise_hw_b)
    plt.show()
