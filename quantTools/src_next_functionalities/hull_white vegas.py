# region Code from notebook


# region Code which should be in code


def plot_model_vegas(swap_rate):
    flat_rate_curve = RateCurve(zc_dates=[0.0, 10.0], zc_values=[0.03, 0.03])
    sigma1 = 0.01
    sigma2 = 0.01
    vols = [sigma1, sigma2]
    hw_model = ModelHullWhite(
        mean_rev=0.02, vol_dates=[2.0, swap_rate.expiry], vol_values=vols, rate_curve=flat_rate_curve
    )
    fwd = swap_rate.computeForward(hw_model)

    option_value = hw_model.compute_swaption(swap_rate, fwd, True)
    vega = math.sqrt(swap_rate.expiry / (2.0 * math.pi)) * swap_rate.computeAnnuity(hw_model)

    implied_vol_ref = compute_model_atm_swaption_vol(hw_model, swap_rate, VolType.NORMAL)

    shift = 1.0e-9
    hw_model_shifted = ModelHullWhite(
        mean_rev=0.02,
        vol_dates=[2.0, swap_rate.expiry],
        vol_values=[sigma1 + shift, sigma2 + shift],
        rate_curve=flat_rate_curve,
    )
    option_value_shifted = hw_model_shifted.compute_swaption(swap_rate, fwd, True)

    vega_model = (option_value_shifted - option_value) / shift

    implied_vol_model_sns = []
    underlying_lengths = np.arange(1.0, 20.5, 1.0)

    for underlying_length in underlying_lengths:
        swap_rate_sns = SwapRate(swap_rate.start_date, underlying_length)
        implied_vol = compute_model_atm_swaption_vol(hw_model, swap_rate_sns, VolType.NORMAL)
        implied_vol_shifted = compute_model_atm_swaption_vol(hw_model_shifted, swap_rate_sns, VolType.NORMAL)
        implied_vol_model_sns.append((implied_vol_shifted - implied_vol) / shift)

    vega_fd = np.divide(vega_model, implied_vol_model_sns)

    df_vegas_analytic = pd.DataFrame({"underlying length": underlying_lengths, "vegas": vega, "type": "Analytic"})
    df_vegas_fd = pd.DataFrame(
        {"underlying length": underlying_lengths, "vegas": vega_fd, "type": "Finite diff (XVA)"}
    )

    df_vegas = pd.concat([df_vegas_analytic, df_vegas_fd])
    plot_data(df_vegas, label_x="underlying length", label_y="vegas", label_color="type")


def compute_model_vegas_1d(hw_model, swap_rate, vol_dates, vol_values, flat_rate_curve, shift):
    date_nb = len(vol_dates)
    vega_model = np.repeat(0.0, date_nb)
    fwd = swap_rate.computeForward(hw_model)

    prev_option_value = hw_model.compute_swaption(swap_rate, fwd, True)
    hw_model_shifted_full = ModelHullWhite(mean_rev, vol_dates, vol_values + shift, flat_rate_curve)

    option_value_shifted = hw_model_shifted_full.compute_swaption(swap_rate, fwd, True)

    return (option_value_shifted - prev_option_value) / shift


def compute_model_vegas(hw_model, swap_rate, vol_dates, vol_values, flat_rate_curve, shift):
    date_nb = len(vol_dates)
    vega_model = np.repeat(0.0, date_nb)
    fwd = swap_rate.computeForward(hw_model)

    prev_option_value = hw_model.compute_swaption(swap_rate, fwd, True)
    for i in reversed(range(date_nb)):
        vol_values_shifted = np.array(vol_values)
        for j in range(i, date_nb):
            vol_values_shifted[j] += shift
        hw_model_shifted = ModelHullWhite(
            mean_rev=0.02, vol_dates=vol_dates, vol_values=vol_values_shifted, rate_curve=flat_rate_curve
        )

        option_value_shifted = hw_model_shifted.compute_swaption(swap_rate, fwd, True)
        vega_model[i] = (option_value_shifted - prev_option_value) / shift
        prev_option_value = option_value_shifted

    return vega_model


def plot_model_vegas_2d(swap_rate, date1, date2):
    mean_rev = 0.02
    flat_rate_curve = RateCurve(zc_dates=[0.0, 10.0], zc_values=[0.03, 0.03])
    sigma1 = 0.01
    sigma2 = 0.01
    date_min = min(date1, date2)
    date_max = max(date1, date2, date_min + 1.0 / 12.0)
    vol_dates = [date_min, date_max]
    date_nb = len(vol_dates)
    vol_values = np.array([sigma1, sigma2])
    hw_model = ModelHullWhite(mean_rev, vol_dates, vol_values, flat_rate_curve)

    vega = math.sqrt(swap_rate.expiry / (2.0 * math.pi)) * swap_rate.computeAnnuity(hw_model)

    shift = 1.0e-6

    vega_model_1d = compute_model_vegas_1d(hw_model, swap_rate, vol_dates, vol_values, flat_rate_curve, shift)
    vega_model_2d = compute_model_vegas(hw_model, swap_rate, vol_dates, vol_values, flat_rate_curve, shift)

    hw_model_shifted_full = ModelHullWhite(mean_rev, vol_dates, vol_values + shift, flat_rate_curve)

    underlying_lengths = np.arange(1.0, 20.5, 1.0)
    vega_fd_2d = [[], []]
    vega_fd_1d = []
    for underlying_length in underlying_lengths:

        swap_rates_1d = SwapRate(swap_rate.expiry, underlying_length)
        jacobian_swap_rates = [SwapRate(vol_dates[j], underlying_length) for j in range(date_nb)]

        implied_vols_1d = compute_model_atm_swaption_vol(hw_model, swap_rates_1d, VolType.NORMAL)
        implied_vols_shifted_1d = compute_model_atm_swaption_vol(hw_model_shifted_full, swap_rates_1d, VolType.NORMAL)
        implied_vols_sns_1d = (implied_vols_shifted_1d - implied_vols_1d) / shift
        vega_fd_1d.append(vega_model_1d / implied_vols_sns_1d)

        jacobian_swap_rates = [SwapRate(vol_dates[j], underlying_length) for j in range(date_nb)]
        implied_vols = [
            compute_model_atm_swaption_vol(hw_model, swap_rate_sns, VolType.NORMAL)
            for swap_rate_sns in jacobian_swap_rates
        ]
        jacobian = np.identity(date_nb)

        prev_implied_vol = implied_vols
        for i in reversed(range(date_nb)):
            vol_values_shifted = np.array(vol_values)
            for j in range(i, date_nb):
                vol_values_shifted[j] += shift
            hw_model_shifted = ModelHullWhite(mean_rev, vol_dates, vol_values_shifted, flat_rate_curve)

            implied_vol_shifted = [
                compute_model_atm_swaption_vol(hw_model_shifted, swap_rate_sns, VolType.NORMAL)
                for swap_rate_sns in jacobian_swap_rates
            ]

            for j in range(date_nb):
                jacobian[i, j] = (implied_vol_shifted[j] - prev_implied_vol[j]) / shift

            prev_implied_vol = implied_vol_shifted

        vega_ul = np.linalg.inv(jacobian).dot(vega_model_2d)

        for i in range(len(vol_dates)):
            vega_fd_2d[i].append(vega_ul[i])

    df_vegas_analytic = pd.DataFrame({"underlying length": underlying_lengths, "vegas": vega, "type": "Analytic"})

    l_df_vegas = [df_vegas_analytic]
    for i in range(len(vol_dates)):
        df_vegas_fd = pd.DataFrame(
            {
                "underlying length": underlying_lengths,
                "vegas": vega_fd_2d[i],
                "type": f"Finite diff (2D - t={vol_dates[i]})",
            }
        )
        l_df_vegas.append(df_vegas_fd)
    l_df_vegas.append(
        pd.DataFrame(
            {
                "underlying length": underlying_lengths,
                "vegas": np.add(vega_fd_2d[0], vega_fd_2d[1]),
                "type": f"Finite diff (2D - Total)",
            }
        )
    )
    l_df_vegas.append(
        pd.DataFrame({"underlying length": underlying_lengths, "vegas": vega_fd_1d, "type": f"Finite diff (1D)"})
    )

    df_vegas = pd.concat(l_df_vegas)
    plot_data(df_vegas, label_x="underlying length", label_y="vegas", label_color="type")


def analyze_vegas(swap_rate):

    w_out = widgets.interactive(
        plot_model_vegas_2d,
        swap_rate=widgets.fixed(swap_rate),
        date1=widgets.FloatSlider(value=swap_rate.expiry, min=0.0, max=20.0, description="t1"),
        date2=widgets.FloatSlider(value=swap_rate.expiry, min=0.0, max=20.0, description="t2"),
    )
    display(w_out)


# endregion


flat_rate_curve = RateCurve(zc_dates=[0.0, 10.0], zc_values=[0.03, 0.03])
mean_rev = 0.02
hw_model = ModelHullWhite(
    mean_rev=mean_rev, vol_dates=[0.0, 10.0], vol_values=[0.02, 0.02], rate_curve=flat_rate_curve
)


plot_model_vegas(SwapRate(10.0, 5.0))
analyze_vegas(SwapRate(5.0, 5.0))

# endregion
