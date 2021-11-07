# region Code from notebook


# VALIDATE VANILLAS OPTIONS

# test_vanilla_option_call_put_parity()
# test_implied_vol_formula()


#  VALIDATE HULL-WHITE


# test_rate_curve_linear_interpolation()
# test_hw_vol_flat_interpolation()


#  VALIDATE INTEGRALS

# validate_integral_exp_ax(sample_a = np.linspace(-0.1, 0.1, 201), bound_inf = 1.2, bound_sup = 2.3)

# piecewise_curve = interpolate.interp1d(x=[1.0, 2.0, 3.0, 4.0], y=[0.01, 0.02, 0.01, 0.03], kind='next', fill_value='extrapolate')
# validate_integral_piecewise_x_exp_ax(sample_a = np.linspace(-0.1, 0.1, 201), piecewise_curve=piecewise_curve, bound_inf = 0.0, bound_sup = 5.0)

# validate_integral_piecewise_x_hw_b(sample_a = np.linspace(-0.1, 0.1, 201), piecewise_curve = piecewise_curve, T = 7.0, bound_inf = 2.0, bound_sup = 5.0)

# endregion
