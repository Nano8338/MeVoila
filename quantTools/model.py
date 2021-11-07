from . import vanilla as qtvanilla


def _compute_model_atm_option_values(model, underlyings):
    atm_strikes = [underlying.compute_forward(model) for underlying in underlyings]
    start_dates = [underlying.start_date for underlying in underlyings]
    option_values = [
        model.compute_option_value(underlying, strike, call=True)
        for underlying, strike in zip(underlyings, atm_strikes)
    ]

    return atm_strikes, start_dates, option_values


def compute_model_atm_option_values(model, underlyings):
    _, start_dates, option_values = _compute_model_atm_option_values(model, underlyings)
    return start_dates, option_values


def compute_model_atm_volatilities(model, underlyings, vol_type):
    atm_strikes, start_dates, option_values = _compute_model_atm_option_values(model, underlyings)
    discounts = [underlying.compute_numeraire(model) for underlying in underlyings]
    expiries = [underlying.expiry for underlying in underlyings]

    implied_vol = [
        qtvanilla.compute_implied_atm_vol(atm_option, discount, expiry, forward, vol_type)
        for atm_option, discount, expiry, forward in zip(option_values, discounts, expiries, atm_strikes)
    ]

    return start_dates, implied_vol


def compute_model_option_values(model, underlying, strikes):
    _, start_dates, option_values = _compute_model_atm_option_values(model, underlying)
    return start_dates, option_values


def compute_model_smile(model, underlying, nb_std_dev, vol_type):
    forward = underlying.compute_forward(model)

    atm_option = model.compute_option_value(underlying, forward, call=True)
    discount = underlying.compute_numeraire(model)
    atm_vol = qtvanilla.compute_implied_atm_vol(atm_option, discount, underlying.expiry, forward, vol_type)
    strikes = qtvanilla.build_range_strikes(forward, atm_vol, underlying.expiry, nb_std_dev, vol_type)

    option_values = [model.compute_option_value(underlying, strike, call=True) for strike in strikes]
    implied_vol = qtvanilla.compute_implied_vol(
        option_values,
        strikes,
        discount,
        forward,
        underlying.expiry,
        call=True,
        vol_type=vol_type,
    )

    return strikes, implied_vol
