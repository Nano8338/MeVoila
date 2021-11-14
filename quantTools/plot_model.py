import ipywidgets as widgets
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from IPython.display import display

from .vanilla import VolType
from .instruments import RateType
from .hull_white import ModelHullWhite

from . import tools as qttools
from . import model as qtmodel
from . import instruments as qtinstruments

label_delta = "delta"
label_dates = "T"
label_caplets = "Caplet price"
label_rate = "Rate"
label_expiry = "T expiry"
label_mean_rev = "mean rev"
label_vol = "volatility"
label_type = "type"
label_error_vol = "vol error (rel)"
label_correl = "correlation"


# region UI TOOLS

# region WIDGET TOOLS


def build_widget_dropdown_df(df_data, col_label):
    values_label = df_data[col_label].unique()
    return widgets.Dropdown(options=values_label, description=f"{col_label}:")


def build_widget_dropdown(description, l_labels, l_values=None, input_value=None):
    l_values = range(len(l_labels)) if l_values is None else l_values
    return widgets.Dropdown(
        options=dict(zip(l_labels, l_values)),
        value=input_value,
        description=description,
    )


def build_widget_slider(input_value, bound_min, bound_max, label, step):
    return widgets.FloatSlider(
        value=input_value,
        min=bound_min,
        max=bound_max,
        description=label,
        step=step,
        continuous_update=False,
        style={"description_width": "initial"},
    )


def build_widget_radio_button(label, options, input_value):
    selected_value = input_value if input_value in options else options[0]
    return widgets.RadioButtons(
        options=options,
        description=label,
        continuous_update=False,
        value=selected_value,
        style={"description_width": "initial"},
    )


# applied widgets

def update_widget_dropdown(w_rate_tenor, rate_type):
    if RateType.SWAP_RATE == rate_type:
        rate_tenor_labels = ["6M", "1Y", "2Y", "3Y", "4Y", "5Y", "10Y", "15Y", "20Y", "25Y", "30Y"]
        rate_tenor_values = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    else:
        rate_tenor_labels = ["1M", "3M", "6M", "1Y"]
        rate_tenor_values = [1.0 / 12.0, 0.25, 0.5, 1.0]
    w_rate_tenor.options = dict(zip(rate_tenor_labels, rate_tenor_values))
    w_rate_tenor.index = 1

def handle_widget_rate_tenor(change, w_rate_tenor):
    rate_type_old = RateType(change['old'])
    rate_type_new = RateType(change['new'])
    rate_type_has_changed = (rate_type_old == RateType.SWAP_RATE) != (rate_type_new == RateType.SWAP_RATE)
    if rate_type_has_changed:
        update_widget_dropdown(w_rate_tenor, rate_type_new)

def w_rate_tenor_dropdown(tenor_swap_rates = False):

    if tenor_swap_rates:
        rate_tenor_labels = ["6M", "1Y", "2Y", "3Y", "4Y", "5Y", "10Y", "15Y", "20Y", "25Y", "30Y"]
        rate_tenor_values = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
        input_value = 1.0
    else:
        rate_tenor_labels = ["1M", "3M", "6M", "1Y"]
        rate_tenor_values = [1.0 / 12.0, 0.25, 0.5, 1.0]
        input_value = 0.25

    return build_widget_dropdown(
        "tenor",
        rate_tenor_labels,
        rate_tenor_values,
        input_value=qttools.closest(rate_tenor_values, input_value),
    )


def w_rate_type_radio_button(input_value=RateType.FORWARD_LOOKING, exclude_swap_rate=False):
    rate_types = [RateType.FORWARD_LOOKING, RateType.BACKWARD_LOOKING, RateType.SWAP_RATE] if not exclude_swap_rate else [RateType.FORWARD_LOOKING, RateType.BACKWARD_LOOKING]
    rate_types_labels = [rate_type.value for rate_type in rate_types]
    return build_widget_radio_button("Rate type:", rate_types_labels, input_value.value)


# endregion

# region PLOT TOOLS


def plot_data(
    df_data,
    label_x,
    label_y,
    label_color,
    label_column=None,
    label_style=None,
    use_plotly=False,
    log_x=False,
    use_markers=False,
    title=None,
):

    if use_plotly:
        fig = px.line(
            data_frame=df_data,
            x=label_x,
            y=label_y,
            color=label_color,
            facet_col=label_column,
            log_x=log_x,
        )
        if use_markers:
            fig.update_traces(mode="lines+markers")
        fig.show()
    else:
        facet_grid = sns.relplot(
            data=df_data,
            x=label_x,
            y=label_y,
            hue=label_color,
            style=label_style,
            markers=use_markers,
            col=label_column,
            kind="line",
        )
        if log_x:
            facet_grid.set(xscale="log")

        facet_grid.fig.subplots_adjust(top=0.9)
        if title is not None:
            facet_grid.fig.suptitle(title)
        plt.show()


# endregion

# endregion


# region GENERIC MODEL ANALYSIS


def build_list_rate_indices(tenor, last_expiry, rate_type, allow_negative_dates=True):

    nb_dates = 250

    key_dates = [-tenor, 0, last_expiry] if rate_type == RateType.BACKWARD_LOOKING else [0, last_expiry]
    start_dates = qttools.multi_linespace(key_dates, nb_dates)

    if not allow_negative_dates:
        start_dates = start_dates[1:]

    return [qtinstruments.build_rate(start_date, tenor, rate_type) for start_date in start_dates]


def underlying_description(tenor, rate_type):
    months = int(tenor / 0.25 * 3 + 0.01)
    return f"{rate_type.value} {tenor}Y" if months >= 12 else f"{rate_type.value}  {months}M"


def plot_model_underlying_outputs(tests, f_compute_xy, label_x, label_y, label_title, label_test_type):

    l_df_implied_vols = []
    for test in tests:
        x, y = f_compute_xy(test["model"], test["underlyings"])
        l_df_implied_vols.append(pd.DataFrame({label_x: x, label_y: y, label_test_type: test["label"]}))

    plot_data(
        pd.concat(l_df_implied_vols, ignore_index=True),
        label_x,
        label_y,
        label_color=label_test_type,
        title=label_title,
    )


# endregion


# region HULL WHITE MODEL ANALYSIS

# applied widgets


def w_hw_mean_rev_slider(mean_rev=0.0):
    return build_widget_slider(mean_rev, 0.0, 2.0, "Mean reversion", 0.01)


# endregion

# region plot HW bond instantaneous volatilities / bond ratio instantaneous volatilities


def build_hw_bond_instantaneous_volatility_tests(
    hw_vol,
    mean_rev_1,
    mean_rev_2,
    expiry_1,
    expiry_2,
):
    hw_model_1 = ModelHullWhite.build_default(mean_rev_1, hw_vol)
    hw_model_2 = ModelHullWhite.build_default(mean_rev_2, hw_vol)

    label_test_1 = f"Bond T={expiry_1} (mean rev = {mean_rev_1})"
    label_test_2 = f"Bond T={expiry_2} (mean rev = {mean_rev_2})"

    test_1 = {
        "model": hw_model_1,
        "underlyings": expiry_1,
        "label": label_test_1,
    }
    test_2 = {
        "model": hw_model_2,
        "underlyings": expiry_2,
        "label": label_test_2,
    }

    return [test_1, test_2]


def plot_interactive_hw_bond_instantaneous_volatility(
    f_compare_hw_bond_volatilities,
    mean_rev_1=0.0,
    mean_rev_2=0.2,
    expiry_1=2.0,
    expiry_2=2.0,
):

    w_bond_1 = widgets.HBox(
        [
            widgets.Label("Bond 1:"),
            w_hw_mean_rev_slider(mean_rev_1),
            build_widget_slider(expiry_1, 0.0, 5.0, "T", 0.1),
        ]
    )
    w_bond_2 = widgets.HBox(
        [
            widgets.Label("Bond 2:"),
            w_hw_mean_rev_slider(mean_rev_2),
            build_widget_slider(expiry_2, 0.0, 5.0, "T", 0.1),
        ]
    )

    w_outputs = widgets.interactive_output(
        f_compare_hw_bond_volatilities,
        {
            "mean_rev_1": w_bond_1.children[1],
            "mean_rev_2": w_bond_2.children[1],
            "expiry_1": w_bond_1.children[2],
            "expiry_2": w_bond_2.children[2],
        },
    )

    display(w_bond_1, w_bond_2, w_outputs)


def analyze_hw_bond_instantaneous_volatility(hw_vol=0.001):
    def f_compute_xy(model, expiry):
        t_expiries = qttools.multi_linespace([0, expiry, expiry + 0.25], 100)
        inst_vol = [model.inst_vol_bond(t, expiry) for t in t_expiries]
        return t_expiries, inst_vol

    def plot_hw_bond_instantaneous_volatility(mean_rev_1, mean_rev_2, expiry_1, expiry_2):
        l_tests = build_hw_bond_instantaneous_volatility_tests(hw_vol, mean_rev_1, mean_rev_2, expiry_1, expiry_2)

        plot_model_underlying_outputs(
            l_tests,
            f_compute_xy,
            label_x=label_dates,
            label_y=label_vol,
            label_title="Bond instantaneous volatility",
            label_test_type="Underlying",
        )

    plot_interactive_hw_bond_instantaneous_volatility(plot_hw_bond_instantaneous_volatility)


def build_hw_bond_ratio_instantaneous_volatility_tests(
    hw_vol,
    mean_rev_1,
    mean_rev_2,
    expiry_1,
    expiry_2,
    delta_1,
    delta_2,
):
    hw_model_1 = ModelHullWhite.build_default(mean_rev_1, hw_vol)
    hw_model_2 = ModelHullWhite.build_default(mean_rev_2, hw_vol)

    label_test_1 = f"P(t,{expiry_1 + delta_1})/P(t,{expiry_1}) (mean rev = {mean_rev_1})"
    label_test_2 = f"P(t,{expiry_2 + delta_2})/P(t,{expiry_2}) (mean rev = {mean_rev_2})"

    test_1 = {
        "model": hw_model_1,
        "underlyings": [expiry_1, delta_1],
        "label": label_test_1,
    }
    test_2 = {
        "model": hw_model_2,
        "underlyings": [expiry_2, delta_2],
        "label": label_test_2,
    }

    return [test_1, test_2]


def plot_interactive_hw_bond_ratio_instantaneous_volatility(
    f_compare_hw_bond_ratio_volatilities,
    mean_rev_1=0.0,
    mean_rev_2=0.2,
    expiry_1=2.0,
    expiry_2=2.0,
    delta_1=0.25,
    delta_2=0.25,
):
    w_bond_ratio_1 = widgets.HBox(
        [
            widgets.Label("Ratio 1:"),
            w_hw_mean_rev_slider(mean_rev_1),
            build_widget_slider(expiry_1, 0.0, 5.0, "T", 0.1),
            build_widget_slider(delta_1, 0.0, 5.0, "Delta", 0.05),
        ]
    )
    w_bond_ratio_2 = widgets.HBox(
        [
            widgets.Label("Ratio 2:"),
            w_hw_mean_rev_slider(mean_rev_2),
            build_widget_slider(expiry_2, 0.0, 5.0, "T", 0.1),
            build_widget_slider(delta_2, 0.0, 5.0, "Delta", 0.05),
        ]
    )

    w_outputs = widgets.interactive_output(
        f_compare_hw_bond_ratio_volatilities,
        {
            "mean_rev_1": w_bond_ratio_1.children[1],
            "mean_rev_2": w_bond_ratio_2.children[1],
            "expiry_1": w_bond_ratio_1.children[2],
            "expiry_2": w_bond_ratio_2.children[2],
            "delta_1": w_bond_ratio_1.children[3],
            "delta_2": w_bond_ratio_2.children[3],
        },
    )

    display(w_bond_ratio_1, w_bond_ratio_2, w_outputs)


def analyze_hw_bond_ratio_instantaneous_volatility(hw_vol=0.001):
    def f_compute_xy(model, bond_ratio):
        expiry, delta = bond_ratio[0], bond_ratio[1]
        t_expiries = qttools.multi_linespace([0, expiry, expiry + delta, expiry + delta + 0.25], 100)
        inst_vol_ratio = [model.inst_vol_bond_ratio(t, expiry, expiry + delta) for t in t_expiries]

        return t_expiries, inst_vol_ratio

    def plot_hw_bond_ratio_instantaneous_volatility(mean_rev_1, mean_rev_2, expiry_1, expiry_2, delta_1, delta_2):
        l_tests = build_hw_bond_ratio_instantaneous_volatility_tests(
            hw_vol, mean_rev_1, mean_rev_2, expiry_1, expiry_2, delta_1, delta_2
        )

        plot_model_underlying_outputs(
            l_tests,
            f_compute_xy,
            label_x="t",
            label_y=label_vol,
            label_title="Bond ratio instantaneous volatility",
            label_test_type="Underlying",
        )

    plot_interactive_hw_bond_ratio_instantaneous_volatility(plot_hw_bond_ratio_instantaneous_volatility)


# endregion

# region HW smile/term structure/caplets prices


def build_hw_option_tests(
    mean_rev_1,
    mean_rev_2,
    tenor_1,
    tenor_2,
    rate_type_1,
    rate_type_2,
    hw_vol,
    max_expiry,
):
    rate_type_1 = RateType(rate_type_1)
    rate_type_2 = RateType(rate_type_2)
    hw_model_1 = ModelHullWhite.build_default(mean_rev_1, hw_vol)
    hw_model_2 = ModelHullWhite.build_default(mean_rev_2, hw_vol)

    allow_negative_dates = RateType.BACKWARD_LOOKING in [rate_type_1, rate_type_2]

    rate_underlyings_1 = build_list_rate_indices(tenor_1, max_expiry, rate_type_1, allow_negative_dates)
    rate_underlyings_2 = build_list_rate_indices(tenor_2, max_expiry, rate_type_2, allow_negative_dates)

    label_test_1 = f"{underlying_description(tenor_1, rate_type_1)} (mean rev {mean_rev_1})"
    label_test_2 = f"{underlying_description(tenor_2, rate_type_2)} (mean rev {mean_rev_2}) "  # extra space to make sure label_test_1 != label_test_2   # noqa: E501

    test_1 = {
        "model": hw_model_1,
        "underlyings": rate_underlyings_1,
        "label": label_test_1,
    }

    test_2 = {
        "model": hw_model_2,
        "underlyings": rate_underlyings_2,
        "label": label_test_2,
    }

    return [test_1, test_2]


def build_hw_option_smile_tests(
    hw_vol,
    mean_rev_1,
    mean_rev_2,
    start_date_1,
    start_date_2,
    underlying_length_1,
    underlying_length_2,
    rate_type_1,
    rate_type_2,
):
    rate_type_1 = RateType(rate_type_1)
    rate_type_2 = RateType(rate_type_2)
    hw_model_1 = ModelHullWhite.build_default(mean_rev_1, hw_vol)
    hw_model_2 = ModelHullWhite.build_default(mean_rev_2, hw_vol)

    rate_underlying_1 = qtinstruments.build_rate(start_date_1, underlying_length_1, rate_type=rate_type_1)
    rate_underlying_2 = qtinstruments.build_rate(start_date_2, underlying_length_2, rate_type=rate_type_2)

    label_test_1 = f"{underlying_description(underlying_length_1, rate_type_1)} (mean rev {mean_rev_1})"
    label_test_2 = f"{underlying_description(underlying_length_2, rate_type_2)} (mean rev {mean_rev_2}) "  # extra space to make sure label_test_1 != label_test_2   # noqa: E501

    test_1 = {
        "model": hw_model_1,
        "underlyings": rate_underlying_1,
        "label": label_test_1,
    }
    test_2 = {
        "model": hw_model_2,
        "underlyings": rate_underlying_2,
        "label": label_test_2,
    }

    return [test_1, test_2]


def plot_interactive_hw_underlyings(
    f_compare_hw_underlyings,
    mean_rev_1=0.0,
    mean_rev_2=0.0,
    rate_type_1=RateType.FORWARD_LOOKING,
    rate_type_2=RateType.FORWARD_LOOKING,
    exclude_swap_rate=False,
):
    w_caplet_1 = widgets.HBox(
        [
            widgets.Label("Caplet 1:"),
            w_hw_mean_rev_slider(mean_rev_1),
            w_rate_type_radio_button(rate_type_1, exclude_swap_rate),
            w_rate_tenor_dropdown(),
        ]
    )
    w_caplet_2 = widgets.HBox(
        [
            widgets.Label("Caplet 2:"),
            w_hw_mean_rev_slider(mean_rev_2),
            w_rate_type_radio_button(rate_type_2, exclude_swap_rate),
            w_rate_tenor_dropdown(),
        ]
    )

    w_outputs = widgets.interactive_output(
        f_compare_hw_underlyings,
        {
            "mean_rev_1": w_caplet_1.children[1],
            "mean_rev_2": w_caplet_2.children[1],
            "rate_type_1": w_caplet_1.children[2],
            "rate_type_2": w_caplet_2.children[2],
            "tenor_1": w_caplet_1.children[3],
            "tenor_2": w_caplet_2.children[3],
        },
    )

    def handle_widget_rate_tenor_1(change):
        return handle_widget_rate_tenor(change, w_caplet_1.children[3])

    def handle_widget_rate_tenor_2(change):
        return handle_widget_rate_tenor(change, w_caplet_2.children[3])

    w_caplet_1.children[2].observe(handle_widget_rate_tenor_1, names='value')
    w_caplet_2.children[2].observe(handle_widget_rate_tenor_2, names='value')

    display(w_caplet_1, w_caplet_2, w_outputs)


def analyze_hw_caplets(hw_vol=0.001, max_expiry=10.0):
    def f_compute_xy(model, underlyings):
        return qtmodel.compute_model_atm_option_values(model, underlyings)

    def plot_hw_atm_option_values(mean_rev_1, mean_rev_2, tenor_1, tenor_2, rate_type_1, rate_type_2):
        l_tests = build_hw_option_tests(
            mean_rev_1,
            mean_rev_2,
            tenor_1,
            tenor_2,
            rate_type_1,
            rate_type_2,
            hw_vol,
            max_expiry,
        )
        plot_model_underlying_outputs(
            l_tests,
            f_compute_xy,
            label_x="underlying start date",
            label_y="option value",
            label_title="ATM option values",
            label_test_type="Underlying",
        )

    plot_interactive_hw_underlyings(
        plot_hw_atm_option_values,
        rate_type_1=RateType.FORWARD_LOOKING,
        rate_type_2=RateType.BACKWARD_LOOKING,
        exclude_swap_rate=True,
    )


def analyze_hw_term_structure(
    hw_vol=0.001,
    max_expiry=10.0,
    mean_rev_1=0.0,
    mean_rev_2=0.2,
    vol_type=VolType.NORMAL,
):
    def f_compute_xy(model, underlyings):
        return qtmodel.compute_model_atm_volatilities(model, underlyings, vol_type)

    def plot_hw_term_structure(mean_rev_1, mean_rev_2, tenor_1, tenor_2, rate_type_1, rate_type_2):
        l_tests = build_hw_option_tests(
            mean_rev_1,
            mean_rev_2,
            tenor_1,
            tenor_2,
            rate_type_1,
            rate_type_2,
            hw_vol,
            max_expiry,
        )
        plot_model_underlying_outputs(
            l_tests,
            f_compute_xy,
            label_x="underlying start date",
            label_y="implied vol",
            label_title="Volatility term structure",
            label_test_type="Underlying",
        )

    plot_interactive_hw_underlyings(
        plot_hw_term_structure,
        mean_rev_1=mean_rev_1,
        mean_rev_2=mean_rev_2,
        rate_type_1=RateType.FORWARD_LOOKING,
        rate_type_2=RateType.FORWARD_LOOKING,
    )


def analyze_hw_smile(hw_vol=0.001, start_date=2.0, vol_type=VolType.NORMAL, nb_std_dev=3):
    def f_compute_xy(model, underlying):
        return qtmodel.compute_model_smile(model, underlying, nb_std_dev, vol_type)

    def plot_hw_smile(mean_rev_1, mean_rev_2, tenor_1, tenor_2, rate_type_1, rate_type_2):
        l_tests = build_hw_option_smile_tests(
            hw_vol,
            mean_rev_1,
            mean_rev_2,
            start_date,
            start_date,
            tenor_1,
            tenor_2,
            rate_type_1,
            rate_type_2,
        )

        plot_model_underlying_outputs(
            l_tests,
            f_compute_xy,
            label_x="strike",
            label_y="implied vol",
            label_title="Volatility smile",
            label_test_type="Underlying",
        )

    plot_interactive_hw_underlyings(plot_hw_smile)


# endregion
