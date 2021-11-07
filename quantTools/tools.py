import numpy as np
from scipy import interpolate


def to_array(value_or_array):
    return np.array([value_or_array] if type(value_or_array) in [float, np.float64, int] else value_or_array)


def closest(input_list, target_value):
    list_as_array = np.asarray(input_list)
    idx = (np.abs(list_as_array - target_value)).argmin()
    return input_list[idx]


def get_curve_data_with_extrapolation_pillars(l_x, l_y, extrapolation_delta=1.0, add_zero=True):
    l_x, l_y = to_array(l_x), to_array(l_y)
    curve_nb = min(len(l_x), len(l_y))
    l_x_extrapolation = list(l_x[:curve_nb])
    l_y_extrapolation = list(l_y[:curve_nb])

    # Add a pillar at the beginning for flat extrapolation
    if add_zero and l_x_extrapolation[0] > 0.0:
        l_x_extrapolation = [0] + l_x_extrapolation
        l_y_extrapolation = [l_y_extrapolation[0]] + l_y_extrapolation
    # Add a pillar at the end for flat extrapolation
    l_x_extrapolation = l_x_extrapolation + [l_x_extrapolation[-1] + extrapolation_delta]
    l_y_extrapolation = l_y_extrapolation + [l_y_extrapolation[-1]]

    return l_x_extrapolation, l_y_extrapolation


def to_squared_curve(curve):
    return interpolate.interp1d(curve.x, curve.y ** 2, kind=curve._kind, fill_value=curve.fill_value)


# Operations on lists


def get_sorted_list(l_x, x_inf, x_sup):
    all_elements = np.unique(np.concatenate(([x_inf, x_sup], l_x)))
    return all_elements[np.logical_and(x_inf <= all_elements, all_elements <= x_sup)]


def multi_linespace(l_x, approx_point_N):
    l_ordered_x = np.copy(l_x)
    while len(l_ordered_x) < 2:
        l_ordered_x.append(0.0)
    l_ordered_x = np.sort(l_ordered_x)

    interval = (l_ordered_x[-1] - l_ordered_x[0]) / approx_point_N
    l_sample_N = [
        max(
            int((l_ordered_x[i + 1] - l_ordered_x[i] + interval * 0.001) / interval) + 1,
            2,
        )
        for i in range(len(l_ordered_x) - 1)
    ]

    l_sample_x = [np.linspace(l_ordered_x[0], l_ordered_x[1], l_sample_N[0])]
    for i in range(1, len(l_ordered_x) - 1):
        l_sample_x.append(np.linspace(l_ordered_x[i], l_ordered_x[i + 1], l_sample_N[i])[1:])

    return np.concatenate(l_sample_x)
