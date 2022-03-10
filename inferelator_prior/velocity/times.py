import numpy as np
import warnings

def assign_times_from_pseudotime(pseudotimes, time_group_labels=None, time_thresholds=None, total_time=None,
                                 time_quantiles=(0.05, 0.95)):
    """
    Assign real times from a pseudotime axis

    :param pseudotimes: Array of pseudotime values (will be scaled 0-1 if not already)
    :type pseudotimes: np.ndarray
    :param time_group_labels: An array of time group labels to go with time_thresholds
    :type time_group_labels: np.ndarray
    :param time_thresholds: A list of tuples where each tuple defines an anchor point
        (group_label, realtime_start, realtime_end).
        Set this or set total_time, defaults to None
    :type time_thresholds: list(tuple(float, float)), optional
    :param total_time: A total length of time to assign to the entire pseudotime axis.
        Set this or set time_thresholds, defaults to None
    :type total_time: numeric, optional
    :param time_quantiles: Anchor the real-time values at these pseudotime quantiles, defaults to (0.05, 0.95).
        Set to None to disable.
    :type time_quantiles: tuple(float, float), optional
    """

    ### CHECK ARGUMENTS ###
    if total_time is None and time_thresholds is None:
        raise ValueError("total_time or time_threshold must be provided")

    if total_time is not None and time_thresholds is not None:
        raise ValueError("One of total_time or time_threshold must be provided, not both")

    if time_quantiles is not None and len(time_quantiles) != 2:
        raise ValueError("time_quantiles must be None or a tuple of two floats")

    if time_thresholds is not None and time_group_labels is None:
        raise ValueError("time_group_labels must be provided if time_thresholds is set")

    pseudotimes = _interval_normalize(pseudotimes)

    ### DO TOTAL TIME IF THAT'S SET ###    
    if total_time is not None:
        return _quantile_shift(pseudotimes, time_quantiles) * total_time

    ### WARN IF GROUPS AREN'T IN ALIGNMENT ###
    time_groups = set(np.unique(time_group_labels))
    time_threshold_groups = set(x[0] for x in time_thresholds)

    _diff_in_thresholds = time_threshold_groups.difference(time_groups)
    if len(_diff_in_thresholds) > 0:
        warnings.warn(f"Labels {list(_diff_in_thresholds)} in time_threshold are not found in time labels")

    _diff_in_labels = time_groups.difference(time_threshold_groups)
    if len(_diff_in_labels) > 0:
        warnings.warn(f"Labels {list(_diff_in_labels)} in time labels are not found in time_threshold")

    ### DO GROUPWISE TIME ASSIGNMENT ###
    real_times = np.full_like(pseudotimes, np.nan, dtype=float)

    for group, rt_start, rt_stop in time_thresholds:
        group_idx = time_group_labels == group

        if group_idx.sum() == 0:
            continue

        rt_interval = rt_stop - rt_start
        group_pts = _quantile_shift(pseudotimes[group_idx], time_quantiles) * rt_interval + rt_start
        real_times[group_idx] = group_pts 

    return real_times


def _quantile_shift(arr, quantiles):
    """
    Shift values so that they're 0-1 where 0 and 1 are set to quantile values from the original data

    :param arr: Numeric array
    :type arr: np.ndarray
    :param quantiles: Quantiles (None disables)
    :type quantiles: tuple(float, float), None
    :return: Numeric array shifted
    :rtype: np.ndarray
    """

    arr = _interval_normalize(arr)

    if quantiles is None:
        return arr

    lq, rq = np.nanquantile(arr, quantiles)

    if lq != rq:
        arr = (arr - lq) / (rq - lq)

    return arr


def _interval_normalize(arr):

    s_min, s_max = np.nanmin(arr), np.nanmax(arr)

    if s_min != s_max:
        scaled_arr = (arr - s_min) / (s_max - s_min)
    else:
        scaled_arr = np.zeros_like(arr)

    return scaled_arr
