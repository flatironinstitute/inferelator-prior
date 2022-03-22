import numpy as np
import warnings

def assign_times_from_pseudotime(pseudotimes, time_group_labels=None, time_thresholds=None, total_time=None,
                                 time_quantiles=(0.05, 0.95), random_order_pools=None, random_seed=256):
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
    rng = np.random.default_rng(random_seed)
    real_times = np.full_like(pseudotimes, np.nan, dtype=float)

    for group, rt_start, rt_stop in time_thresholds:
        group_idx = time_group_labels == group

        if group_idx.sum() == 0:
            continue

        rt_interval = rt_stop - rt_start

        # Randomly order times if random_order_pools is set and group is in the passed list
        if random_order_pools is not None and any(x == group for x in random_order_pools):
            real_times[group_idx] = rng.uniform(0., 1., group_idx.sum()) * rt_interval + rt_start

        # Otherwise interval normalize
        else:
            try:
                group_pts = _quantile_shift(pseudotimes[group_idx], time_quantiles) * rt_interval + rt_start
            except ValueError:
                group_pts = np.full_like(pseudotimes[group_idx], np.nan)
            real_times[group_idx] = group_pts

    return real_times


def assign_times_from_pseudotime_sliding(pseudotime, time_group_labels, time_order, time_thresholds, window_width=1,
                                         edges=(0.05, 0.95)):
    """
    Assign real times using a sliding window around groups of pseudotimes

    :param pseudotimes: Array of pseudotime values (will be scaled 0-1 if not already)
    :type pseudotimes: np.ndarray
    :param time_group_labels: An array of time group labels to go with time_thresholds
    :type time_group_labels: np.ndarray
    :param time_order: A list of group labels in order temporally
    :type time_order: list
    :param time_thresholds: A list of tuples where each tuple defines an anchor point
        (group_label, realtime_start, realtime_end).
    :type time_thresholds: list(tuples)
    :param window_width: The number of groups to consider on each side of the center group, defaults to 1
    :type window_width: int, optional
    :param edges: The quantiles for the outer edge groups, defaults to (0.05, 0.95)
    :type edges: tuple, optional
    :raises ValueError: Raises ValueError if the windowing is bad or if unknown time labels are provided in time_order
    :return: Real-time values
    :rtype: np.ndarray
    """

    n = len(time_order)
    span = 2 * window_width + 1

    if not isinstance(time_thresholds, dict):
        time_thresholds = {k: (k1, k2) for k, k1, k2 in time_thresholds}

    _unknown_times = [t for t in time_order if t not in time_thresholds]
    if len(_unknown_times) != 0:
            raise ValueError(f"Unable to find times {_unknown_times}")

    if n < span:
        raise ValueError(f"Cannot make windows of size {window_width} from {n} groups")

    time_vector = np.full_like(pseudotime, np.nan)

    for i in range(n):

        ### GET INDICES AROUND EACH GROUP ###
        left_idx, right_idx = i - window_width, i + window_width + 1

        ### MAKE SURE INDICES AREN'T TOO BIG/SMALL ###
        if left_idx < 0:
            left_idx, right_idx = 0, span
        elif right_idx > n - 1:
            left_idx, right_idx = n - span - 1, n - 1

        ### DEFINE WINDOW TIMES ###
        select_times = time_order[left_idx:right_idx]
        center_time = time_order[i]

        left_time = min(x[0] for k, x in time_thresholds.items() for y in select_times if k == y)
        right_time = max(x[1] for k, x in time_thresholds.items() for y in select_times if k == y)
        interval_time = right_time - left_time

        ### GET PT THRESHOLDS OF LEFTMOST AND RIGHTMOST ###
        lq = _finite_quantile(pseudotime[time_group_labels == time_order[left_idx]], edges[0])
        rq = _finite_quantile(pseudotime[time_group_labels == time_order[right_idx - 1]], edges[1])

        ### INDICES FOR PT VALUES OF INTEREST ###
        keep_window = time_group_labels == center_time

        ### CONVERT TO TIMES ###
        window_pts = _quantile_shift(pseudotime[keep_window].copy(),
                                     thresholds=(lq, rq))

        window_pts[(window_pts < 0) | (window_pts > 1)] = np.nan
        window_pts *= interval_time
        window_pts += left_time

        ### ADD TO VECTOR ##
        time_vector[time_group_labels == center_time] = window_pts[time_group_labels[keep_window] == center_time]

    return time_vector


def _quantile_shift(arr, quantiles=None, thresholds=None):
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

    if quantiles is None and thresholds is None:
        return arr.copy()

    if quantiles is not None:
        lq, rq = _finite_quantile(arr, quantiles)
    elif thresholds is not None:
        lq, rq = thresholds

    if not np.isfinite(lq) or not np.isfinite(rq):
        raise ValueError(f"Unable to anchor values {lq} and {rq}")

    if lq != rq:
        arr = (arr - lq) / (rq - lq)
    else:
        arr = arr.copy()

    return arr


def _finite_quantile(arr, quantile):

    if quantile is None:
        return None

    _is_finite = np.isfinite(arr)

    try:
        n = len(quantile)
    except TypeError:
        n = 1

    if np.sum(_is_finite) < n:
        raise ValueError(f"Cannot find {n} quantiles from {np.sum(_is_finite)} values")

    return np.nanquantile(arr[_is_finite], quantile)


def _interval_normalize(arr):
    """
    Normalize to 0-1. Ignore NaN and Inf.

    :param arr: Data to normalize
    :type arr: np.ndarray
    :return: Normalized data
    :rtype: no.ndarray
    """

    _is_finite = np.isfinite(arr)

    if np.sum(_is_finite) == 0:
        return np.zeros_like(arr)

    s_min, s_max = np.nanmin(arr[_is_finite]), np.nanmax(arr[_is_finite])

    if s_min != s_max:
        scaled_arr = (arr - s_min) / (s_max - s_min)
    else:
        scaled_arr = np.zeros_like(arr)

    return scaled_arr
