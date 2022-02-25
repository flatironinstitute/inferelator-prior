import unittest

import numpy as np
import numpy.testing as npt

from inferelator_prior.velocity.calc import calc_velocity
from inferelator_prior.velocity.decay import calc_decay, calc_decay_sliding_windows

N = 10

V_SLOPES = np.array([1, -1, 0, 1])
V_EXPRESSION = np.random.default_rng(222222).random((N, 4))
VELOCITY = np.multiply(V_EXPRESSION, V_SLOPES[None, :])
V_EXPRESSION[:, 3] = 0

T_SLOPES = np.array([0.5, -0.5, 0, 1])

TIME = np.arange(N)
T_EXPRESSION = np.multiply(TIME[:, None], T_SLOPES[None, :])
T_EXPRESSION = np.add(T_EXPRESSION, np.array([0, 10, 1, 5])[None, :])


class TestVelocity(unittest.TestCase):

    def test_calc_velocity(self):

        correct_velo = np.tile(T_SLOPES[:, None], N).T
        velo = calc_velocity(T_EXPRESSION, TIME, np.ones((N, N)), N,
                             wrap_time=None)

        npt.assert_array_almost_equal(correct_velo, velo)

        velo_wrap = calc_velocity(T_EXPRESSION, TIME, np.ones((N, N)), N,
                                  wrap_time=0)

        npt.assert_array_almost_equal(correct_velo, velo_wrap)

    def test_calc_decay_no_alpha(self):

        decays, decay_se, alpha_est = calc_decay(V_EXPRESSION, VELOCITY,
                                                 decay_quantiles=(0, 1),
                                                 include_alpha=False)

        correct_ses = np.zeros_like(decay_se)
        correct_ses[0] = V_SLOPES[0] / N

        correct_decays = np.maximum(V_SLOPES * -1, np.zeros_like(V_SLOPES))

        self.assertIsNone(alpha_est)
        npt.assert_array_almost_equal(decays, correct_decays)
        npt.assert_array_almost_equal(decay_se, correct_ses)

    def test_calc_decay_alpha(self):

        velo = np.vstack((VELOCITY, np.array([1, 0, 0, 0])))
        expr = np.vstack((V_EXPRESSION, np.array([1, 0, 0, 0])))

        decays, decay_se, alpha_est = calc_decay(expr, velo,
                                                 decay_quantiles=(0, 1),
                                                 include_alpha=True,
                                                 alpha_quantile=1.0)

        correct_alpha = np.maximum(np.max(velo, axis=0), 0)

        correct_ses = np.zeros_like(decay_se)

        correct_decays = np.array([0.323975, 1.,  0., 0.])
        correct_ses = np.array([0.2287143, 0., 0., 0.])

        npt.assert_array_almost_equal(alpha_est, correct_alpha)
        npt.assert_array_almost_equal(decays, correct_decays)
        npt.assert_array_almost_equal(decay_se, correct_ses)

    def test_calc_decay_window(self):

        d, s, a, c = calc_decay_sliding_windows(V_EXPRESSION, VELOCITY, TIME,
                                                decay_quantiles=(0, 1),
                                                include_alpha=False,
                                                n_windows=5, add_pseudocount=True)

        self.assertEqual(len(d), 5)
        self.assertEqual(len(d[0]), 4)

        correct_decays = np.maximum(V_SLOPES * -1, np.zeros_like(V_SLOPES))

        for d_win in d:
            npt.assert_array_almost_equal(d_win, correct_decays)
