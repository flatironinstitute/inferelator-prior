import unittest

import numpy as np
import numpy.testing as npt

from inferelator_prior.velocity.programs import (information_distance, _mutual_information,
                                                 _shannon_entropy, _make_array_discrete)

N = 1000
BINS = 10

EXPRESSION = np.random.default_rng(222222).random((N, 5))
EXPRESSION[:, 0:3] = (100 * EXPRESSION[:, 0:3]).astype(int)
EXPRESSION[:, 3] = 0
EXPRESSION[:, 4] = np.arange(N)

class TestVelocity(unittest.TestCase):

    def test_binning(self):

        expr = _make_array_discrete(EXPRESSION, BINS)

        npt.assert_equal(expr[:, 3], np.zeros_like(expr[:, 3]))
        npt.assert_equal(expr[:, 4], np.repeat(np.arange(BINS), N / BINS))

        self.assertEqual(expr[:, 0].min(), 0)
        self.assertEqual(expr[:, 0].max(), 9)

    def test_entropy(self):

        expr = _make_array_discrete(EXPRESSION, BINS)
        entropy = _shannon_entropy(expr, 10, logtype=np.log2)

        print(entropy)
        self.assertTrue(np.all(entropy >= 0))
        npt.assert_almost_equal(entropy[4], np.log2(BINS))
        npt.assert_almost_equal(entropy[3], 0.)

        entropy = _shannon_entropy(expr, 10, logtype=np.log)

        self.assertTrue(np.all(entropy >= 0))
        npt.assert_almost_equal(entropy[3], 0.)
        npt.assert_almost_equal(entropy[4], np.log(BINS))

    def test_mutual_info(self):

        expr = _make_array_discrete(EXPRESSION, BINS)

        entropy = _shannon_entropy(expr, 10, logtype=np.log2)
        mi = _mutual_information(expr, 10, logtype=np.log2)

        self.assertTrue(np.all(mi >= 0))
        npt.assert_array_equal(mi[:, 3], np.zeros_like(mi[:, 3]))
        npt.assert_array_equal(mi[3, :], np.zeros_like(mi[3, :]))
        npt.assert_array_almost_equal(np.diagonal(mi), entropy)

    def test_info_distance(self):

        expr = _make_array_discrete(EXPRESSION, BINS)

        entropy = _shannon_entropy(expr, 10, logtype=np.log2)
        mi = _mutual_information(expr, 10, logtype=np.log2)

        with np.errstate(divide='ignore', invalid='ignore'):
            calc_dist = 1 - mi / (entropy[:, None] + entropy[None, :] - mi)
            calc_dist[np.isnan(calc_dist)] = 0.

        i_dist, mi_from_dist = information_distance(expr, BINS, logtype=np.log2, return_information=True)

        self.assertTrue(np.all(i_dist >= 0))
        npt.assert_almost_equal(mi, mi_from_dist)
        npt.assert_almost_equal(i_dist, calc_dist)
        npt.assert_array_almost_equal(np.diagonal(i_dist), np.zeros_like(np.diagonal(i_dist)))
