from collections import deque
from types import SimpleNamespace
import unittest

import numpy as np

from matra.baseline import _estimate_scale, _to_metric_depth
from matra.experiment import experiments, scene


class ScaleBoundaries(unittest.TestCase):
    def test_known_metric_depth_recovers_the_synthetic_camera_baseline(self):
        K, _, points1, points2, depth = scene()
        state = SimpleNamespace(K=K, scale_history=deque(maxlen=30))
        scale = _estimate_scale(state, points1, points2, np.ones((len(points1), 1), np.uint8), depth, None, np.eye(3), np.array([[-1.], [0], [0]]))
        self.assertAlmostEqual(float(scale), .4, places=5)

    def test_world_and_camera_scaling_leave_the_image_ambiguous(self):
        result = experiments()
        self.assertLess(result['two_worlds']['max_projection_difference_px'], 1e-10)
        half, metric, double = result['scale_recovery']
        self.assertAlmostEqual(half['estimated_translation_scale'] / metric['estimated_translation_scale'], .5, places=5)
        self.assertAlmostEqual(double['estimated_translation_scale'] / metric['estimated_translation_scale'], 2., places=5)

    def test_unrelated_extreme_changes_the_unchanged_pixel_conversion(self):
        raw = np.linspace(4, 8, 100, dtype=np.float32).reshape(10, 10)
        other = raw.copy()
        other[-1, -1] = 80
        converter = SimpleNamespace(config=SimpleNamespace(min_depth=.1, max_depth=20., scale_factor=1.))
        before = _to_metric_depth(converter, raw)
        after = _to_metric_depth(converter, other)
        self.assertEqual(raw[5, 5], other[5, 5])
        self.assertGreater(float(before[5, 5] / after[5, 5]), 10.)


if __name__ == '__main__':
    unittest.main()
