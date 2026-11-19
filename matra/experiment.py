"""Controlled geometry and the archive's scene-dependent depth conversion."""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import platform
from types import SimpleNamespace

import cv2
import numpy as np

from .baseline import _estimate_scale, _to_metric_depth


def scene():
    """Known pinhole geometry; not a rendering claimed to be recorded footage."""
    K = np.array([[800., 0, 320], [0, 800, 240], [0, 0, 1.]])
    u, v = np.meshgrid(np.arange(100, 541, 44), np.arange(100, 381, 40))
    points1 = np.column_stack((u.ravel(), v.ravel())).astype(np.float32)
    z = 4 + (points1[:, 0] - 100) * .005 + (points1[:, 1] - 100) * .003
    world = np.column_stack(((points1[:, 0] - 320) * z / 800, (points1[:, 1] - 240) * z / 800, z)).astype(np.float64)
    translated = world - np.array([.4, 0, 0])
    points2 = (translated[:, :2] / translated[:, 2, None] * 800 + np.array([320, 240])).astype(np.float32)
    y_grid, x_grid = np.mgrid[:480, :640]
    depth = (4 + (x_grid - 100) * .005 + (y_grid - 100) * .003).astype(np.float32)
    return K, world, points1, points2, depth


def experiments():
    cv2.setRNGSeed(0)
    K, world, points1, points2, depth = scene()
    mask = np.ones((len(points1), 1), dtype=np.uint8)
    R = np.eye(3)
    direction = np.array([[-1.], [0], [0]])
    scale_rows = []
    for factor in (.5, 1., 2.):
        state = SimpleNamespace(K=K, scale_history=deque(maxlen=30))
        estimate = _estimate_scale(state, points1, points2, mask, depth * factor, None, R, direction)
        scale_rows.append({
            'input_depth_factor': factor,
            'known_scene_baseline_m': .4,
            'estimated_translation_scale': float(estimate),
            'error_if_mistaken_for_original_scene_metres': float(estimate - .4),
        })
    unscaled_state = SimpleNamespace(K=K, scale_history=deque(maxlen=30))
    no_depth = _estimate_scale(unscaled_state, points1, points2, mask, None, None, R, direction)

    # Scaling the entire world AND the camera baseline leaves projection unchanged.
    world_b = world * 2
    moved_b = world_b - np.array([.8, 0, 0])
    projected_b = moved_b[:, :2] / moved_b[:, 2, None] * 800 + np.array([320, 240])
    moved_a = world - np.array([.4, 0, 0])
    projected_a = moved_a[:, :2] / moved_a[:, 2, None] * 800 + np.array([320, 240])
    projection_delta = float(np.max(np.abs(projected_a - projected_b)))

    raw = np.linspace(4, 8, 100, dtype=np.float32).reshape(10, 10)
    changed = raw.copy()
    changed[-1, -1] = 80
    config = SimpleNamespace(min_depth=.1, max_depth=20., scale_factor=1.)
    converter = SimpleNamespace(config=config)
    normalized_a = _to_metric_depth(converter, raw)
    normalized_b = _to_metric_depth(converter, changed)
    pixel = (5, 5)
    normalization = {
        'held_pixel_row_column': list(pixel),
        'unchanged_raw_value': float(raw[pixel]),
        'mapped_before': float(normalized_a[pixel]),
        'mapped_after_one_other_pixel_becomes_80': float(normalized_b[pixel]),
        'configured_output_range': [.1, 20.],
        'kind': 'Direct call to the archived conversion; no neural inference or physical observation',
    }
    return {
        'software': {'python': platform.python_version(), 'numpy': np.__version__, 'opencv': cv2.__version__},
        'input': f'{len(points1)} exact synthetic correspondences, known intrinsics, identity rotation, 0.4 m baseline',
        'scale_recovery': scale_rows,
        'without_depth_or_history': float(no_depth),
        'two_worlds': {'baseline_a_m': .4, 'baseline_b_m': .8, 'max_projection_difference_px': projection_delta},
        'normalization': normalization,
        'conclusion': 'The scale estimator inherits the units of its depth input. Frame-wise min-max remapping does not establish those units.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/controlled.json'))
    args = parser.parse_args()
    result = experiments()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
