import numpy as np

from training.prepare_distillation import mix_targets


def test_mix_targets_averages_both_directions_before_kd():
    teacher = np.array([0.0, 1.0, 1 / 3])
    forward = np.array([0.0, 2.0, -2.0])
    reverse = np.array([0.0, -2.0, 2.0])

    mixed = mix_targets(teacher, forward, reverse, teacher_weight=0.5)

    np.testing.assert_allclose(mixed, [0.25, 0.75, 5 / 12], rtol=0, atol=1e-12)


def test_mix_targets_rejects_invalid_weight():
    with np.testing.assert_raises(ValueError):
        mix_targets(np.array([0.0]), np.array([0.0]), np.array([0.0]), teacher_weight=1.01)
