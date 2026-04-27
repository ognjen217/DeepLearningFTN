import numpy as np

from stable_icnn_physics import DampedPendulum, MassSpringDamper


def test_oscillator_rhs_shape_and_known_behavior():
    system = MassSpringDamper(mass=2.0, damping=0.5, stiffness=4.0)
    x = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    y = system.rhs(x)
    assert y.shape == x.shape
    np.testing.assert_allclose(y[0], [0.0, -2.0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(y[1], [2.0, -0.5], rtol=1e-6, atol=1e-6)


def test_pendulum_rhs_is_finite_for_sampled_states():
    system = DampedPendulum(n_links=1)
    x = system.sample_states(32, seed=123)
    y = system.rhs(x)
    assert y.shape == x.shape
    assert np.isfinite(y).all()

