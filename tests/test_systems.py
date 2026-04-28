import numpy as np

from stable_icnn_physics import (
    CustomStateSpaceSystem,
    DampedPendulum,
    MassSpringDamper,
    VanDerPolOscillator,
    make_system,
)


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


def test_van_der_pol_rhs_shape_and_known_behavior():
    system = VanDerPolOscillator(mu=2.0)
    x = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, -1.0]], dtype=np.float32)
    y = system.rhs(x)
    assert y.shape == x.shape
    np.testing.assert_allclose(y[0], [0.0, -1.0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(y[1], [1.0, 2.0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(y[2], [-1.0, 4.0], rtol=1e-6, atol=1e-6)


def test_van_der_pol_sampled_states_are_finite():
    system = VanDerPolOscillator(mu=1.5)
    x = system.sample_states(64, seed=321)
    y = system.rhs(x)
    assert x.shape == y.shape == (64, 2)
    assert np.isfinite(x).all()
    assert np.isfinite(y).all()


def test_factory_parses_n_link_pendulum_name():
    system = make_system("damped_pendulum_4")
    assert system.state_dim == 8
    assert system.name == "damped_pendulum_n4"


def test_pendulum_batch_and_single_rhs_shapes():
    system = make_system("damped_pendulum_4")
    batch = system.sample_states(5, seed=42)
    single = batch[0]
    assert system.rhs(batch).shape == (5, 8)
    assert system.rhs(single).shape == (1, 8)


def test_pendulum_angle_wrapping_and_error():
    system = DampedPendulum(n_links=2)
    x = np.array([[3 * np.pi, -3 * np.pi, 0.0, 0.0]], dtype=np.float32)
    wrapped = system.wrap_state(x)
    assert np.all(wrapped[:, :2] <= np.pi)
    assert np.all(wrapped[:, :2] >= -np.pi)

    true = np.array([[np.pi - 0.1, -np.pi + 0.1, 0.0, 0.0]], dtype=np.float32)
    pred = np.array([[-np.pi + 0.1, np.pi - 0.1, 0.0, 0.0]], dtype=np.float32)
    err = system.state_error(true, pred)
    np.testing.assert_allclose(err, np.array([0.08], dtype=np.float32), rtol=1e-5, atol=1e-5)


def test_custom_state_space_system_shapes_and_wrapping():
    def rhs_fn(x, params):
        return np.stack([x[:, 1], -params["k"] * x[:, 0]], axis=1)

    system = CustomStateSpaceSystem(
        name="custom_oscillator",
        state_dim=2,
        rhs_fn=rhs_fn,
        state_ranges=[(-1.0, 1.0), (-2.0, 2.0)],
        angle_indices=[0],
        params={"k": 2.0},
    )
    x = np.array([3 * np.pi, 1.0], dtype=np.float32)
    assert system.rhs(x).shape == (1, 2)
    wrapped = system.wrap_state(x)
    assert -np.pi <= wrapped[0] <= np.pi
