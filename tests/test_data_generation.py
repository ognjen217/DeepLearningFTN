import numpy as np

from stable_icnn_physics.data import (
    load_trajectory_dataset,
    save_trajectory_dataset,
    simulate_trajectories,
    trajectory_to_derivative_dataset,
    trajectory_to_transition_dataset,
)
from stable_icnn_physics.systems import make_system


def test_trajectory_simulation_and_flattening_shapes():
    system = make_system("damped_pendulum_1")
    trajectories, derivatives = simulate_trajectories(system, n_trajectories=3, steps=7, dt=0.02, seed=11)

    assert trajectories.shape == (3, 8, 2)
    assert derivatives.shape == (3, 8, 2)
    assert np.isfinite(trajectories).all()
    assert np.isfinite(derivatives).all()

    x_deriv, y_deriv = trajectory_to_derivative_dataset(trajectories, derivatives)
    assert x_deriv.shape == y_deriv.shape == (24, 2)

    x_trans, y_trans = trajectory_to_transition_dataset(trajectories)
    assert x_trans.shape == y_trans.shape == (21, 2)


def test_save_and_load_trajectory_dataset_metadata(tmp_path):
    system = make_system("mass_spring_damper")
    trajectories, derivatives = simulate_trajectories(system, n_trajectories=2, steps=4, dt=0.05)
    path = tmp_path / "dataset.npz"
    save_trajectory_dataset(
        path,
        trajectories,
        derivatives,
        metadata={
            "system": system.metadata(),
            "system_name": system.name,
            "state_dim": system.state_dim,
            "dataset_type": "derivative",
            "generated_from": "trajectories",
        },
    )

    loaded = load_trajectory_dataset(path)
    assert loaded["x"].shape == (10, 2)
    assert loaded["y"].shape == (10, 2)
    assert loaded["trajectories"].shape == (2, 5, 2)
    assert loaded["metadata"]["system_name"] == "mass_spring_damper"
