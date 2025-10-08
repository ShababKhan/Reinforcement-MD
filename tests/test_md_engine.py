"""Unit tests for the core md_engine module."""
import unittest
from pathlib import Path
import numpy as np
import sys

# Import the classes to be tested and required components
try:
    from src.md_engine import SystemState, AbstractIntegrator, VelocityVerlet, MDSimulator
    from src.force_provider import MockForceProvider 
except ImportError:
    current_dir = Path(__file__).parent
    sys.path.append(str(current_dir.parent / "src"))
    from md_engine import SystemState, AbstractIntegrator, VelocityVerlet, MDSimulator
    from force_provider import MockForceProvider


class TestMDEngine(unittest.TestCase):
    """
    Tests the structural integrity of SystemState, the AbstractIntegrator interface,
    the numerical accuracy of VelocityVerlet, and the logic of MDSimulator.
    """
    
    # --- US 2.1 Tests (SystemState and AbstractIntegrator) ---

    def setUp(self):
        """Set up standard parameters."""
        self.num_atoms = 10

    def test_system_state_initialization(self):
        """Verify that SystemState initializes arrays with correct shape and type."""
        state = SystemState(self.num_atoms)
        
        # Check basic properties
        self.assertEqual(state.num_atoms, self.num_atoms)
        
        # Check Position (r) dimensions and type (N x 3, float64)
        self.assertEqual(state.r.shape, (self.num_atoms, 3))
        self.assertEqual(state.r.dtype, np.float64)
        
        # Check Mass dimensions and initial values
        self.assertEqual(state.masses.shape, (self.num_atoms,))
        np.testing.assert_array_equal(state.masses, np.ones(self.num_atoms))

    def test_integrator_abstraction(self):
        """Verify that the AbstractIntegrator step method raises NotImplementedError."""
        integrator = AbstractIntegrator(time_step=0.002)
        state = SystemState(5)
        
        # Should raise an error if the abstract step method is called
        with self.assertRaises(NotImplementedError):
            integrator.step(state)

    # --- US 2.2 Tests (VelocityVerlet Integrator) ---

    def test_velocity_verlet_constant_force(self):
        """
        [SCIENTIFIC BENCHMARK] Verify VelocityVerlet integration for a simple case: constant force.
        Analytic Solution for constant acceleration (since force is constant and mass is 1.0):
        r(t+dt) = r(t) + v(t)dt + 0.5*a*dt^2
        v(t+dt) = v(t) + a*dt
        """
        dt = 1.0  # Time step
        num_atoms = 1
        
        # 1. Initialize System State
        state = SystemState(num_atoms)
        state.r = np.array([[[1.0, 2.0, 3.0]]])  # Initial position
        state.v = np.array([[[0.5, 0.0, 0.0]]])  # Initial velocity
        integrator = VelocityVerlet(time_step=dt)
        
        # 2. Apply a constant force along the X-axis (F=1.0)
        F_const = np.array([[[1.0, 0.0, 0.0]]]) 
        f_new = F_const # Since force is constant, f(t+dt) = f(t)
        
        # 3. Calculate expected values analytically for t + dt
        a_const = F_const / state.masses[:, None] # Acceleration is [1.0, 0.0, 0.0]
        
        r_expected = state.r + state.v * dt + 0.5 * a_const * dt**2
        # Expected value: [2.0, 2.0, 3.0]
        v_expected = state.v + a_const * dt
        # Expected value: [1.5, 0.0, 0.0]

        # Perform the integration step
        new_state = integrator.step(state, f_new)

        # 4. Verify results scientifically (high precision required)
        np.testing.assert_array_almost_equal(new_state.r, r_expected, decimal=8)
        np.testing.assert_array_almost_equal(new_state.v, v_expected, decimal=8)
        np.testing.assert_array_almost_equal(new_state.f, F_const)
        

    # --- US 2.4 Tests (MDSimulator) ---
    
    def test_simulator_initialization(self):
        """Verify that the simulator initializes all components and calculates the initial force."""
        num_atoms = 3
        state = SystemState(num_atoms)
        integrator = VelocityVerlet(time_step=0.1)
        force_provider = MockForceProvider(num_atoms, mock_force=5.0)

        simulator = MDSimulator(state, integrator, force_provider)
        
        # Verify initial force calculation occurred (Mock force is 5.0 on [0, 0])
        self.assertAlmostEqual(simulator.state.f[0, 0], 5.0)
        self.assertAlmostEqual(simulator.time, 0.0)

    def test_simulator_run_steps_scientific_benchmark(self):
        """
        [SCIENTIFIC BENCHMARK] Verify the simulator runs correctly and matches analytical
        kinematics for a constant force.
        """
        num_atoms = 1
        num_steps = 10
        save_interval = 3
        dt = 1.0

        state = SystemState(num_atoms)
        # Set a clear initial position and velocity to 0
        state.r = np.array([[[0.0, 0.0, 0.0]]])
        state.v = np.array([[[0.0, 0.0, 0.0]]])
        
        integrator = VelocityVerlet(time_step=dt)
        # Mock force is 1.0 along X (f=1.0, m=1.0, a=1.0)
        force_provider = MockForceProvider(num_atoms, mock_force=1.0)
        
        simulator = MDSimulator(state, integrator, force_provider)
        
        trajectory_r, trajectory_u = simulator.run_steps(num_steps, save_interval)

        # Check total number of saves (saves at steps 3, 6, 9)
        self.assertEqual(len(trajectory_r), 3)
        self.assertEqual(len(trajectory_u), 3)
        
        # Check final time
        self.assertAlmostEqual(simulator.time, num_steps * dt)
        
        # Scientific Check: Position after 't' steps with constant acceleration a=1.0 (since F=1, m=1) 
        # r(t) = 0.5 * a * t^2. Position is saved at step t=9.
        t_final_save = 9.0
        expected_r_final = np.array([[[0.5 * 1.0 * t_final_save**2, 0.0, 0.0]]])
        # Expected value: 40.5
        
        actual_r_final = trajectory_r[-1]
        
        np.testing.assert_array_almost_equal(actual_r_final, expected_r_final, decimal=5)
