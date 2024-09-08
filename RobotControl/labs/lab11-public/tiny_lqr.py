import numpy as np
import control as ctrl
import time

# Define system matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5], [6]])

# Define cost matrices
Q = np.array([[1, 0], [0, 1]])
R = np.array([[1]])

# Compute LQR controller gain
K, S, E = ctrl.lqr(A, B, Q, R)

# K is your controller gain
print("Controller gain K:", K)


class System:
    def __init__(self):
        self.x = np.array([[1], [2]])  # Initial state

    def measure_state(self):
        return self.x

    def apply_control(self, u):
        dt = 0.01
        self.x = self.x + np.dot(A, self.x) * dt + np.dot(B, u) * dt

    def print_state(self):
        print(f"{self.x[0].item():.2f}, {self.x[1].item():.2f}")


system = System()

# for _ in range(1000):
#     # Measure or estimate the current state
#     x = system.measure_state()

#     # Calculate the control input
#     u = -np.dot(K, x)

#     # Apply the control input to the system
#     system.apply_control(u)

#     # Wait for the system to react before next iteration
#     # (This could be a time delay in a real-time system)
#     time.sleep(0.01)

#     # Print the current state
#     system.print_state()

import matplotlib.pyplot as plt

for Q_value in [1, 10, 100]:
    for R_value in [1, 10, 100]:
        Q = np.array([[Q_value, 0], [0, Q_value]])
        R = np.array([[R_value]])

        # Recompute LQR controller gain for new Q and R
        K, S, E = ctrl.lqr(A, B, Q, R)

        # Simulate the system with LQR controller
        states = []
        for _ in range(1000):
            x = system.measure_state()
            u = -np.dot(K, x)
            system.apply_control(u)
            states.append(x.flatten())

            # Wait for the system to react before next iteration
            time.sleep(0.01)

        # Plot state trajectory
        states = np.array(states)
        plt.plot(states[:, 0], states[:, 1], label=f'Q={Q_value}, R={R_value}')

plt.xlabel('State x1')
plt.ylabel('State x2')
plt.title('State Trajectory for Different Q and R Values')
plt.legend()
plt.show()