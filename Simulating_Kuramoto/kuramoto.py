import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tqdm
# import pillowrtiter as pillow

class KuramotoModel:
    """
    A class to simulate the Kuramoto model of coupled oscillators.
    Attributes
    ----------
    N : int
        Number of oscillators (default is 2).
    K : float
        Coupling constant (default is 1).
    T : int
        Total number of time steps for the simulation (default is 1000).
    dt : float
        Time step size (default is 0.01).
    omega : numpy.ndarray
        Natural frequencies of the oscillators, drawn from a normal distribution.
    theta : numpy.ndarray
        Initial phases of the oscillators, drawn from a uniform distribution.
    theta_t : numpy.ndarray
        Array to store the phases of the oscillators at each time step.
    R : numpy.ndarray
        Order parameter array to measure the synchronization of the oscillators.
    
    Methods
    -------
    run_simulation():
        Runs the Kuramoto model simulation.
    plot_phases():
        Plots the phases of the oscillators over time.
    plot_order_parameter():
        Plots the order parameter over time to show the level of synchronization.
    animate():
        Animates the phases of the oscillators on a polar plot.
    find_critical_K():
        Returns the critical coupling constant for synchronization
    find_terminal_order_parameter():
        Returns the average order parameter over the last 100 time steps.
    """
    def __init__(self, N=2, K=1, T=10000, dt=0.1):
        self.N = N
        self.K = K
        self.T = T
        self.dt = dt
        self.omega = np.random.normal(0, 1, N) # Gaussian distribution of natural frequencies
        self.theta = np.random.uniform(0, 2*np.pi, N) # Uniform distribution of initial phases
        self.theta_t = np.zeros((T, N)) 
        self.theta_t[0] = self.theta 
        self.R = np.zeros(T)

    def run_simulation(self):
        for t in tqdm.tqdm(range(1, self.T)):
            for i in range(self.N):
                self.theta_t[t, i] = (self.theta_t[t-1, i] + self.dt * (
                    self.omega[i] + self.K/self.N * np.sum(np.sin(self.theta_t[t-1] - self.theta_t[t-1, i]))
                )) % (2*np.pi)
        self.R = np.abs(np.sum(np.exp   (1j * self.theta_t), axis=1)) / self.N

    def plot_phases(self):
        plt.plot(self.theta_t)
        plt.xlabel("Time")
        plt.ylabel("Phase")
        plt.show()

    def plot_order_parameter(self):
        plt.plot(self.R)
        plt.xlabel("Time")
        plt.ylabel("Order Parameter")
        plt.show()

    def animate(self):
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        diag, = ax.plot([], [], 'o', markersize=5)
        time_text = ax.text(0.05, 1.05, '', transform=ax.transAxes)

        def init():
            ax.set_xlim(0, 2*np.pi)
            ax.set_ylim(0, 1)
            diag.set_data([], [])
            time_text.set_text('')
            return diag, time_text

        def update(frame):
            diag.set_data(self.theta_t[frame], np.ones(self.N))  # Unit radius
            time_text.set_text(f'Timestep: {frame}')  # Update timestep text
            return diag, time_text

        ani = FuncAnimation(fig, update, frames=range(self.T), init_func=init, interval=1, blit=False)

        print("Animating...")
        plt.show()

    def find_critical_K(self):
        return 2 * np.sqrt(2/np.pi) 
    
    def find_terminal_order_parameter(self):
        return np.mean(self.R[-100:])
    