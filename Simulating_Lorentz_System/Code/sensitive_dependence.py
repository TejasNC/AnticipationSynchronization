import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from  matplotlib.animation import FuncAnimation

# Params
sigma = 10
beta = 8/3
r = 28

c_plus = [np.sqrt(beta * (r-1)),np.sqrt(beta * (r-1)),r-1]
c_minus = [-1* np.sqrt(beta * (r-1)), -1 * np.sqrt(beta * (r-1)),r-1]

# Create time domain
t = np.linspace(0,50,2000)

# Define the Loretz System
def lorentz_system(vars, t, sigma, beta, r):
    x = vars[0]
    y = vars[1]
    z = vars[2]
    return np.array([
        sigma * (y-x),
        x * (r - z) -y,
        x*y - beta * z
    ])

# Initial Conditions
r0 = np.array([0,1,0])
r1 = np.array([0,1,0.00001])

# Solution
positions1 = odeint(lorentz_system, r0, t, args=(sigma, beta, r))
positions2 = odeint(lorentz_system, r1, t, args=(sigma, beta, r))

x1 = positions1[:,0]
y1 = positions1[:,1]
z1 = positions1[:,2]

x2 = positions2[:,0]
y2 = positions2[:,1]
z2 = positions2[:,2]

# --------------------------------------------------------------------------------
# Plotting the Phase Space

fig, ax = plt.subplots(subplot_kw={'projection':'3d'})

diag1, = ax.plot(x1, y1, z1, color='blue', label=f'Starting from {r0}')
diag2, = ax.plot(x2, y2, z2, color='red', label=f'Starting from {r1}')

print(f"Simulating Lorentz System (sigma,beta,r) = ({sigma},{beta},{r}) starting from the initial coordinates ({r0})")

LAG = 20 
# LAG = 100 # Set to 100 for better visualisation of the phase space as a whole

def update(frame):
    lower_lim = max(0,frame-LAG)
    diag1.set_data(x1[lower_lim:frame+1],y1[lower_lim:frame+1])
    diag1.set_3d_properties(z1[lower_lim:frame+1])
    diag2.set_data(x2[lower_lim:frame+1],y2[lower_lim:frame+1])
    diag2.set_3d_properties(z2[lower_lim:frame+1])
    return diag1, diag2

ani = FuncAnimation(fig, update, frames=len(t), interval=1, blit=False)

ax.set_title('Phase Space Demonstrating Sensitive Dependence on Initial Conditions')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
ani.save('../Results/sensitive_dependence.gif', fps=50)
# --------------------------------------------------------------------------------
# Plotting the Time Series

fig, ax = plt.subplots(3,1,figsize=(10,10))

ax[0].plot(t, x1, label='x1', color='blue')
ax[0].plot(t, x2, label='x2', color='red')
ax[0].set_title('X vs Time')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('X')
ax[0].legend()

ax[1].plot(t, y1, label='y1', color='blue')
ax[1].plot(t, y2, label='y2', color='red')
ax[1].set_title('Y vs Time')
ax[1].set_xlabel('Time')

ax[1].set_ylabel('Y')
ax[1].legend()

ax[2].plot(t, z1, label='z1', color='blue')
ax[2].plot(t, z2, label='z2', color='red')
ax[2].set_title('Z vs Time')
ax[2].set_xlabel('Time')
ax[2].set_ylabel('Z')
ax[2].legend()


plt.tight_layout()
plt.show()
plt.savefig('../Results/sensitive_dependence_time_series.png')
# --------------------------------------------------------------------------------
