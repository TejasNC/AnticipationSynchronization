# Imports
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from  matplotlib.animation import FuncAnimation

# Params
sigma = 10
beta = 8/3
r_values = [0.5,10,28]

for r in r_values[:2]:

    # Create time domain
    t = np.linspace(0,50,1000)

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

    # Solution
    positions = odeint(lorentz_system, r0, t, args=(sigma, beta, r))
    x = positions[:,0]
    y = positions[:,1]
    z = positions[:,2]

    # --------------------------------------------------------------------------------
    # Plotting the Phase Space
    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})


    diag, = ax.plot(x, y, z)

    print(f"Simulating Lorentz System (sigma,beta,r) = ({sigma},{beta},{r}) starting from the initial coordinates ({r0})")

    # update function for the animation (older values will get erased)
    def update(frame):

        lower_lim = max(0,frame-200) 

        x_curr = x[lower_lim:frame+1]
        y_curr = y[lower_lim:frame+1]
        z_curr = z[lower_lim:frame+1]

        diag.set_data(x_curr, y_curr)
        diag.set_3d_properties(z_curr)

        return diag,

    animation = FuncAnimation(fig, update, frames=len(t), interval = 1, blit=True)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    animation.save(f'loretz_system_animation_r={r}.gif', fps = 60) 

    # --------------------------------------------------------------------------------
    # Plotting the individual vars against time
    fig, axs = plt.subplots(2,2,figsize=(12,8))

    axs[0,0].plot(t,x,color='red',label='x(t) vs t')
    axs[0,0].set_title('x(t) vs t')
    axs[0,0].set_xlabel('t')
    axs[0,0].set_ylabel('x(t)')
    axs[0,0].legend()

    axs[0,1].plot(t,y,color='blue',label='y(t) vs t')
    axs[0,1].set_title('y(t) vs t')
    axs[0,1].set_xlabel('t')
    axs[0,1].set_ylabel('y(t)')
    axs[0,1].legend()   

    axs[1,0].plot(t,z,color='green',label='z(t) vs t')
    axs[1,0].set_title('z(t) vs t')
    axs[1,0].set_xlabel('t')
    axs[1,0].set_ylabel('z(t)')
    axs[1,0].legend()

    axs[1,1].plot(x,z,color='purple',label='z(t) vs x(t)')
    axs[1,1].set_title('z(t) vs x(t)')
    axs[1,1].set_xlabel('x(t)')
    axs[1,1].set_ylabel('z(t)')
    axs[1,1].legend()

    # plt.show()
    # plt.savefig(f'loretz_system_r={r}.png')

    # --------------------------------------------------------------------------------`

r = r_values[2]

c_plus = [np.sqrt(beta * (r-1)),np.sqrt(beta * (r-1)),r-1]
c_minus = [-1* np.sqrt(beta * (r-1)), -1 * np.sqrt(beta * (r-1)),r-1]

# Create time domain
t = np.linspace(0,50,1000)

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

# Solution
positions = odeint(lorentz_system, r0, t, args=(sigma, beta, r))
x = positions[:,0]
y = positions[:,1]
z = positions[:,2]

# --------------------------------------------------------------------------------
# Plotting the Phase Space
fig, ax = plt.subplots(subplot_kw={'projection':'3d'})


diag, = ax.plot(x, y, z)

print(f"Simulating Lorentz System (sigma,beta,r) = ({sigma},{beta},{r}) starting from the initial coordinates ({r0})")

# Plotiing c+ and c-

# Function for plotting bulges
def plot_bulge(ax, center, radius=0.5, color='blue', alpha=0.2):
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha)

points = np.array([c_plus,c_minus])
colors = ['red','red']

# Plot the points
for i, point in enumerate(points):
    ax.scatter(point[0], point[1], point[2], color=colors[i], s=100)

# Plot the bulges
for point in points:
    plot_bulge(ax, center=point, radius=0.5, color='blue', alpha=0.3)

# update function for the animation (older values will get erased)
def update(frame):

    lower_lim = max(0,frame-200) 

    x_curr = x[lower_lim:frame+1]
    y_curr = y[lower_lim:frame+1]
    z_curr = z[lower_lim:frame+1]

    diag.set_data(x_curr, y_curr)
    diag.set_3d_properties(z_curr)

    return diag,

animation = FuncAnimation(fig, update, frames=len(t), interval= 1, blit=True)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
animation.save(f'loretz_system_animation_r={r}.gif', fps = 60) 

# --------------------------------------------------------------------------------
# Plotting the individual vars against time
fig, axs = plt.subplots(2,2,figsize=(12,8))

axs[0,0].plot(t,x,color='red',label='x(t) vs t')
axs[0,0].set_title('x(t) vs t')
axs[0,0].set_xlabel('t')
axs[0,0].set_ylabel('x(t)')
axs[0,0].legend()

axs[0,1].plot(t,y,color='blue',label='y(t) vs t')
axs[0,1].set_title('y(t) vs t')
axs[0,1].set_xlabel('t')
axs[0,1].set_ylabel('y(t)')
axs[0,1].legend()   

axs[1,0].plot(t,z,color='green',label='z(t) vs t')
axs[1,0].set_title('z(t) vs t')
axs[1,0].set_xlabel('t')
axs[1,0].set_ylabel('z(t)')
axs[1,0].legend()

axs[1,1].plot(x,z,color='purple',label='z(t) vs x(t)')
axs[1,1].set_title('z(t) vs x(t)')
axs[1,1].set_xlabel('x(t)')
axs[1,1].set_ylabel('z(t)')
axs[1,1].legend()

# plt.show()
# plt.savefig(f'loretz_system_r={r}.png')
