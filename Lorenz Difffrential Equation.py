# Required libraries
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider

# Lorenz system definition
def lorenz(t, state, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Solver and plot function
def plot_lorenz(sigma=10.0, rho=28.0, beta=8.0/3.0):
    # Initial conditions and time span
    initial_state = [1.0, 1.0, 1.0]
    t_span = (0, 40)
    t_eval = np.linspace(t_span[0], t_span[1], 10000)

    # Solve the system
    sol = solve_ivp(lorenz, t_span, initial_state, args=(sigma, rho, beta), t_eval=t_eval)

    # Plotting
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(projection='3d')
    ax.plot(sol.y[0], sol.y[1], sol.y[2], lw=0.5)
    ax.set_title(f"Lorenz Attractor (σ={sigma}, ρ={rho}, β={beta})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

# Interactive sliders
interact(plot_lorenz,
         sigma=FloatSlider(min=0, max=20, step=0.5, value=10),
         rho=FloatSlider(min=0, max=50, step=1, value=28),
         beta=FloatSlider(min=0.5, max=5, step=0.1, value=8/3))
