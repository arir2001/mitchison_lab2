# Fitting, ODEs etc. with SciPy

# Import packages
import numpy as np
from numpy.random import randn
from scipy import integrate, optimize
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 16,
    "figure.autolayout": True
})


####################### ROOT FINDING ######################

# Function whose zeros determines the mean-field spin S
def f_ising(S,tau):
    return S - np.tanh(S/tau)    

# Mean-field prediction for the Ising spin as a function of reduced temperature
def ising_S(tau):
    roots = optimize.fsolve(lambda S: f_ising(S,tau), 1)
    return roots.max()

# Plot the result as a function of temperature
temps = np.linspace(0.001,2)
spins = [ising_S(T) for T in temps]
plt.plot(temps, spins)
plt.xlabel('$k_B T/z J$')
plt.ylabel('$\\langle S\\rangle$')
plt.show()
plt.close() 


######################## FITTING ######################

# Define a function
def f(x):
    return np.exp(-0.25*np.abs(x)*np.cos(2*x))

# Plot the function for a given range and number N of sample
# N = 500
N = 2**10+1
x_min, x_max = 0,6
x1 = np.linspace(x_min, x_max, N)
plt.figure()
plt.plot(x1, f(x1))

# Generate a set of M noisy samples
M = 50
x2 = np.linspace(x_min, x_max, M)
y_noise = f(x2) + 0.1*randn(M)
plt.plot(x2, y_noise, linestyle = '', marker = '.', color = 'g')

# Define the function to fit
def fit_func(x, A, B):
    return np.exp(-A*np.abs(x)*np.cos(B*x))

# Define a function of residuals
def res(par, x, y):
    A,B = par
    return y - fit_func(x, A, B)

# Initial guess
A0, B0 = 0.1, 2.3

# Compute the fit
fit = optimize.least_squares(res, (A0, B0), args = (x2, y_noise))
print(fit)

# Plot the fitted results
A, B = fit.x
plt.plot(x1, fit_func(x1, A, B), 'r:')


######################## ODE SOLVING ######################

# Simple harmonic oscillator
m, k = 1.0, 1.0

# Derivative matrix
M = np.array([[0,1/m],[-k, 0]])
print(M)

# Time derivative function
def dydt(t,y):
    return M@y

# Fix the time domain
t0, tf = 0, 20

# Set the initial conditions
y0 = np.array([0.0,1.0])

# Solve the ODE
sol = integrate.solve_ivp(dydt, (t0,tf), y0)
print(sol)
# Plot the result
plt.figure()
plt.plot(sol.t, sol.y[0])

# Solve the ODE over a finer time grid
t0, tf, dt = 0, 20, 0.1
t_axis = np.arange(t0, tf, dt)
sol = integrate.solve_ivp(lambda t,y: M@y, (t0,tf), y0, t_eval = t_axis)
# Plot the result
plt.figure()
plt.plot(sol.t, sol.y[0])
plt.xlabel('$\\omega_0 t$')
plt.ylabel('$ x/x_0$')
plt.savefig('sho_x.png')

# Plot the solution in phase space
plt.figure()
plt.plot(sol.y[0], sol.y[1])
plt.xlabel('$x/x_0$')
plt.ylabel('$p/\\omega_0 m$')
plt.savefig('sho_xp.png')


##### Damped, driven harmonic oscillator

# Function to return the derivative given a force
def deriv(f, m, t, y):
    x,p = y
    dxdt = p/m
    dpdt = f(t, x, p)
    return (dxdt, dpdt)

# Forces acting on the damped, driven harmonic oscillator
A, w, gamma = 0.2, 1.2, 0.1
def forces_SHO(t, x, p):
    return -k*x - gamma*p + A*np.cos(w*t)

# Solve the differential equation
t0, tf, dt = 0, 100, 0.1
t_axis = np.arange(t0, tf, dt)
y0 = (0,0)
sol = integrate.solve_ivp(lambda t,y: deriv(forces_SHO,m,t,y), (t0,tf), y0, t_eval = t_axis) 
    
# Plot the result
plt.figure()
plt.plot(sol.t, sol.y[0])
plt.xlabel('t')
plt.ylabel('x')

# Plot the solution in phase space
plt.figure()
plt.plot(sol.y[0], sol.y[1])
plt.xlabel('x')
plt.ylabel('p')

##### van der Pol oscillator

# Forces for the van der Pol model
k, gamma, l = 1.0, 5.0, 1.0
def forces_vdP(t, x, p):
    return -k*x + gamma*(l**2-x**2)*p

# Solve the differential equation for several random initial conditions
# and plot the result in phase space
t0, tf, dt = 0, 100, 0.01
t_axis = np.arange(t0, tf, dt)
plt.figure() 
n_rand = 10
for nn in range(n_rand):
    y0 = randn(2)
    sol = integrate.solve_ivp(lambda t,y: deriv(forces_vdP,m,t,y), (t0,tf), y0, t_eval = t_axis) 
    plt.plot(sol.y[0], sol.y[1])
plt.xlabel('x')
plt.ylabel('p')

# Plot the position versus time
plt.figure()
plt.plot(sol.t, sol.y[0])
plt.xlabel('t')
plt.ylabel('x')
plt.xlim([0,50])



