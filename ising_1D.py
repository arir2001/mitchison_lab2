import numpy as np
import matplotlib.pyplot as plt
from time import time

class IsingModel1D:
    def __init__(self, N, T, h, state = None):
        self.N = N  # Number of spins
        self.temp = T  # Temperature
        self.field = h
        if state is None:
            # Initialize spins randomly if no initial state is given
            self.spins = np.random.choice([-1, 1], size=N)  
        else:
            self.spins = state

    def energy(self):
        """Calculate the energy of the current spin configuration."""
        return -np.sum(self.spins[:-1] * self.spins[1:]) - self.field*self.magnetization()

    def metropolis_step(self, sweep = 'random'):
        """Perform one Metropolis step."""
        for j in range(self.N):
            if sweep == 'random':
                i = np.random.randint(0, self.N)  # Choose a random spin
            elif sweep == 'sequential':
                i = j
            delta_E = 2 * self.spins[i] * (self.spins[i-1] + self.spins[(i+1) % self.N]) + 2*self.field*self.spins[i]  # Energy change if this spin is flipped
            if delta_E < 0 or np.random.rand() < np.exp(-delta_E / self.temp):
                self.spins[i] *= -1  # Flip the spin

    def print_info(self):
        print('1D Ising spin lattice')
        print('-'*20)
        print('Number of spins:', self.N)
        print('Temperature:', self.temp, ' J/k_B')
        print('Magnetisation:', self.magnetization())
        print('Energy:', self.energy(), ' J')        

    def simulate(self, steps = 500, init_steps = 100, sweep = 'random'):
        """Simulate the Ising model for a given number of steps."""
        # Initialise the lattice 
        for _ in range(init_steps):
            self.metropolis_step(sweep)
        # Perform metropolis steps and average observables
        M, E, M_var = 0, 0, 0
        for _ in range(steps):
            self.metropolis_step()
            M += self.magnetization()
            M_var += self.magnetization()**2
            E += self.energy()
        E = E/steps
        M = M/steps
        M_var = M_var/steps - M**2
        return M, E, M_var, self.spins

    def magnetization(self):
        """Calculate the magnetization of the current spin configuration."""
        return np.sum(self.spins)

total_time = time()

N = 100  # Number of spins
temp = 1.0  # Temperature
hs = np.linspace(0,2,50)

# ###### RANDOM UPDATING ######

# Compute average magnetisation and energy as a function of the magnetic field
Ms, Es, chis = [], [], []
state = np.random.choice([-1,1], size = N)
for h in hs:
    ising_model = IsingModel1D(N, temp, h)
    M, E, M_var, _ = ising_model.simulate()
    Ms.append(M)
    Es.append(E)
    chis.append(M_var/temp)
# Print the information about the lattice    
ising_model.print_info()

# Plot the magnetisation as a function of external magnetic field,
# demonstrating paramagnetic behaviour at low field and saturation at large field
fig, ax = plt.subplots()
ax.plot(hs, Ms)
ax.set_xlabel('$h/J$')
ax.set_ylabel('$\\langle M \\rangle$')

# Plot the energy as a function of magnetic field, showing near-linear decrease for large field
fig, ax = plt.subplots()
ax.plot(hs, Es)
ax.set_xlabel('$h/J$')
ax.set_ylabel('$\\langle E \\rangle$')

# Plot the slope of the magnetisation together with the susceptibility
fig, ax = plt.subplots()
dh = hs[1]-hs[0]
dMdh = np.diff(Ms)/dh
ax.plot(hs[:-1], dMdh)
ax.plot(hs, chis, 'r:')
ax.set_xlabel('$h/J$')
ax.set_ylabel('$d\\langle M\\rangle/dh$ $[J^{-1}]$')


# Compute average magnetisation as a function of temperature for various fields
# Use the final state of the last run as the initial state of the next run, allowing us to skip the initialisation steps
# Plot on logarithmic scale to show Curie's law 
# Show the value where k_B T = h as a vertical dashed line
hs = [0.5, 1.0, 2.0, 3.0]
Ts = np.logspace(-1.0,1.5, 50)
spins = np.ones(N)
fig, ax = plt.subplots()
for h in hs:
    Ms = []
    for temp in Ts:
        ising_model = IsingModel1D(N, temp, h, state = spins)
        M, E, M_var, spins = ising_model.simulate(init_steps = 0)
        Ms.append(M)
    line, = ax.loglog(Ts, Ms, label = ['h = '+str(h)+'J'])
    ax.axvline(x=h, color = line.get_color(), linestyle = ':')
ax.set_xlabel('$k_BT/J$')
ax.set_ylabel('$\\langle M\\rangle$')
ax.legend()


###### SEQUENTIAL VS. RANDOM UPDATING ######

# Parameters
N, T, h = 400, 2.0, 2.0
n_steps, dn = 400, 1

# List of possible values for the total number of steps/sweeps
steps_list = np.arange(dn,n_steps,dn)


# Check for convergence using random updating
ising_lattice = IsingModel1D(N, T, h, state = np.random.choice([-1,1], N))
Ms_random = []
for i in range(len(steps_list)):
    M , _ , _ , _ = ising_lattice.simulate(steps = dn, init_steps = 0)
    Ms_random.append(M)

# Check for convergence using sequential updating
ising_lattice = IsingModel1D(N, T, h, state =  np.random.choice([-1,1], N))
Ms_sequential = []
for i in range(len(steps_list)):
    M , _ , _ , _ = ising_lattice.simulate(steps = dn, init_steps = 0, sweep = 'sequential')
    Ms_sequential.append(M)

# Find the corresponding averages by adding up the values and dividing by the number of steps
Ms_random = np.cumsum(Ms_random)/steps_list/N
Ms_sequential = np.cumsum(Ms_sequential)/steps_list/N

# Compare the two sets of results 
fig, ax = plt.subplots()
ax.plot(steps_list, Ms_random)
ax.plot(steps_list, Ms_sequential, 'r:')
ax.set_xlabel('number of steps')
ax.set_ylabel('$\\langle M\\rangle$')

# Report total simulation time
total_time = time()-total_time
print('Total simulation time: ', total_time, 's')
