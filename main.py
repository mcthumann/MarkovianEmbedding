# This is the main file for a program that  will simulate the full hydrodynamic theory
# of short time brownian motion by embedding the memory dependent kernal into Markovian
# states

import math
import numpy as np
import matplotlib.pyplot as plt

class MarkovianEmbeddingProcess:
    """
    Markovian processes use the current state to find the next state - in this class, we
    have 2 + n phase space variables to track at each state of the process. x and v are the
    position and velocity phase space variables and u is a vector of n variables that help
    project the non-Markovian nature of memory dependent hydrodynamic effects into a markov process.
    """
    # First, we initialize x, v and u appropriately. Entries of u should be initialized by sampling randomly
    # from a gaussian distribution with mean 0 and variance gamma_i.
    def __init__(self, n, v_i, gamma_i, delta, timestep):
        self.n = n # Number of auxliary stochastic variables
        self.v_i = v_i
        self.gamma_i = gamma_i
        self.delta = delta
        self.timestep = timestep
        self.curr_x = 0
        self.curr_v = 0
        self.curr_u = [np.random.normal(0, math.sqrt(var)) for var in gamma_i]
        self.all_x = [self.curr_x]
        self.all_v = [self.curr_v]
        self.all_u = [self.curr_u]

    # Begin stepping. At each step we generate N+1 random numbers from the standard
    # normal distribution. We use these to calculate u, v, and x. We save the values from this state and
    # continue on tho generate the next state.
    def compute_next_state(self):
        N_i = np.random.normal(0,1, self.n+1)
        N_0 = sum([math.sqrt(self.gamma_i[j]/(self.v_i[j]*self.delta))*N_i[j] for j in range(self.n)])

        next_u = [((1 - self.v_i[j]*self.timestep)*self.curr_u[j] - self.gamma_i[j]*self.timestep*self.curr_v +
                   math.sqrt(2*self.gamma_i[j]*self.v_i[j]*self.timestep)*N_i[j]) for j in range(self.n)]
        next_v = ((1 - (1+self.delta)*self.timestep)*self.curr_v + - self.timestep*sum(self.curr_u) +
                  math.sqrt(2*self.timestep)*(math.sqrt(self.delta)*N_0 + N_i[self.n]))
        next_x = self.curr_x + self.timestep*self.curr_v

        self.all_x.append(next_x)
        self.all_v.append(next_v)
        self.all_u.append(next_u)

        self.curr_x = next_x
        self.curr_v = next_v
        self.curr_u = next_u

    def graph_x(self):
        plt.plot(self.all_x)
        plt.show()


def run():
    # PARAMETERS
    n = 13 # Number of auxliary stochastic variables
    b = 5 # Scaling dilation parameter that determines the low cutoff frequency ν_0*b^-n
    c = 1.78167 # a prefactor that depends on the particular choice of b
    v_0 = 10E3 # High cutoff frequency
    a = 10E-6 # Particle size
    eta = 0.89*10E-3 # Viscosity of water
    rho_silica = 2200 # Density of silica
    rho_f = 1000 # Density of water
    M = (4 / 3) * math.pi * a/2.0 ** 3 * rho_silica + .5 * (4 / 3) * math.pi * a/2.0 ** 3 * rho_f # Mass plus added mass

    # SECONDARY PARAMETERS
    tao_f = (rho_f * a**2)/eta # Characteristic time for vorticy diffusion across length of sphere
    gamma = 6*math.pi*a*eta # Steady-state friction coefficient of the particle
    tao_c = M/gamma # Momentum relaxation time of the particle
    timestep = 10E-4 # Simulation timestep
    v_i = [v_0*b**(-i) for i in range(1, n+1)] # Decaying samples from an exponential distribution ...
    gamma_i = [0.5*gamma*c*math.sqrt(tao_f/math.pi)*(v_i[i]**(3.0/2.0)) for i in range(n)]
    gamma_0 = 0.5*gamma*c*sum(math.sqrt(x) for x in v_i)
    delta = gamma_0/gamma

    mep = MarkovianEmbeddingProcess(n, v_i, gamma_i, delta, timestep)
    for i in range(10000):
        mep.compute_next_state()
    mep.graph_x()

if __name__ == '__main__':
    run()
