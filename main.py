# This is the main file for a program that  will simulate the full hydrodynamic theory
# of short time brownian motion by embedding the memory dependent kernal into Markovian
# states

# TODO Follow the approach in the Hanggi papers to arrive at equation A.1 from Gomez-Solano
# TODO Find how to arrive at the system of coupled equations from the memory kernal A.1
# TODO Ensure that the rescaling process is accurate
# TODO Use the Euler-Maruyama method to discretize A.2
# TODO Figure out why the parameters n, b, c, v_0, were set as stated

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
        # Parameters
        self.n = n # Number of auxliary stochastic variables
        self.v_i = v_i
        self.gamma_i = gamma_i
        self.delta = delta
        self.timestep = timestep

        # Single step variables
        self.curr_x = 0
        self.curr_v = np.random.normal(0, 1)
        self.curr_u = [np.random.normal(0, math.sqrt(var)) for var in gamma_i]

        # Memory for single time trace
        self.all_x = [self.curr_x]
        self.all_v = [self.curr_v]
        self.all_u = [self.curr_u]

        # Memory of multiple traces
        self.all_pacf = []
        self.all_vacf = []
        self.all_msd = []

    # reset_trace() should be called at the end of simulating a single time trace. It will
    # set the variables to initial conditions, add the time trace correlation function and
    # MSD to memory, and clear the last time trace from memory
    def reset_trace(self):
        self.curr_x = 0
        self.curr_v = np.random.normal(0, 1)
        self.curr_u = [np.random.normal(0, math.sqrt(var)) for var in self.gamma_i]
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

    # Function to graph all x positions
    def graph_x(self):
        plt.plot(self.all_x, linewidth=0.5)

    # Function to graph all velocities
    def graph_v(self):
        plt.plot(self.all_v, linewidth=0.5)

    def compute_PACF(self, lag_fraction=0.1, transient=0.0):
        trace = self.all_x[int(transient * len(self.all_x)):]
        N = len(trace)
        max_lag = int(lag_fraction * N)
        mean_x = np.mean(trace)
        x_centered = [x - mean_x for x in trace]
        acf = np.correlate(x_centered, x_centered, mode='full')  # Compute correlation
        acf = acf[N:N + max_lag]  # Keep positive lags only, up to max_lag

        # Normalize by the number of terms contributing to each lag
        acf /= np.arange(N - 1, N - max_lag - 1, -1)

        # Normalize ACF so that ACF(0) = 1
        acf /= acf[0]
        self.all_pacf.append(acf)

    def compute_VACF(self, lag_fraction=0.01, transient = 0.0):
        trace = self.all_v[int(transient * len(self.all_v)):]
        N = len(trace)
        max_lag = int(lag_fraction * N)

        acf = np.correlate(trace, trace, mode='full')  # Compute correlation
        acf = acf[N:N + max_lag]  # Keep positive lags only, up to max_lag

        # Normalize by the number of terms contributing to each lag
        acf /= np.arange(N - 1, N - max_lag - 1, -1)

        # Normalize ACF so that ACF(0) = 1
        acf /= acf[0]
        self.all_vacf.append(acf)

    def compute_MSD(self, lag_fraction=0.1, skip_lags=1, transient=0.0):
        # Apply transient trimming to the time trace
        trace = np.array(self.all_x[int(transient * len(self.all_x)):])
        n = len(trace)
        max_lag = int(lag_fraction * N)

        # Initialize MSD array
        msd = np.zeros(n)

        # Loop over time lags but only compute MSD for every `skip_lags` lag
        for delta_t in range(1, n, skip_lags):
            # Use vectorized operation to compute displacements for this delta_t
            displacements = trace[delta_t:] - trace[:-delta_t]
            squared_displacements = displacements ** 2

            # Compute the mean of squared displacements for this lag
            msd[delta_t] = np.mean(squared_displacements)

        # Append the computed MSD for this trace
        self.all_msd.append(msd)

    def graph_PACF(self):
        all_pacf_np = np.array(self.all_pacf)
        mean_pacf = np.mean(all_pacf_np, axis = 0)
        plt.plot(mean_pacf)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()

    def graph_VACF(self):
        all_vacf_np = np.array(self.all_vacf)
        mean_vacf = np.mean(all_vacf_np, axis=0)
        plt.plot(mean_vacf)
        plt.xscale('log')
        plt.show()

    def graph_MSD(self):
        all_msd_np = np.array(self.all_msd)
        mean_msd = np.mean(all_msd_np, axis = 0)
        plt.plot(mean_msd)
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
    for i in range(100):
        for j in range(10000):
            mep.compute_next_state()
        mep.graph_v()
        mep.compute_PACF()
        mep.compute_VACF()
        mep.compute_MSD()
        mep.reset_trace()

    plt.show()
    mep.graph_PACF()
    mep.graph_VACF()
    mep.graph_MSD()

if __name__ == '__main__':
    run()
