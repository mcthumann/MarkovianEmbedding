# This is the main file for a program that  will simulate the full hydrodynamic theory
# of short time brownian motion by embedding the memory dependent kernal into Markovian
# states

# TODO Follow the approach in the Hanggi papers to arrive at equation A.1 from Gomez-Solano
# TODO Find how to arrive at the system of coupled equations from the memory kernal A.1
# TODO Ensure that the rescaling process is accurate
# TODO Use the Euler-Maruyama method to discretize A.2
# TODO Figure out why the parameters n, b, c, v_0, were set as stated

# TODO add the looping to the mep
import math
import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import scipy.constants as const
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)

class MarkovianEmbeddingProcess:
    """
    Markovian processes use the current state to find the next state - in this class, we
    have 2 + n phase space variables to track at each state of the process. x and v are the
    position and velocity phase space variables and u is a vector of n variables that help
    project the non-Markovian nature of memory dependent hydrodynamic effects into a markov process.
    """
    # First, we initialize x, v and u appropriately. Entries of u should be initialized by sampling randomly
    # from a gaussian distribution with mean 0 and variance gamma_i.
    def __init__(self, n, v_i, gamma_i, delta, timestep, sample_rate=10, temp=-1, mass=-1, gamma=-1):
        # Parameters
        self.n = n # Number of auxliary stochastic variables
        self.v_i = v_i
        self.gamma_i = gamma_i
        self.delta = delta
        self.timestep = timestep

        # Optional Params
        self.sample_rate = sample_rate
        self.temp = temp
        self.mass = mass
        self.gamma = gamma

        self.t_c = 1
        self.x_c = 1
        self.v_c = 1
        if self.temp > 0 and self.mass > 0 and self.gamma > 0:
            print("Dimensionful quantities")
            self.v_c = math.sqrt((const.k*self.temp)/self.mass)
            self.t_c = self.mass/self.gamma
            self.x_c = self.v_c*self.t_c

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
        curr_u = self.curr_u
        curr_v = self.curr_v
        curr_x = self.curr_x

        for k in range(self.sample_rate):
            N_i = np.random.normal(0,1, self.n+1)
            N_0 = sum([math.sqrt(self.gamma_i[j]/(self.v_i[j]*self.delta))*N_i[j] for j in range(self.n)])

            next_u = [((1 - self.v_i[j]*self.timestep)*curr_u[j] - self.gamma_i[j]*self.timestep*curr_v +
                       math.sqrt(2*self.gamma_i[j]*self.v_i[j]*self.timestep)*N_i[j]) for j in range(self.n)]
            next_v = ((1 - (1+self.delta)*self.timestep)*curr_v + - self.timestep*sum(curr_u) +
                      math.sqrt(2*self.timestep)*(math.sqrt(self.delta)*N_0 + N_i[self.n]))
            next_x = curr_x + self.timestep*curr_v
            curr_u = next_u
            curr_v = next_v
            curr_x = next_x

        self.all_x.append(curr_x)
        self.all_v.append(curr_v)
        self.all_u.append(curr_u)

        self.curr_x = curr_x
        self.curr_v = curr_v
        self.curr_u = curr_u

    def run_numerical_simulation(self, sim_num, trace_len, pacf=True, vacf=True, msd=True, graph_traces=False):
        for i in range(sim_num):
            for j in range(trace_len):
                if j%(int(trace_len/100))==0:
                    print("#", end="")
                self.compute_next_state()
            if graph_traces:
                self.graph_x()
            if pacf:
                print("Computing PACF")
                self.compute_PACF()
            if vacf:
                print("Computing VACF")
                self.compute_VACF()
            if msd:
                print("Computing MSD")
                self.compute_MSD()
            self.reset_trace()
        if graph_traces:
            plt.show()
        print("Simulated " + str(trace_len*self.sample_rate) + " points, sampling every " + str(self.sample_rate)
              + " for a duration of " + str(trace_len*self.sample_rate*self.timestep) + " time constants")

    # Function to graph all x positions
    def graph_x(self):
        plt.plot([t*(self.timestep*self.sample_rate)*self.t_c for t in range(len(self.all_x))], [x*self.x_c for x in self.all_x], linewidth=0.5)

    # Function to graph all velocities
    def graph_v(self):
        plt.plot([t*(self.timestep*self.sample_rate)*self.t_c for t in range(len(self.all_v))], [v*self.v_c for v in self.all_v], linewidth=0.5)

    def compute_PACF(self, lag_fraction=0.1, transient=0.0):
        # Remove transient section
        trace = np.array(self.all_x[int(transient * len(self.all_x)):])
        N = len(trace)
        max_lag = int(lag_fraction * N)

        # Compute the mean and center the trace (vectorized)
        mean_x = np.mean(trace)
        x_centered = trace - mean_x

        # Use FFT to compute autocorrelation efficiently
        f_x = fft(x_centered, n=2 * N)  # Zero-padding to avoid aliasing
        acf = ifft(f_x * np.conj(f_x)).real[:N]  # Autocorrelation via inverse FFT
        acf = acf[:max_lag]  # Keep only positive lags, up to max_lag

        # Normalize by the number of terms contributing to each lag
        acf /= np.arange(N, N - max_lag, -1)

        # Normalize ACF so that ACF(0) = 1
        acf /= acf[0]

        # Append the computed PACF for this trace
        self.all_pacf.append(acf)

    def compute_VACF(self, lag_fraction=0.01, transient=0.0):
        # Remove transient section
        v = self.all_v[int(transient * len(self.all_v)):]
        N = len(v)
        max_lag = int(lag_fraction * N)

        # Compute correlation using FFT for faster performance
        f_v = fft(v, n=2 * N)  # zero-pad to double length to avoid aliasing
        acf = ifft(f_v * np.conj(f_v)).real[:N]  # autocorrelation via inverse FFT
        acf = acf[:max_lag]  # Keep positive lags only, up to max_lag

        # Normalize by the number of terms contributing to each lag
        acf /= np.arange(N, N - max_lag, -1)

        # Normalize ACF so that ACF(0) = 1
        acf /= acf[0]
        self.all_vacf.append(acf)

    def compute_MSD(self, lag_fraction=0.1, skip_lags=1, transient=0.0):
        # Apply transient trimming to the time trace
        trace = np.array(self.all_x[int(transient * len(self.all_x)):])
        N = len(trace)
        max_lag = int(lag_fraction * N)

        # Initialize MSD array
        msd = np.zeros(max_lag)

        # Vectorized computation for all lags at once
        for delta_t in range(1, max_lag, skip_lags):
            # Vectorized displacement calculation: trace[delta_t:] - trace[:-delta_t]
            displacements = trace[delta_t:] - trace[:-delta_t]

            # Use np.mean to compute the mean squared displacement
            msd[delta_t] = np.mean(displacements ** 2)

        # Append the computed MSD for this trace
        self.all_msd.append(msd)

    def graph_PACF(self):
        all_pacf_np = np.array(self.all_pacf)
        mean_pacf = np.mean(all_pacf_np, axis = 0)
        plt.plot([t*(self.timestep*self.sample_rate)*self.t_c for t in range(np.size(mean_pacf))], mean_pacf)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()

    def graph_VACF(self):
        all_vacf_np = np.array(self.all_vacf)
        mean_vacf = np.mean(all_vacf_np, axis=0)
        plt.plot([t*(self.timestep*self.sample_rate)*self.t_c for t in range(np.size(mean_vacf))], mean_vacf)
        plt.xscale('log')
        plt.show()

    def graph_MSD(self):
        all_msd_np = np.array(self.all_msd)
        mean_msd = np.mean(all_msd_np, axis = 0)
        plt.plot([t*(self.timestep*self.sample_rate)*self.t_c for t in range(np.size(mean_msd))], mean_msd/self.x_c**2)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()

def run():
    # PARAMETERS
    n = 13 # Number of auxliary stochastic variables
    b = 5 # Scaling dilation parameter that determines the low cutoff frequency ν_0*b^-n
    c = 1.78167 # a prefactor that depends on the particular choice of b
    v_0 = 1E3 # High cutoff frequency
    a = 1E-6 # Particle size
    eta = 0.89*1E-3 # Viscosity of water
    rho_silica = 2200 # Density of silica
    rho_f = 1000 # Density of water
    M = (4 / 3) * math.pi * (a/2.0)** 3 * rho_silica + .5 * (4 / 3) * math.pi * (a/2.0)** 3 * rho_f # Mass plus added mass
    temp = 273

    # SECONDARY PARAMETERS
    tao_f = (rho_f * a**2)/eta # Characteristic time for vorticy diffusion across length of sphere
    gamma = 6*math.pi*a*eta # Steady-state friction coefficient of the particle
    tao_c = M/gamma # Momentum relaxation time of the particle
    v_c = math.sqrt((const.k*temp)/M)
    x_c = tao_c*v_c
    f_c = (const.k*temp)/x_c
    timestep = 1E-4 # Simulation timestep
    v_i = [v_0*b**(-i) for i in range(1, n+1)] # Decaying samples from an exponential distribution ...
    gamma_i = [(x_c/f_c)*(0.5*gamma*c*math.sqrt(tao_f/math.pi)*(v_i[i]**(3.0/2.0))) for i in range(n)]
    gamma_0 = 0.5*gamma*c*sum(math.sqrt(v*x_c) for v in v_i)
    delta = gamma_0/gamma

    sample_rate = 10
    simulation_number = 100
    trace_length = 10**6

    mep = MarkovianEmbeddingProcess(n, v_i, gamma_i, delta, timestep, sample_rate=sample_rate, temp=temp, mass=M, gamma=gamma)
    mep.run_numerical_simulation(simulation_number, trace_length, graph_traces=True, msd=True)

    mep.graph_PACF()
    mep.graph_VACF()
    mep.graph_MSD()

if __name__ == '__main__':
    run()
