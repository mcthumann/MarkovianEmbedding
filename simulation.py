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
    def __init__(self, n, v_i, gamma_i, delta, timestep, sample_rate, lag_fraction, temp=-1, mass=-1, gamma=-1):
        # Parameters
        self.n = n # Number of auxliary stochastic variables
        self.v_i = v_i
        self.gamma_i = gamma_i
        self.delta = delta

        # Optional Params
        self.sample_rate = sample_rate
        self.lag_fraction = lag_fraction
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

        self.timestep = timestep
        # Single step variables
        self.curr_x = None
        self.curr_v = None
        self.curr_u = None

        # Memory for single time trace
        self.all_x = None
        self.all_v = None
        self.all_u = None

        # Memory of multiple traces
        self.all_pacf = []
        self.all_vacf = []
        self.all_msd = []
        self.all_psd = []

    # reset_trace() should be called at the end of simulating a single time trace. It will
    # set the variables to initial conditions, add the time trace correlation function and
    # MSD to memory, and clear the last time trace from memory
    def reset_trace(self, trace_len):
        self.curr_x = 0
        self.curr_v = 0
        self.curr_u = [np.random.normal(0, math.sqrt(var)) for var in self.gamma_i]
        # Allocate arrays and set first value
        self.all_x = np.zeros(trace_len)
        self.all_x[0] = self.curr_x
        self.all_v = np.zeros(trace_len)
        self.all_v[0] = self.curr_v
        self.all_u = np.empty(trace_len, dtype=object)
        self.all_u[0] = self.curr_u

    # Begin stepping. At each step we generate N+1 random numbers from the standard
    # normal distribution. We use these to calculate u, v, and x. We save the values from this state and
    # continue on tho generate the next state.
    def compute_next_state(self, state_ind):
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

        self.all_x[state_ind] = curr_x
        self.all_v[state_ind] = curr_v
        self.all_u[state_ind] = curr_u

        self.curr_x = curr_x
        self.curr_v = curr_v
        self.curr_u = curr_u

    def run_numerical_simulation(self, sim_num, trace_len, pacf=True, vacf=True, msd=True, psd=True, graph=False):
        print("Will Simulate " + str(trace_len * self.sample_rate) + " points, sampling every " + str(self.sample_rate)
              + " for a duration of " + str(trace_len * self.sample_rate * self.timestep) + " time constants")
        print("trace_len size" + str(trace_len))
        self.reset_trace(trace_len)
        for i in range(sim_num):
            for j in range(trace_len):
                if j%(int(trace_len/100))==0:
                    print("#", end="-")
                self.compute_next_state(j)
            if graph:
                self.graph_v()
            if pacf:
                print("Computing PACF")
                self.compute_PACF()
            if vacf:
                print("Computing VACF")
                self.compute_VACF()
            if msd:
                print("Computing MSD")
                self.compute_MSD()
            if psd:
                print("Computing MSD")
                self.compute_PSD()
            if i < sim_num - 1:
                self.reset_trace(trace_len)
        if graph:
            plt.show()

    # Function to graph all x positions
    def graph_x(self):
        plt.plot([t*(self.timestep*self.sample_rate)*self.t_c for t in range(len(self.all_x))], self.all_x*self.x_c, linewidth=0.5)

    # Function to graph all velocities
    def graph_v(self):
        plt.plot([t*(self.timestep*self.sample_rate)*self.t_c for t in range(len(self.all_v))], self.all_v*self.v_c, linewidth=0.5)

    def compute_PACF(self, transient=0.0):
        # Remove transient section
        trace = self.all_x[int(transient * len(self.all_x)):]
        N = len(trace)
        max_lag = int(self.lag_fraction * N)

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

    def compute_VACF(self, transient=0.0):
        # Remove transient section
        v = self.all_v[int(transient * len(self.all_v)):]
        N = len(v)
        max_lag = int(self.lag_fraction * N)

        # Compute correlation using FFT for faster performance
        f_v = fft(v, n=2 * N)  # zero-pad to double length to avoid aliasing
        acf = ifft(f_v * np.conj(f_v)).real[:N]  # autocorrelation via inverse FFT
        acf = acf[:max_lag]  # Keep positive lags only, up to max_lag

        # Normalize by the number of terms contributing to each lag
        acf /= np.arange(N, N - max_lag, -1)

        self.all_vacf.append(acf)

    def compute_MSD(self, skip_lags=1, transient=0.0):
        # Apply transient trimming to the time trace
        trace = self.all_x[int(transient * len(self.all_x)):]
        N = len(trace)
        max_lag = int(self.lag_fraction * N)

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

    def compute_PSD(self, transient=0.0):
        # Extract the velocity trace, excluding the transient portion if necessary
        trace = np.array(self.all_v[int(transient * len(self.all_v)):])
        # Compute the Fourier transform of the velocity data
        fft_result = fft(trace)
        # Compute the Power Spectral Density (PSD)
        psd = np.abs(fft_result) ** 2 / len(trace)
        # Generate the corresponding frequencies
        freqs = np.fft.fftfreq(len(trace), d=self.timestep)
        # Take only the positive frequencies and PSD values
        positive_freqs = freqs[:len(freqs) // 2]
        psd = psd[:len(psd) // 2]
        # Store or return the computed PSD
        self.all_psd.append((positive_freqs, psd))

    def graph_PSD(self):
        # Unpack and average the PSDs over all stored tuples
        all_psd_np = np.array([psd for _, psd in self.all_psd])
        mean_psd = np.mean(all_psd_np, axis=0)
        mean_psd=mean_psd*((self.v_c**2)*self.t_c)
        # Frequencies should be the same for all PSDs, so just take the first one
        positive_freqs = np.array(self.all_psd[0][0])/self.t_c

        # Plot the mean PSD
        plt.plot(positive_freqs, mean_psd, label="Simulation")
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD')

    def graph_PACF(self):
        all_pacf_np = np.array(self.all_pacf)
        mean_pacf = np.mean(all_pacf_np, axis = 0)
        plt.plot([t*(self.timestep*self.sample_rate)*self.t_c for t in range(np.size(mean_pacf))], mean_pacf, label="Simulation")

    def graph_VACF(self, start, stop):
        all_vacf_np = np.array(self.all_vacf)
        mean_vacf = np.mean(all_vacf_np, axis=0)
        print("mean vacf size " + str(np.size(mean_vacf)))
        print("last x " + str(np.size(mean_vacf)*self.t_c))
        plt.plot([t*(self.timestep*self.sample_rate)*self.t_c for t in range(np.size(mean_vacf))], mean_vacf, label="Simulation")
        plt.xlim(start, stop)

    def graph_MSD(self):
        all_msd_np = np.array(self.all_msd)
        mean_msd = np.mean(all_msd_np, axis = 0)
        plt.plot([t*(self.timestep*self.sample_rate)*self.t_c for t in range(np.size(mean_msd))], mean_msd/self.x_c**2, label="Simulation")
        plt.xscale('log')
        plt.yscale('log')

# TODO Look for superdiffusion!

class SuperDiffusiveSimulation(MarkovianEmbeddingProcess):
    def __init__(self, n, v_i, gamma_i, delta, timestep, sample_rate, lag_fraction, velocity_tolerance, temp=-1, mass=-1, gamma=-1):
        # Call the parent class's constructor
        super().__init__(n, v_i, gamma_i, delta, timestep, sample_rate, lag_fraction, temp=-1, mass=-1, gamma=-1)
        self.all_x_list = None
        self.all_v_list = None
        self.zero_indicies = None
        self.vel_tolerance = velocity_tolerance

    def save_trace(self, i):
        self.all_x_list[i] = (self.all_x)
        self.all_v_list[i] = (self.all_v)

    def reset_trace_sd(self, trace_len):
        self.curr_x = 0
        self.curr_v = 0
        self.curr_u = [np.random.normal(0, math.sqrt(var)) for var in self.gamma_i]
        # Allocate arrays and set first value
        self.all_x = np.zeros(trace_len)
        self.all_x[0] = self.curr_x
        self.all_v = np.zeros(trace_len)
        self.all_v[0] = self.curr_v
        self.all_u = np.empty(trace_len, dtype=object)
        self.all_u[0] = self.curr_u

    def run_sim_find_super_diff(self, sim_num, trace_len):
        self.all_x_list = np.zeros((sim_num, trace_len))
        self.all_v_list = np.zeros((sim_num, trace_len))
        self.reset_trace_sd(trace_len)
        for i in range(sim_num):
            for j in range(trace_len):
                if j % (int(trace_len / 100)) == 0:
                    print("#", end="-")
                self.compute_next_state(j)
            print()
            self.save_trace(i)
            if i < sim_num - 1:
                self.reset_trace_sd(trace_len)

        # Note, we are assuming the msd we want will be 10% of the total trace len
        # We have thrown out indicies in the first and last 10%

        # now find where its zero
        close_to_zero_indices = []
        for i in range(sim_num):
            close_to_zero_indices.append(np.where(abs(self.all_v_list[i,:]) < self.vel_tolerance)[0])

        self.zero_indicies = [row[(row > int(trace_len*0.1)) & (row < int(trace_len*0.9))] for row in close_to_zero_indices]

        # find some random indicies (use same amount as close to zero indicies)
        random_indicies = []
        for i in range(sim_num):
            random_indicies.append(np.random.randint(int(trace_len*0.1), int(trace_len*0.9), size=len(self.zero_indicies[i])))

        rand_msd = []
        # calculate msd for everything, for the random sample, and for the zero velocity sample
        for i in range(len(random_indicies)):
            for idx in random_indicies[i]:
                rand_msd.append(self.get_msd_from_idx(i, idx, int(trace_len*0.01)))

        zero_msd = []
        for i in range(len(self.zero_indicies)):
            for idx in self.zero_indicies[i]:
                zero_msd.append(self.get_msd_from_idx(i, idx, int(trace_len*0.01)))

        # Graph the two msd's
        print("averaged " + str(len(zero_msd)) + " msds")
        zero_msd_np = np.array(zero_msd)
        mean_msd = np.mean(zero_msd_np, axis=0)
        plt.plot([t for t in range(np.size(mean_msd))],
                 mean_msd / self.x_c ** 2, label="Super Diffusive MSD")

        rand_msd_np = np.array(rand_msd)
        mean_msd = np.mean(rand_msd_np, axis=0)
        plt.plot([t for t in range(np.size(mean_msd))],
                 mean_msd / self.x_c ** 2, label="Regular MSD")
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.show()

        self.graph_all_traces()

    def get_msd_from_idx(self, trace_idx, trace_pos_start, siz):
        # Extract the relevant portion of the trace
        trace = self.all_x_list[trace_idx, trace_pos_start:trace_pos_start + siz]

        # Initialize MSD array
        msd = np.zeros(len(trace))

        # Compute MSD for each lag delta_t
        for delta_t in range(1, len(trace)):
            # Displacement relative to the first point (trace[0])
            displacements = trace[delta_t:] - trace[0]
            # Mean squared displacement
            msd[delta_t] = np.mean(displacements ** 2)

        # Return the computed MSD for this trace
        return msd

    def graph_all_traces(self):

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(self.all_x_list)]
        for i in range(len(self.all_x_list)):
            """* (self.timestep * self.sample_rate) * self.t_c"""
            plt.plot([t for t in range(len(self.all_x_list[i]))],
                     self.all_x_list[i] * self.x_c, color=colors[i], linewidth=0.5)
            # Draw vertical lines where values are close to zero
            for index in self.zero_indicies[i]:
                plt.axvline(x=index, color=colors[i], linestyle='-', linewidth=1)
        plt.show()
        for i in range(len(self.all_v_list)):
            plt.plot([t for t in range(len(self.all_v_list[i]))],
                     self.all_v_list[i] * self.v_c, color=colors[i], linewidth=0.3)
            # Draw vertical lines where values are close to zero
            for index in self.zero_indicies[i]:
                plt.axvline(x=index, color=colors[i], linestyle='-', linewidth=1)
        plt.show()
