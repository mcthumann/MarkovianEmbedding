# This is the main file for a program that  will simulate the full hydrodynamic theory
# of short time brownian motion by embedding the memory dependent kernal into Markovian
# states

# TODO Follow the approach in the Hanggi papers to arrive at equation A.1 from Gomez-Solano
# TODO Find how to arrive at the system of coupled equations from the memory kernal A.1
# TODO Ensure that the rescaling process is accurate
# TODO Use the Euler-Maruyama method to discretize A.2
# TODO Figure out why the parameters n, b, c, v_0, were set as stated

# TODO add the looping to the mep
import scipy
import math
import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import scipy.constants as const
import pandas as pd
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
    def __init__(self, n, v_i, gamma_i, delta, timestep, sample_rate, lag_fraction, K = 0, temp=-1, mass=-1, gamma=-1):
        # Parameters
        self.n = n # Number of auxliary stochastic variables
        self.v_i = v_i
        self.gamma_i = gamma_i
        self.delta = delta

        # Optional Params
        self.sample_rate = sample_rate
        print(self.sample_rate)
        self.lag_fraction = lag_fraction
        self.temp = temp
        self.mass = mass
        self.gamma = gamma
        self.K = K

        self.t_c = 1
        self.x_c = 1
        self.v_c = 1
        if self.temp > 0 and self.mass > 0 and self.gamma > 0:
            self.v_c = math.sqrt((const.k*self.temp)/self.mass)
            self.t_c = self.mass/self.gamma
            self.x_c = self.v_c*self.t_c
            self.f_c = (const.k*self.temp)/(self.t_c*self.v_c)

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
            next_v = ((1 - (1+self.delta)*self.timestep)*curr_v - self.timestep*sum(curr_u) -
                      self.timestep*self.K*curr_x*(self.x_c/self.f_c) +
                      math.sqrt(2*self.timestep)*(math.sqrt(self.delta)*N_0 + N_i[self.n]))
            next_x = curr_x + self.timestep*curr_v
            curr_u = next_u
            curr_v = next_v
            curr_x = next_x

        # print(const.k)
        # print(self.v_c)
        # print(self.t_c)
        # print((self.v_c/((const.k * self.temp)/(self.v_c*self.t_c))))
        self.all_x[state_ind] = curr_x
        self.all_v[state_ind] = curr_v
        self.all_u[state_ind] = curr_u

        self.curr_x = curr_x
        self.curr_v = curr_v
        self.curr_u = curr_u

    def run_numerical_simulation(self, sim_num, trace_len, pacf=False, vacf=False, msd=False, psd=False, df = None, graph=False, save=True):
        print("Will Simulate " + str(trace_len * self.sample_rate) + " points, sampling every " + str(self.sample_rate)
              + " for a duration of " + str(trace_len * self.sample_rate * self.timestep) + " time constants")

        self.reset_trace(trace_len)

        for i in range(sim_num):
            print("Trace " + str(i) + ": ")
            for j in range(int(trace_len/self.sample_rate)):
                if j%(int((trace_len/self.sample_rate)/100))==0:
                    print("#", end="-")
                self.compute_next_state(j)
            if graph:
                self.graph_v()
            if pacf:
                print("Computing PACF")
                self.compute_PACF()
            if vacf:
                print("Computing VACF")
                self.compute_VACF_time_domain()
            if msd:
                print("Computing MSD")
                self.compute_MSD()
            if psd:
                print("Computing MSD")
                self.compute_PSD()

            # save off data
            if save:
                # Create a temporary DataFrame for this trace
                temp_df = pd.DataFrame({'Position ' + str(i): self.all_x*self.x_c, 'Velocity ' + str(i): self.all_v*self.v_c})

                if df.empty:
                    # If the main DataFrame is empty, just set it equal to the temp_df
                    df = temp_df
                    print("empty df")
                else:
                    print("concat")
                    # Concatenate the new columns with the existing DataFrame
                    df = pd.concat([df.reset_index(drop=True), temp_df.reset_index(drop=True)], axis=1)

            if i < sim_num:
                if save:
                    df.to_csv('position_velocity_data.csv', index=False, mode='w')
                self.reset_trace(trace_len)
            print("")
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

    # def compute_VACF(self, transient=0.0):
    #     series = self.all_x[int(transient * len(self.all_x)):]
    #     v_series = np.diff(series)/self.timestep
    #     # v_series -= np.mean(v_series)
    #     f_signal = np.fft.fft(v_series, n=2 * len(v_series))
    #     vacf = np.fft.ifft(f_signal * np.conjugate(f_signal)).real[:len(v_series)]
    #     vacf /= len(self.all_x)*self.sample_rate
    #     self.all_vacf.append(vacf)

    def compute_VACF_time_domain(self, transient=0.0):

        v_series = (np.diff(self.all_x)/self.timestep)*self.v_c
        # v_series = self.all_v * self.v_c

        n = len(v_series)
        vacf = np.zeros(n)
        # Compute VACF with normalization
        for lag in range(n):
            vacf[lag] = np.dot(v_series[:n - lag], v_series[lag:]) / (
                        n - lag)  # Normalize by number of overlapping terms

        self.all_vacf.append(vacf)

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
        trace = np.array(self.all_x[int(transient * len(self.all_x)):])

        frequency, psd = scipy.signal.periodogram(trace, 1 / (self.timestep*self.sample_rate), scaling="density")
        frequency /= self.t_c
        psd *= (self.x_c**2)*self.t_c

        # Store or return the computed PSD
        self.all_psd.append((frequency, psd))

    def graph_PSD(self):
        # Unpack and average the PSDs over all stored tuples
        all_psd_np = np.array([psd for _, psd in self.all_psd])
        mean_psd = np.mean(all_psd_np, axis=0)
        # Frequencies should be the same for all PSDs, so just take the first one
        positive_freqs = np.array(self.all_psd[0][0])

        # Plot the mean PSD
        plt.plot(positive_freqs, mean_psd, label="Simulation")
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD')

    def graph_PACF(self):
        all_pacf_np = np.array(self.all_pacf)
        mean_pacf = np.mean(all_pacf_np, axis = 0)
        plt.plot([t*(self.timestep*self.sample_rate)*self.t_c for t in range(1, np.size(mean_pacf))], mean_pacf[1:], label="Simulation")
        plt.xscale('log')

    def graph_VACF(self, start, stop):
        all_vacf_np = np.array(self.all_vacf)
        mean_vacf = np.mean(all_vacf_np, axis=0)
        plt.plot([t*(self.timestep*self.sample_rate)*self.t_c for t in range(np.size(mean_vacf))], mean_vacf, label=f"Simulation")
        plt.xlim(start, stop)


    def graph_MSD(self):
        all_msd_np = np.array(self.all_msd)
        mean_msd = np.mean(all_msd_np, axis = 0)
        plt.plot([t*(self.timestep*self.sample_rate)*self.t_c for t in range(np.size(mean_msd))], mean_msd/self.x_c**2, label="Simulation")
        plt.xscale('log')
        plt.yscale('log')

# TODO Look for superdiffusion!