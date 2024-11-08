import pandas as pd
import numpy as np
from numpy.fft import fft, ifft
import matplotlib
import matplotlib.pyplot as plt
import pickle
import scipy

matplotlib.use('TkAgg')

# Load the DataFrame from a CSV file
df = pd.read_csv('position_velocity_data.csv')

# Assuming each trace has two columns: 'Position' and 'Velocity'
num_columns = df.shape[1]
num_traces = num_columns // 2

class DataProcessor:
    def __init__(self, df, lag_fraction=0.1):
        self.df = df
        self.lag_fraction = lag_fraction

        self.mass_total = df['mass_total']
        self.timestep = df['timestep']
        self.sample_rate = df['sample_rate']
        self.t_c  = df['tao_c']
        self.v_c = df['v_c']
        print("v_c " + str(self.v_c))
        self.x_c  = df['x_c']
        self.k_b = scipy.constants.k

        print(self.timestep, self.sample_rate, self.t_c, self.v_c, self.x_c)

        self.all_pacf = []
        self.all_vacf = []
        self.all_msd = []
        self.all_psd = []


        self.heat_transfer_lags = np.array([.003, .01, .03, .1, .3, 1, 3])
        self.all_delta_Q = [[] for _ in range(len(self.heat_transfer_lags))]

    def compute_PACF(self, position_trace, transient=0.0):
        trace = position_trace[int(transient * len(position_trace)):]
        N = len(trace)
        max_lag = int(self.lag_fraction * N)
        mean_x = np.mean(trace)
        x_centered = trace - mean_x
        f_x = fft(x_centered, n=2 * N)
        acf = ifft(f_x * np.conj(f_x)).real[:N]
        acf = acf[:max_lag]
        acf /= np.arange(N, N - max_lag, -1)
        acf /= acf[0]
        self.all_pacf.append(acf)

    def compute_VACF(self, velocity_trace, transient=0.0):
        v = velocity_trace[int(transient * len(velocity_trace)):]
        N = len(v)
        max_lag = int(self.lag_fraction * N)
        # Ensure v is a NumPy array for FFT operation
        v = np.array(v)

        f_v = fft(v, n=2 * N)
        acf = ifft(f_v * np.conj(f_v)).real[:N]
        acf = acf[:max_lag]
        acf /= np.arange(N, N - max_lag, -1)
        self.all_vacf.append(acf)

    def compute_MSD(self, position_trace, transient=0.0):
        # Apply transient trimming to the position trace
        trace = np.array(position_trace[int(transient * len(position_trace)):])  # Convert to NumPy array
        N = len(trace)  # Length of the trace

        # Set a cutoff where we switch from linear to logarithmic spacing
        linear_cutoff = 100  # Compute linearly up to this point

        # Generate linear lags for small lag times
        linear_lags = np.arange(1, linear_cutoff, dtype=int)

        # Generate logarithmic lags for larger lag times (avoiding duplicates)
        log_lags = np.unique(np.logspace(np.log10(linear_cutoff), np.log10(N), num=500, dtype=int))

        # Combine linear and logarithmic lags
        all_lags = np.unique(np.concatenate([linear_lags, log_lags]))

        # Initialize MSD array (fill with NaN to indicate unused indices)
        msd = np.full(N, np.nan)

        # Compute MSD for the selected lags
        for delta_t in all_lags:
            if delta_t < N:  # Ensure delta_t is within the valid range
                displacements = trace[delta_t:] - trace[:-delta_t]  # Vectorized displacement calculation
                msd[delta_t] = np.mean(displacements ** 2)
            else:
                print(f"Warning: delta_t={delta_t} exceeds trace length. Skipping this lag.")
                break  # If delta_t exceeds N, stop computing lags

        # Append the computed MSD for this trace
        self.all_msd.append(msd)

    def compute_PSD(self, velocity_trace, transient=0.0):
        trace = np.array(velocity_trace[int(transient * len(velocity_trace)):])
        fft_result = fft(trace)
        psd = np.abs(fft_result) ** 2 / len(trace)
        freqs = np.fft.fftfreq(len(trace), d=self.timestep)
        positive_freqs = freqs[:len(freqs) // 2]
        psd = psd[:len(psd) // 2]
        self.all_psd.append((positive_freqs, psd))

    def compute_heat_transfer(self, velocity_trace, lags, transient=0.0):
        trace = np.array(velocity_trace[int(transient * len(velocity_trace)):])
        lag_idxs = np.round(lags/self.timestep).astype(int)  # Convert to integer number of time steps

        # Compute Delta Q in a vectorized manner for each lag
        for i, lag in enumerate(lag_idxs):
            if lag < len(trace):  # Ensure we don't go out of bounds
                v_t0 = trace[:len(trace)-lag]  # Velocity at time t_0
                v_t_lag = trace[lag:]  # Velocity at time t_0 + lag

                # Compute Delta Q for all t_0 at once
                self.all_delta_Q[i].extend(((self.mass_total / 2) * (v_t_lag ** 2) * self.v_c**2 / (self.k_b*293)) - ((self.mass_total / 2) * (v_t0 ** 2) * self.v_c**2 / (self.k_b*293)))

    def process_traces(self):
        for i in range(0, num_traces):
            print("Working on trace", i)
            position_trace = self.df[f'Position {i}']
            velocity_trace = self.df[f'Velocity {i}']
            # self.compute_PACF(position_trace)
            self.compute_VACF(velocity_trace)
            # self.compute_MSD(position_trace)
            # self.compute_PSD(velocity_trace)
            self.compute_heat_transfer(velocity_trace, self.heat_transfer_lags)

    def graph_PSD(self):
        all_psd_np = np.array([psd for _, psd in self.all_psd])
        mean_psd = np.mean(all_psd_np, axis=0)
        mean_psd = mean_psd * (self.v_c ** 2) * self.t_c
        positive_freqs = np.array(self.all_psd[0][0]) / self.t_c

        # Plot the mean PSD with labels and legend
        plt.plot(positive_freqs, mean_psd, label="Mean PSD")
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power Spectral Density')
        plt.title('Mean Power Spectral Density (PSD)')
        plt.legend()
        plt.show()

    def graph_PACF(self):
        all_pacf_np = np.array(self.all_pacf)
        mean_pacf = np.mean(all_pacf_np, axis=0)

        # Plot the mean PACF with labels and legend
        plt.plot([t * (self.timestep * self.sample_rate) * self.t_c for t in range(1, np.size(mean_pacf))], mean_pacf[1:], label="Mean PACF")
        plt.xlabel('Time Lag')
        plt.xscale("log")
        plt.ylabel('Position Autocorrelation')
        plt.title('Mean Position Autocorrelation Function (PACF)')
        plt.legend()
        plt.show()

    def graph_VACF(self):
        all_vacf_np = np.array(self.all_vacf)
        mean_vacf = np.mean(all_vacf_np, axis=0)

        # Compute the time lags for the x-axis
        num_lags = np.size(mean_vacf)
        time_lags = np.arange(0, num_lags) * self.timestep * self.t_c

        # Plot the mean VACF with labels and legend
        plt.plot(time_lags, mean_vacf, label="Mean VACF")
        plt.xscale("log")
        plt.xlabel('Time Lag')
        plt.ylabel('Velocity Autocorrelation')
        plt.title('Mean Velocity Autocorrelation Function (VACF)')
        plt.legend()
        plt.show()

    def graph_MSD(self):
        all_msd_np = np.array(self.all_msd)
        mean_msd = np.mean(all_msd_np, axis=0)
        # Plot the mean MSD with labels and legend
        plt.plot([t * (self.timestep * self.sample_rate) * self.t_c for t in range(np.size(mean_msd))], mean_msd / self.x_c ** 2, label="Mean MSD")
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Time Lag')
        plt.ylabel('Mean Squared Displacement')
        plt.title('Mean Squared Displacement (MSD)')
        plt.legend()
        plt.show()

    def graph_heat_transfer_variance(self):
        # Flatten and concatenate all delta Q arrays into one
        all_delta_Q_flat = np.hstack(self.all_delta_Q)  # Flatten the list of arrays into a single array

        # Automatically set bin edges based on the cleaned data
        bins = np.histogram_bin_edges(all_delta_Q_flat, bins='fd')  # Use 'fd' or manually set the bins
        total_nan_count = 0

        # Plot each array as an overlapping histogram (raw counts, not normalized)
        for i, dataset in enumerate(self.all_delta_Q):
            # Count NaN values before cleaning
            nan_count = np.isnan(dataset).sum()
            total_nan_count += nan_count

            counts, bin_edges = np.histogram(dataset, bins=bins)
            # Plot the dataset using the bin edges and modified counts
            plt.step(bin_edges[:-1], counts/len(self.all_delta_Q[i]), where='mid', label=f't = {self.heat_transfer_lags[i]} * t_c', alpha=0.7)

        # Set y-axis to log scale (ensures no zero counts)
        plt.yscale('log')

        # Add labels and legend
        plt.xlabel('Heat Exchanged (Q/k_b*t)')
        plt.ylabel('Normalized Counts P(Q, t)')
        plt.title('Probability Density of Exchanged Energy: Silica and Water')
        plt.legend()
        plt.show()


# Initialize the data processor
processor = DataProcessor(df)

# Process all traces
processor.process_traces()

# # Plot the results
# processor.graph_PSD()
# processor.graph_PACF()
processor.graph_VACF()
# processor.graph_MSD()
processor.graph_heat_transfer_variance()
