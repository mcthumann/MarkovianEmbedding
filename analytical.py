import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.integrate
import pandas as pd
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
import numpy as np
import scipy

class Analytical_Solution:
    def __init__(self, density, c, shear, bulk, a, particle_density, K, tau_f, m, M, gamma_s, T, VSP_length, integ_points, time_range=(-10, -5), time_points=60):
        self.density = density
        self.c = c
        self.shear = shear
        self.bulk = bulk
        self.a = a
        self.particle_density = particle_density
        self.K = K
        self.tau_f = tau_f
        self.m = m
        self.M = M
        self.gamma_s = gamma_s
        self.k_b = scipy.constants.k
        self.T = T
        self.VSP_length = VSP_length
        self.integ_points = integ_points
        self.time_range = time_range
        self.time_points = time_points

        # Automatically calculate frequencies based on times
        self.times, self.frequencies = self.set_times_and_frequencies()

    def set_times_and_frequencies(self):
        # Generate time values logarithmically spaced
        times = np.logspace(self.time_range[0], self.time_range[1], self.time_points)

        # Maximum and minimum times determine the frequency range
        t_min = np.min(times)
        t_max = np.max(times)

        # Maximum and minimum frequencies are the inverse of times
        f_min = 1 / t_max
        f_max = 1 / t_min

        # Generate frequency values logarithmically spaced based on time-derived limits
        frequencies = np.logspace(np.log10(f_min), np.log10(f_max), self.integ_points)

        return times, frequencies

    def gamma(self, omega):
        alpha = np.sqrt(-1j * omega * self.tau_f)
        lambdas = self.a * omega / np.sqrt(-1 * self.c ** 2 + 1j * omega * (self.bulk + 4 / 3 * self.shear) / self.density)
        numerator = (1 + lambdas) * (9 + 9 * alpha + alpha ** 2) + 2 * lambdas ** 2 * (1 + alpha)
        denominator = 2 * (1 + lambdas) + (1 + alpha + alpha ** 2) * lambdas ** 2 / alpha ** 2
        return 4 * np.pi * self.shear * self.a / 3 * numerator / denominator


    def admittance(self, omega):
        return 1 / (-1j * omega * self.m + self.gamma(omega) + self.K / (-1j * omega))


    def incompressible_admittance(self, omega):
        return 1 / (-1j * omega * (self.M) + self.gamma_s * (1 + np.sqrt(-1j * omega * self.tau_f)) + self.K / (-1j * omega))


    def incompressible_gamma(self, omega):
        return self.gamma_s * (1 + np.sqrt(-1j * omega * self.tau_f))


    def velocity_spectral_density(self, omega, admit_func):
        return 2 * self.k_b * self.T * np.real(admit_func(omega))


    def position_spectral_density(self, omega, admit_func):
        return self.velocity_spectral_density(omega, admit_func) / omega ** 2


    def thermal_force_PSD(self, omega, SPD, gamma, mass):
        G = (-1 * omega ** 2 * mass - 1j * omega * gamma + self.K) ** -1
        return np.abs(G) ** -2 * SPD

    def ACF_from_SPD(self, admit_function, SPD_func, times):
        ACF = np.zeros(len(times))

        for i in range(len(times)):
            ACF[i] = 2 * np.real(
                scipy.integrate.simps(SPD_func(self.frequencies, admit_function) * np.exp(-1j * self.frequencies * times[i]),
                                      self.frequencies)) / (2 * np.pi)
        return ACF


    def ACF_from_admit(self, admit_func, times):
        ACF = np.zeros(len(times))
        admit_guy = np.real(admit_func(self.frequencies)) / self.frequencies ** 2
        for i in range(len(times)):
            ACF[i] = scipy.integrate.simps(np.cos(self.frequencies * times[i]) * admit_guy, self.frequencies)

        return ACF


    def thermal_ACF_from_SPD(self, admit_func, tSPD_func, times, SPD_func, gamma, mass):
        low_freq = np.linspace(10 ** -4, 10 ** 4, self.integ_points * 10)
        mid_freq = np.linspace(10 ** 4, 10 ** 6, self.integ_points * 5)
        high_freq = np.linspace(10 ** 6, 10 ** 9, self.integ_points)
        top_freq = np.linspace(10 ** 9, 10 ** 12, self.integ_points)

        # frequencies = np.concatenate((mid_freq, high_freq, top_freq))
        frequencies = low_freq
        ACF = np.zeros(len(times))
        SPD = tSPD_func(frequencies, SPD_func(frequencies, admit_func), gamma(frequencies), mass)
        for i in range(len(times)):
            ACF[i] = 2 * np.real(
                scipy.integrate.simps(SPD * np.exp(-1j * frequencies * times[i] * frequencies) / (2 * np.pi)))

        return ACF


    def mean_square_displacement(self, PACF):
        if self.K == 0:
            return 2 * self.k_b * self.T / self.gamma_s - 2 * PACF
        else:
            return 2 * self.k_b * self.T / self.K - 2 * PACF


    def calculate(self):
        power = np.linspace(0, 10.5, self.VSP_length)
        freq = (np.ones(self.VSP_length) * 10) ** power
        VSPD_compressible = self.velocity_spectral_density(freq, self.admittance)
        VSPD_incompressible = self.velocity_spectral_density(freq, self.incompressible_admittance)
        PSD_incompressible = VSPD_incompressible / freq ** 2
        PSD_compressible = VSPD_compressible / freq ** 2

        TPSD_compressible = self.thermal_force_PSD(freq, PSD_compressible, self.gamma(freq), self.m)
        TPSD_incompressible = self.thermal_force_PSD(freq, PSD_incompressible, self.incompressible_gamma(freq), self.M)

        VACF_compressible = self.ACF_from_SPD(self.admittance, self.velocity_spectral_density, self.times)
        VACF_incompressible = self.ACF_from_SPD(self.incompressible_admittance, self.velocity_spectral_density, self.times)

        PACF_compressible = self.ACF_from_SPD(self.admittance, self.position_spectral_density, self.times)
        PACF_incompressible = self.ACF_from_SPD(self.incompressible_admittance, self.position_spectral_density, self.times)

        MSD_compressible = self.mean_square_displacement(PACF_compressible)
        MSD_incompressible = self.mean_square_displacement(PACF_incompressible)

        confinement = self.K
        if self.K == 0:
            confinement = self.gamma_s

        compress_correction = (self.k_b * self.T / confinement / PACF_compressible[0])
        incompress_correction = (self.k_b * self.T / confinement / PACF_incompressible[0])

        PACF_incompressible *= compress_correction
        PACF_compressible *= incompress_correction

        TPSD_compressible = self.thermal_force_PSD(freq, PSD_compressible, self.gamma(freq), self.m)
        TPSD_incompressible = self.thermal_force_PSD(freq, PSD_incompressible, self.incompressible_gamma(freq), self.M)

        return self.times, freq, VSPD_compressible, VSPD_incompressible, PSD_incompressible, PSD_compressible, VACF_compressible, VACF_incompressible, PACF_compressible, PACF_incompressible, TPSD_compressible, TPSD_incompressible


    def shot_noise_VPSD(omega, sensitivity):
        return sensitivity * omega ** 2


    def cumulative(SD):
        cumulative = np.zeros(len(SD))
        for i in range(len(SD)):
            for j in range(i):
                cumulative[i] += SD[j]

        return cumulative