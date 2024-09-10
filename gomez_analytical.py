import numpy as np
from scipy.special import erfc
import matplotlib.pyplot as plt
import math
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
import scipy.constants as const


class GomezAnalytical:

    def __init__(self, tau_c, tau_f, start, stop):
        self.tau_c = tau_c
        self.tau_f = tau_f
        self.t_c = 1
        self.t_f = tau_f/tau_c
        self.start = float(start)
        self.stop = float(stop)

    def psi(self, tau, a_plus, a_minus):
        """
        Compute psi(τ) for each value in the log-spaced array of tau.

        Parameters:
        - tau: array of time values (log-spaced)
        - a_plus: constant a_+
        - a_minus: constant a_-

        Returns:
        - Array of psi(τ) values
        """
        psi_tau = (1 / (a_plus - a_minus)) * (
            a_plus * np.exp(a_plus**2 * tau) * erfc(a_plus * np.sqrt(tau)) -
            a_minus * np.exp(a_minus**2 * tau) * erfc(a_minus * np.sqrt(tau))
        )
        return psi_tau

    # Corrected function to compute a_+ and a_- (based on the paper's equation 48)
    def compute_a_plus_minus(self):
        """
        Calculate a_+ and a_- from time constants τc and τf (Eq. 48).
        """
        discriminant = np.sqrt(1 - 4 * (self.t_c / self.t_f))
        a_plus = (1 / (2 * np.sqrt(self.t_f))) * (1 + discriminant) / (self.t_c/self.t_f)
        a_minus = (1 / (2 * np.sqrt(self.t_f))) * (1 - discriminant) /(self.t_c/self.t_f)
        return a_plus, a_minus

    def graph(self):
        tau_start = np.power(10, self.start) / self.tau_c
        tau_stop = np.power(10, self.stop) / self.tau_c
        tau = np.logspace(np.log10(tau_start), np.log10(tau_stop), 100)

        # Calculate a_+ and a_-
        a_plus, a_minus = self.compute_a_plus_minus()

        # Calculate psi(τ) for the log-spaced array of τ
        psi_values = self.psi(tau, a_plus, a_minus)

        # Plotting the results
        plt.figure()
        plt.plot(tau, psi_values)
