from simulation import *
from analytical import *

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
    m = (4 / 3) * math.pi * (a / 2.0) ** 3 * rho_silica
    M = m + .5 * (4 / 3) * math.pi * (a/2.0)** 3 * rho_f # Mass plus added mass
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

    lag_fraction = 0.1
    sample_rate = 1
    simulation_number = 1
    trace_length = 10**6

    acf_time = timestep*sample_rate*trace_length*lag_fraction

    # Analytical Params
    c_water = 1500
    bulk = 2.5E-3
    K = 1E-6  # May need update here
    k_b = scipy.constants.k
    VSP_length = 1000
    integ_points = 10 ** 4 * 8
    times = np.logspace(-10, -5, 60)

    sol = Analytical_Solution(rho_f, c_water, eta, bulk, a, rho_silica, K, tao_f, m, M, gamma, k_b, temp, VSP_length, integ_points, times)
    times, freq, VSPD_cw, VSPD_iw, PSD_iw, PSD_cw, VACF_cw, VACF_iw, PACF_cw, PACF_iw, TPSD_cw, TPSD_iw = sol.calculate()

    mep = MarkovianEmbeddingProcess(n, v_i, gamma_i, delta, timestep, sample_rate=sample_rate, temp=temp, mass=M, gamma=gamma)
    mep.run_numerical_simulation(simulation_number, trace_length, graph=False, msd=False)

    #mep.graph_VACF()
    plt.semilogx(times, VACF_iw)
    plt.show()

if __name__ == '__main__':
    run()
