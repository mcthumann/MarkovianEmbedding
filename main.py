import math

import matplotlib.pyplot as plt

from simulation import *
from analytical import *
from gomez_analytical import *

def run():
    # SIMULATION PARAMETERS
    n = 13 # Number of auxliary stochastic variables
    b = 5 # Scaling dilation parameter that determines the low cutoff frequency ν_0*b^-n
    c = 1.78167 # a prefactor that depends on the particular choice of b
    v_0 = 1e3 # High cutoff frequency (already scaled by tao_c)
    a = 1e-6 # Particle size
    eta = 1e-3 # Viscosity of water
    rho_silica = 2200 # Density of silica
    rho_f = 1000 # Density of water

    # Might be arrays
    mass = (4 / 3) * math.pi * (a / 2.0) ** 3 * rho_silica
    mass_total = mass + .5 * (4 / 3) * math.pi * (a/2.0)** 3 * rho_f # Mass plus added mass

    temp = 273

    lag_fraction = 1
    sample_rate = 1
    simulation_number = 5

    # ANALYTICAL PARAMETERS
    c_water = 1500
    bulk = 2.5E-3
    K = 0  # May need update here

    VSP_length = 1000
    integ_points = 10 ** 4 * 8
    start = -10
    stop = -7
    time_range = (start, stop)
    time_points = 60


    # HELPER PARAMETERS
    timestep = 1E-4  # Simulation timestep
    gamma = 6*math.pi*a*eta # Steady-state friction coefficient of the particle
    tao_c = (mass_total/gamma) # Momentum relaxation time of the particle
    #tao_f = (rho_f * (a ** 2)) / (eta)  # Characteristic time for vorticy diffusion across length of sphere
    tao_f = (9*tao_c/(2*(rho_silica/rho_f)*+1))
    tao_fc = tao_f/tao_c
    print("t_c " + str(tao_c) + " t_f " + str(tao_f) + " t_fc " + str(tao_f/tao_c))
    v_c = math.sqrt((const.k*temp)/mass_total)
    x_c = tao_c*v_c
    f_c = (const.k*temp)/x_c
    v_i = [v_0/(b**(i)) for i in range(1, n+1)] # Decaying samples from an exponential distribution ...
    gamma_i = [(x_c/f_c/tao_c)*(0.5*gamma*c*math.sqrt(tao_fc/math.pi)*(v_i[i]**(3.0/2.0))) for i in range(n)]
    gamma_0 = 0.5*gamma*c*math.sqrt(tao_fc/math.pi)*sum(math.sqrt(v) for v in v_i)
    delta = gamma_0/gamma

    trace_length = int((10**stop)/(timestep*tao_c))

    # Run the analytics
    sol = Analytical_Solution(rho_f, c_water, eta, bulk, a, rho_silica, K, tao_f, mass, mass_total, gamma, temp, VSP_length, integ_points, time_range=time_range, time_points=time_points)
    times, freq, VPSD_cw, VPSD_iw, PSD_iw, PSD_cw, VACF_cw, VACF_iw, PACF_cw, PACF_iw, TPSD_cw, TPSD_iw = sol.calculate()

    # Run the simulation
    mep = MarkovianEmbeddingProcess(n, v_i, gamma_i, delta, timestep, sample_rate, lag_fraction, temp=temp, mass=mass_total, gamma=gamma)
    mep.run_numerical_simulation(simulation_number, trace_length, graph=True, msd=True)

    gom = GomezAnalytical(tao_c, tao_f, start, stop)

    # Graph
    #gom.graph()
    VACF_iw/= (const.k * temp / mass_total)
    mep.graph_VACF(10**start, 10**stop)
    plt.plot(times, VACF_iw, label="Analytical")
    plt.legend()
    plt.title("VACF")
    plt.xscale("log")
    plt.show()

    # Graph
    mep.graph_PACF()
    PACF_iw /= PACF_iw[0]
    plt.plot(times, PACF_iw, label="Analytical")
    plt.legend()
    plt.title("PACF")
    plt.show()

    # Graph
    mep.graph_PSD()
    plt.semilogx(freq, VPSD_iw, label="Analytical")
    plt.legend()
    plt.xlim(1/10**stop, 1/10**start)
    plt.title("PSD")
    plt.show()

    # Graph
    mep.graph_MSD()
    plt.title("MSD")
    plt.show()

    # Find spots where velocity is zero
    index_mult = (timestep * sample_rate) * tao_c
    plt.plot([t * index_mult for t in range(len(mep.all_x))], mep.all_x * x_c, linewidth=0.5)

    tolerance = .001
    close_to_zero_indices = np.where(abs(mep.all_v) < tolerance)[0]
    print("len is " + str(len(close_to_zero_indices)))
    print(close_to_zero_indices)

    plt.title("Positions")
    # Draw vertical lines where values are close to zero
    for index in close_to_zero_indices:
        plt.axvline(x=index*index_mult, color='red', linestyle='-', linewidth=0.1)
    plt.show()

if __name__ == '__main__':
    run()
