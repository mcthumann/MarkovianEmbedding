import math
import pickle
from simulation import *
from analytical import *

# "100 independent trajectories starting from the abovementioned initial conditions are simulated with a
# time-step ∆t = 10−4 τc, a total duration of 103 τc and then sampled at a frequency 103τ−1c thus amounting
# to 10^8 data points over which all the average quantities characterizing the dynamics of the system
# are computed"

def run():
    # SIMULATION PARAMETERS
    n = 13 # Number of auxliary stochastic variables
    b = 5 # Scaling dilation parameter that determines the low cutoff frequency ν_0*b^-n
    c = 1.78167 # a prefactor that depends on the particular choice of b
    v_0 = 1e3 # High cutoff frequency (already scaled by tao_c)
    a = 4.464e-6 # Particle size
    eta = 0.36e-3 # ACETONE 1e-3 # Viscosity of water
    rho_bati = 4400
    rho_silica = 2200 # Density of silica
    rho_acetone = 789
    rho_f = 1000 # Density of water

    mass = (4 / 3) * math.pi * a** 3 * rho_bati
    mass_total = mass + .5 * (4 / 3) * math.pi * a**3 * rho_acetone # Mass plus added mass

    temp = 293
    K = 1 # Does K do anything?

    lag_fraction = 1
    sample_rate = 1
    simulation_number = 3

    # ANALYTICAL PARAMETERS
    c_water = 1500
    c_acetone = 1174
    bulk_acetone = 1.4e-3
    # bulk = 2.5E-3

    VSP_length = 1000
    integ_points = 10 ** 4 * 8
    start = -10
    stop = -4
    time_range = (start, stop)
    time_points = 600

    # HELPER PARAMETERS
    timestep = 1E-4  # Simulation timestep
    gamma = 6*math.pi*a*eta # Steady-state friction coefficient of the particle
    tao_c = (mass_total/gamma) # Momentum relaxation time of the particle
    #tao_f = (rho_acetone * (a ** 2)) / (eta)  # Characteristic time for vorticy diffusion across length of sphere
    tao_f = (9*tao_c/(2*(rho_bati/rho_acetone)*+1))
    tao_fc = tao_f/tao_c
    v_c = math.sqrt((const.k*temp)/mass_total)
    x_c = tao_c*v_c
    f_c = (const.k*temp)/x_c
    v_i = [v_0/(b**(i)) for i in range(1, n+1)] # Decaying samples from an exponential distribution ...
    gamma_i = [(x_c/f_c/tao_c)*(0.5*gamma*c*math.sqrt(tao_fc/math.pi)*(v_i[i]**(3.0/2.0))) for i in range(n)]
    gamma_0 = 0.5*gamma*c*math.sqrt(tao_fc/math.pi)*sum(math.sqrt(v) for v in v_i)
    delta = gamma_0/gamma

    trace_length = int((10**stop)/(timestep*tao_c))

    df = pd.DataFrame({
        'CreationDate': [pd.Timestamp.now()],
        'a': [a],
        'eta': [eta],
        'rho_bati': [rho_bati],
        'rho_f': [rho_acetone],
        'sampling_rate': [(1.0/timestep)],
        'stop': [stop],
        'start': [start],
        'track_len': [trace_length],
        'sample_rate': [sample_rate],
        'tao_c': [tao_c],
        'v_c': [v_c],
        'x_c': [x_c]
    })

    # Run the analytics
    sol = Analytical_Solution(rho_acetone, c_acetone, eta, bulk_acetone, a, rho_bati, K, tao_f, mass, mass_total, gamma, temp, VSP_length, integ_points, time_range=time_range, time_points=time_points, sample_rate=sample_rate)
    times, freq, VPSD_iw, PSD_iw, VACF_iw, PACF_iw, TPSD_iw = sol.calculate()

    pacf = True
    vacf = True
    msd = True
    psd = True
    save = True

    # Run the simulation
    mep = MarkovianEmbeddingProcess(n, v_i, gamma_i, delta, timestep, sample_rate, lag_fraction, K=K, temp=temp, mass=mass_total, gamma=gamma)
    mep.run_numerical_simulation(simulation_number, trace_length, pacf, vacf, msd, psd, df, graph=False, save=save)

    # Graph
    if vacf:
        vacf_s = sol.standalone_vacf(times)
        mep.graph_VACF(10**start, 10**stop)
        plt.plot(times, VACF_iw, label="Analytical")
        plt.plot(times, vacf_s, label="Standalone Analytical")
        plt.legend()
        plt.title("VACF")
        plt.xscale("log")
        plt.xlim(left = 1e-7, right = 1e-5)
        plt.show()

    if pacf:
        mep.graph_PACF()
        PACF_iw /= PACF_iw[0]
        plt.plot(times, PACF_iw, label="Analytical")
        plt.legend()
        plt.title("PACF")
        plt.show()

    if psd:
        mep.graph_PSD()
        plt.semilogx(freq, PSD_iw, label="Analytical")
        plt.legend()
        plt.xlim(1/10**stop, 1/10**start)
        plt.title("PSD")
        plt.show()

    if msd:
        mep.graph_MSD()
        plt.title("MSD")
        plt.show()
    #
    # # Find spots where velocity is zero
    # index_mult = (timestep * sample_rate) * tao_c
    # plt.plot([t * index_mult for t in range(len(mep.all_x))], mep.all_x * x_c, linewidth=0.5)
    # print(mep.all_x)
    # tolerance = (np.std(mep.all_v))
    # close_to_zero_indices = np.where(abs(mep.all_v - np.mean(mep.all_v)) < tolerance)[0]
    # print("zero idx: " + str(close_to_zero_indices))
    # plt.title("Positions")
    # # Draw vertical lines where values are close to zero
    # for index in close_to_zero_indices:
    #     plt.axvline(x=index*index_mult, color='red', linestyle='-', linewidth=0.1)

    plt.show()

if __name__ == '__main__':
    run()
