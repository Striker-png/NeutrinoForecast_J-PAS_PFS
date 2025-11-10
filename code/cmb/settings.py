################### survey specifications and fiducial parameters #####################
import numpy as np
import seaborn as sns

sns.set_context("notebook")
sns.set_palette("husl")

# basic settings
f_sky = 0.09
ell_min = 41
ell_max = 1350
label = "LiteBIRD"
FWHM = np.array([31]) / 60 * np.pi / 180  # rad, 150 GHz
sigma2_T = (np.array([4.1]) / 60 * np.pi / 180) ** 2  # (muK rad)^2
sigma2_P = (np.array([5.8]) / 60 * np.pi / 180) ** 2  # (muK rad)^2

# f_sky = 0.61
# ell_min = 41
# ell_max = 3000
# label = "SO"
# FWHM = np.array([7.4, 5.1, 2.2, 1.4, 1.0, 0.9]) / 60 * np.pi / 180  # rad, 150 GHz
# sigma2_T = (np.array([61, 30, 5.3, 6.6, 15, 35]) / 60 * np.pi / 180) ** 2  # (muK rad)^2
# sigma2_P = 2 * sigma2_T  # (muK rad)^2
k_max_transfer = 10  # 1/Mpc

# accuracy settings
nonlinear = True
halofit_ver = "mead2020"  # 'mead2020': more accurate for massive neutrinos, higher errors expected
AccuracyBoost = 2  # 3
lAccuracyBoost = 2  # 3
lens_accuracy = 2  # 1 is only for Planck-like level
k_per_logint = 100  # 50
# CAMB/fortran/halofit.f90 tolerance=1e-6

# constants
nu_mass_num = 1

# fiducial values
omega_m0 = 0.32
omega_b0 = 0.049
h = 0.67
n_s = 0.96
sigma_8 = 0.81
m_nu = 0.06
N_eff = 3.044
tau = 0.054
w0 = -1
wa = 0

# variables
var_name = [
    "omega_m0",
    "omega_b0",
    "h",
    "n_s",
    "sigma_8",
    "m_nu",
    "N_eff",
    "tau",
    "w0",
    "wa",
]
var_exp = [
    r"$\Omega_\mathrm{m,0}$",
    r"$\Omega_\mathrm{b,0}$",
    r"$h$",
    r"$n_\mathrm{s}$",
    r"$\sigma_8$",
    r"$\sum m_\nu$",
    r"$N_\mathrm{eff}$",
    r"$\tau$",
    r"$w_0$",
    r"$w_a$",
]
var_num = len(var_name)
cosmo_value = [omega_m0, omega_b0, h, n_s, sigma_8, m_nu, N_eff, tau, w0, wa]
cosmo_num = len(cosmo_value)
powers = 0
