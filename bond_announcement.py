import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

import matrix_wise_ajd

# TODO - fix plots

# ---
# PART # 1 - Simulation
# simulate data for three factors
# ---

# set the seed to generate a certain set of random numbers
np.random.seed(1)

# number of observations
nobs        = 1000

# number of factors which characterize the instantaneous interest rate
dim         = 3

# bond maturities
tau         = np.array([1/12,6/12,12/12,36/12,60/12,120/12])
dt          = 1

# Define parameters for factor dynamics

# kappa (alpha) is the speed of mean reversion, is lower triangular
k_11        = .170
k_21        = -.2318
k_31        = -.0294
k_22        = 1.0373
k_32        = -1.9081
k_33        = .6713
K_P         = np.matrix([[k_11, 0, 0], [k_21, k_22, 0], [k_31, k_32, k_33]])

# theta is the long run mean of each state
theta_P     = np.zeros([dim, 1])

# sigma
c           = .1
sigma_31    = .0012
sigma_32    = -.0043
sigma_33    = -.0073
sigma       = np.matrix(([c, 0., 0.], [0., c, 0.], [sigma_31, sigma_32, sigma_33]))

# variance
sigma2      = sigma.dot(sigma.T)

# Initial factor values
x           = np.zeros([nobs,dim])
x[0,:]      = np.matrix([0., 0., 0.])

# Simulation
for t in range(1,nobs):
    error_terms = np.random.normal(0,1,dim).reshape(dim, 1)
    x[t, :]     = (x[t-1, :].reshape(dim,1) + K_P.dot(theta_P-x[t-1].reshape(dim,1)) * dt + sigma * error_terms).reshape(1, dim)

plt.figure( figsize = (15,8) )
plt.plot(x[:,0], label='factor 1')
plt.plot(x[:,1], label='factor 2')
plt.plot(x[:,2], label='factor 3')
plt.xlabel("Number of Observations", fontdict=None, labelpad=None)
plt.legend(loc='upper left', ncol=1)
plt.show()

# ---
# PART # 2 - Transformation
# get Q-Parameters from P-Parameters
# ---

sigma_lam    = np.matrix([[-0.1703, -0.3564, 0.2796], [0.3357, -0.6992, -0.2888], [-0.7580 , -0.7048, 0.5355]])
lam          = np.matrix([[0.0635],[-1.0458],[-4.9772]])

K_Q          = K_P + sigma_lam
theta_Q      = np.linalg.inv(K_Q).dot(K_P.dot(theta_P - sigma.dot(lam)))

# ---
# PART # 3 - Yields
# get yields
# ---

# constant part for interest rate
rho_0 = .0
# loading of interest rate. In the setting of following
rho = np.matrix(([0., 0., 1.])).T

A   = np.zeros([len(tau),1])
B   = np.zeros([len(tau),dim])

JW_NoJump       = matrix_wise_ajd.cf_matirix_wise_ajd(dim=dim, rho_0=rho_0, rho=rho, K_Q=K_Q, theta_Q=theta_Q, Sigma=sigma)
B_0             = np.matrix(([0., 0., 0.])).T

for t in range (1, len(tau)):
    A_tau, B_tau    = JW_NoJump.A_B(tau[t], B_0)
    A[t]            = A_tau
    B[t]            = B_tau.T

# Calculate Prices
prices = np.zeros([nobs,len(tau)])
for m in range (1, len(tau)):
    for t in range(1,nobs):
        prices[t, m] = np.exp(A[m].dot(B[m].dot(x[t])))

plt.figure()
plt.plot(prices[:,5], label='10 year')
plt.plot(prices[:,4], label='5 year')
plt.plot(prices[:,3], label='3 year')
plt.plot(prices[:,2], label='1 year')
plt.plot(prices[:,1], label='6 month')
plt.plot(prices[:,0], label='1 month')
plt.xlabel("Number of Observations", fontdict=None, labelpad=None)
plt.xlabel("Price", fontdict=None, labelpad=None)
plt.legend(loc='upper left', ncol=7)
plt.show()

# Calculate yields and add measurement error
y_error = 0.0001
yields  =  (x[:, 0].reshape(nobs, 1).dot(B[:, 0].reshape(1, len(tau))) + x[:, 1].reshape(nobs, 1).dot(B[:, 1].reshape(1, len(tau))) + x[:, 2].reshape(nobs, 1).dot(B[:, 2].reshape(1, len(tau))) - A.T) / tau + y_error * np.random.normal(0, 1, [nobs, len(tau)])

plt.figure(1)
plt.plot(yields[:,5], label='10 year', color='0.00')
plt.plot(yields[:,4], label='5 year', color='0.20')
plt.plot(yields[:,3], label='3 year', color='0.40')
plt.plot(yields[:,2], label='1 year', color='0.60')
plt.plot(yields[:,1], label='6 month',color='0.80')
plt.plot(yields[:,0], label='1 month',color='1.00')
plt.xlabel("Number of Observations", fontdict=None, labelpad=None)
plt.xlabel("Yield", fontdict=None, labelpad=None)
plt.legend(loc='upper left', ncol=7)