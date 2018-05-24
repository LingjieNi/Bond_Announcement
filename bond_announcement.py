# Import packages for econometric analysis
import numpy as np

# Load plotting library
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

# ---
# PART # 1 - Simulation
# simulate data for three factors
# ---

# Set the seed to generate a certain set of random numbers
np.random.seed(12)
nobs        = 1000

# monthly observations, 1 equals one year
tau         = np.array([1/12,6/12,12/12,36/12,60/12,120/12])
dt          = 1

# kappa is the speed of mean reversion, is lower triangular
k_11        = .170
k_21        = -.2318
k_31        = -.0294
k_22        = 1.0373
k_32        = -1.9081
k_33        = .6713
K_P         = np.matrix([[k_11, 0, 0], [k_21, k_22, 0], [k_31, k_32, k_33]])

# theta is the long run mean of each state
thetaP      = np.array([0,0,0])

# sigma
c           = .1
sigma_31    = .0012
sigma_32    = -.0043
sigma_33    = -.0073
sigma       = np.matrix(([c, 0., 0.], [0., c, 0.], [sigma_31, sigma_32, sigma_33]))

# Initial factor values
# TODO - chose proper initial values
k      = 3
x      = np.empty([nobs,k])
x[0,:] = np.array([0.0042, 0.0021, 0.0028])

# Constant unconditional variance
sigma2 = (sigma**2).dot(np.linalg.inv(K_P) * (1 - np.exp(- 2 * K_P)))  # constant volatility

# Simulation
for t in range(1,nobs):
    er = np.array([np.random.normal(0,1,1), np.random.normal(0,1,1), np.random.normal(0,1,1)])
    x[t] = x[t-1] + K_P.dot(thetaP-x[t-1]) * dt + sigma * np.sqrt(dt) * er

plt.figure(2)
plt.plot(x, label='state variables',  color='#1E90FF')
plt.legend(loc='upper left', ncol=1)

# ---
# PART # 2 - Transformation
# get Q Parameters
# ---

# Transform values form P-Density to Q-Density
trans_values    = np.matrix([[-0.1703, -0.3564, 0.2796], [0.3357, -0.6992, -0.2888], [-0.7580 , -0.7048, 0.5355]])
trans_lambda    = np.matrix([[0.0635],[-1.0458],[-4.9772]])

K_Q             = K_P + trans_values
thetaQ          = np.linalg.inv(K_Q).dot(K_P.dot(thetaP - sigma.dot(trans_lambda)))

# ---
# PART # 3 - Yields
# get yields
# ---

# Calculate Vasicek implied bond loadings
Btau = ( np.linalg.inv(K_Q).dot(1 - np.exp( - K_Q * tau)))

Atau = ((thetaQ - sigma * sigma / alphaQ / alphaQ / 2)*(Btau - tau) - sigma * sigma * np.square(Btau) / (4 * alphaQ))
yields = np.zeros([1000,6])
for i in range(0,6):
    for t in range(0,1000):
        yields[t,i] = (riskfree[t]*Btau[i]/tau[i] - Atau[i]/tau[i])

yields = yields + np.random.normal(0, 0.0001, [nobs, len(tau)])

plt.figure(1)
plt.plot(yields[:,5], label='10 year', color='0.00')
plt.plot(yields[:,4], label='5 year', color='0.20')
plt.plot(yields[:,3], label='3 year', color='0.40')
plt.plot(yields[:,2], label='1 year', color='0.60')
plt.plot(yields[:,1], label='6 month',color='0.80')
plt.plot(yields[:,0], label='1 month',color='1.00')
plt.plot(riskfree, label='inst. rate',  color='#1E90FF')
plt.legend(loc='upper left', ncol=7)


# ---
# PART - 2
# Calculate Price
# ---