import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import json

# Parameters
phi = 0.8  # AR
theta = 0.2  # MA
mu = 0.0  # Constant
sigma = 0.3  # Variance
ns = [100, 1000, 10000, 100000, 1000000] # Number of simulations

# Define ARMA model
class ARMAProcess:
    def __init__(self, phi, theta, mu, sigma):
        self.phi = phi
        self.theta = theta
        self.mu = mu
        self.sigma = sigma

    def simulate(self, n):
        ar = np.array([1, -self.phi])
        ma = np.array([1, self.theta])
        arma_process = np.random.normal(self.mu, self.sigma, n)
        return pd.Series(arma_process)

for n in ns:
    # Simulate data
    my_arma = ARMAProcess(phi, theta, mu, sigma)
    y = my_arma.simulate(n)

    # Estimate ARIMA model
    model = ARIMA(y, order=(1, 0, 1))
    model_fit = model.fit()

    # Save to JSON file
    output = {
        "y": y.tolist(),
        "T": len(y)
    }

    file_path = f'arma_data_{len(y)}.json'
    with open(file_path, 'w') as file:
        json.dump(output, file, indent=4)

    print(f"Data saved to {file_path}")
