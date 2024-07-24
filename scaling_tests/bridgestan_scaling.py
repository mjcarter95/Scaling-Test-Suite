import bridgestan as bs

from time import time
from numpy.random import randn
import numpy as np


data_paths = [
    "/mnt/storage/mjc/scaling_test_suite/arma_data_100.json",
    "/mnt/storage/mjc/scaling_test_suite/arma_data_1000.json",
    "/mnt/storage/mjc/scaling_test_suite/arma_data_10000.json",
    "/mnt/storage/mjc/scaling_test_suite/arma_data_100000.json",
    "/mnt/storage/mjc/scaling_test_suite/arma_data_1000000.json",
]

for data_path in data_paths:
    model = bs.StanModel.from_stan_file(
        "/mnt/storage/mjc/scaling_test_suite/arma.stan",
        data_path
    )

    dim = model.param_unc_num()
    x = randn(dim)

    stan_rng = model.new_rng(seed=0)
    x = model.param_constrain(x, rng=stan_rng)

    print(f"Data path: {data_path}")
    x = np.zeros([1000, dim])
    start = time()
    for i in range(1000):
        x[i] = model.log_density(x, propto=True, jacobian=True)
    print(f"Log density: {time() - start}")

    x_grad = np.zeros([1000, dim])
    start = time()
    for i in range(1000):
        model.log_density_gradient(x, propto=True, jacobian=True, out=x_grad[i])
    print(f"Log density gradient: {time() - start}")
    print("--------------------------------")