

# %% [markdown]
# # Real-Time Optimization
# ## Modifier Adaptation with Bayesian Optimization using EIC acquisition
# ### Preliminary thesis results generation

# %%
# Loading the necessary packages
import numpy as np
import pandas as pd
import logging
from sklearn.utils import Bunch
logging.basicConfig(level=logging.DEBUG)

from create_database import create_database
from rto.models.semi_batch import SemiBatchReactor
from rto.optimization.optimizer import ModifierAdaptationOptimizer, ModelBasedOptimizer
from rto.optimization.bayesian import ModelBasedBayesianOptimizer
from rto.rto import RTOBayesian, RTO
from rto.adaptation.ma_gaussian_processes import MAGaussianProcesses
from rto.utils import generate_samples_uniform

# backup script
## 05: inclusão do NM simplex na comparação e amostrando proximo do otimo do modelo
## 06: inclusão do NM simplex na comparação e amostrando proximo do ponto inicial do artigo
## 07: 06 + retirando restarts do treino do modelo GP
DATABASE = "/home/victor/git/rto-data/thesis-results-rap.db"
#create_database(DATABASE)

# %%
# Our complete model will be called the "plant"
plant = SemiBatchReactor()
# And the uncertain is the "model"
model = SemiBatchReactor(k=[0.053, 0.128, 0.0, 0.0, 5])
# define the constraints
g0 = 0.025
g1 = 0.15
g = np.array([g0, g1])
ubx = [30, 0.002, 250]
lbx = [0, 0, 200]


# %%
# optimizer = ModelBasedOptimizer(ub=ubx, lb=lbx, g=g)
# f_plant, u_plant ,_ = optimizer.run(plant, [])
# f_model, u_model ,_ = optimizer.run(model, [])

# print(f'Plant: u*={u_plant}, f*={f_plant}')
# print(f'Model: u*={u_model}, f*={f_model}')

# %% [markdown]
# ## Real-Time Optimization
# %% [markdown]
# ### Modifier Adaptation with Gaussian Processes

# %%
# Define the system parameters
u_0 = [10.652103265931729, 0.0005141834799295323, 224.48063936756103] # use the same from the paper?
iterations = 60
initial_data_size = 5

# sample some initial data
u, y, measurements = generate_samples_uniform(model, plant, g, u_0, initial_data_size, noise=0.0)
initial_data = Bunch(u=u, y=y, measurements=measurements)
u_0_feas = u[-1]


# %%
# create the adaptation strategy
adaptation_de = MAGaussianProcesses(model, initial_data, ub=ubx, lb=lbx, filter_data=True, neighbors_type='k_last')
adaptation_sqp = MAGaussianProcesses(model, initial_data, ub=ubx, lb=lbx, filter_data=True, neighbors_type='k_last')
# create the optmizer instances
optimizer_ma_de = ModifierAdaptationOptimizer(ub=ubx, lb=lbx, g=g, solver={'name': 'de'}, backoff=0.0)
optimizer_ma_sqp = ModifierAdaptationOptimizer(ub=ubx, lb=lbx, g=g, solver={'name': 'sqp'}, backoff=0.0)

rto_ma_sqp = RTO(model, plant, optimizer_ma_sqp, adaptation_sqp, iterations, db_file=DATABASE, name='MA-GP-SQP', noise=0.0)
rto_ma_de = RTO(model, plant, optimizer_ma_de, adaptation_de, iterations, db_file=DATABASE, name='MA-GP-DE', noise=0.0)

rto_ma_sqp.run(u_0_feas)
rto_ma_de.run(u_0_feas)

# %% [markdown]
# ### Effect of Noise
# 
# The results above are for the scenario where we have no noise in the plant measuremets. Since this is not the reality, an interesting test is to check how it can impact the RTO performance. For that, we consider a 0.01 additive gaussian noise, but using the same parameters as the previous system.

# %%
# noise = 0.01
# repetitions = 10

# adaptation_de_noise = MAGaussianProcesses(model, initial_data, ub=ubx, lb=lbx, filter_data=True, neighbors_type='k_last')
# adaptation_sqp_noise = MAGaussianProcesses(model, initial_data, ub=ubx, lb=lbx, filter_data=True, neighbors_type='k_last')

# rto_ma_sqp_noise = RTO(model, plant, optimizer_ma_sqp, adaptation_sqp_noise, iterations, db_file=DATABASE, name='MA-GP-SQP+noise', noise=noise)
# rto_ma_de_noise = RTO(model, plant, optimizer_ma_de, adaptation_de_noise, iterations, db_file=DATABASE, name='MA-GP-DE+noise', noise=noise)

# for i in range(repetitions):
#     rto_ma_sqp_noise.run(u_0_feas)
#     rto_ma_de_noise.run(u_0_feas)

# %% [markdown]
# ### Using different initial data points

# %%
# sample some initial data
# generate all the data before
n_datasets = 30
noise = 0.01
datasets = []
for i in range(n_datasets):
    u_i, y_i, measurements_i = generate_samples_uniform(model, plant, g, u_0, initial_data_size, noise=noise)
    initial_dataset = Bunch(u=u_i, y=y_i, measurements=measurements_i)
    datasets.append(initial_dataset)


# %%
for i in range(n_datasets):
    initial_dataset = datasets[i]
    # create the adaptation strategy
    adaptation_de_noise_dataset = MAGaussianProcesses(model, initial_dataset, ub=ubx, lb=lbx, filter_data=True, neighbors_type='k_last')
    adaptation_sqp_noise_dataset = MAGaussianProcesses(model, initial_dataset, ub=ubx, lb=lbx, filter_data=True, neighbors_type='k_last')
    # create the RTO
    rto_ma_sqp_noise_dataset = RTO(model, plant, optimizer_ma_sqp, adaptation_sqp_noise_dataset, iterations, db_file=DATABASE, name='MA-GP-SQP+noise-datasets', noise=noise)
    rto_ma_de_noise_dataset = RTO(model, plant, optimizer_ma_de, adaptation_de_noise_dataset, iterations, db_file=DATABASE, name='MA-GP-DE+noise-datasets', noise=noise)
    # run the RTO
    u_0_feas_noise = initial_dataset.u[-1]
    rto_ma_sqp_noise_dataset.run(u_0_feas_noise)
    rto_ma_de_noise_dataset.run(u_0_feas_noise)

# %% [markdown]
# ## EIC acquisition function
# %% [markdown]
# ### Optimizer Choice
# Using the same initial data as MA-GP without noise.

# %%
adaptation_bay_de = MAGaussianProcesses(model, initial_data, ub=ubx, lb=lbx, filter_data=True, neighbors_type='k_last')
adaptation_bay_sqp = MAGaussianProcesses(model, initial_data, ub=ubx, lb=lbx, filter_data=True, neighbors_type='k_last')
adaptation_bay_nm = MAGaussianProcesses(model, initial_data, ub=ubx, lb=lbx, filter_data=True, neighbors_type='k_last')

optimizer_bay_de = ModelBasedBayesianOptimizer(ub=ubx, lb=lbx, g=g, solver={'name': 'de'}, backoff=0.0)
optimizer_bay_sqp = ModelBasedBayesianOptimizer(ub=ubx, lb=lbx, g=g, solver={'name': 'sqp'}, backoff=0.0)
optimizer_bay_nm = ModelBasedBayesianOptimizer(ub=ubx, lb=lbx, g=g, solver={'name': 'nm'}, backoff=0.0)

rto_bay_de = RTOBayesian(model, plant, optimizer_bay_de, adaptation_bay_de, iterations, db_file=DATABASE, name='MA-GP-Bayesian-DE', noise=0.0)
rto_bay_sqp = RTOBayesian(model, plant, optimizer_bay_sqp, adaptation_bay_sqp, iterations, db_file=DATABASE, name='MA-GP-Bayesian-SQP', noise=0.0)
rto_bay_nm = RTOBayesian(model, plant, optimizer_bay_nm, adaptation_bay_nm, iterations, db_file=DATABASE, name='MA-GP-Bayesian-NM', noise=0.0)

u_0_feas = u[-1]
rto_bay_sqp.run(u_0_feas)
rto_bay_de.run(u_0_feas)
rto_bay_nm.run(u_0_feas)

# %% [markdown]
# ### Different initial points with noise

# %%
for i in range(n_datasets):
    initial_dataset = datasets[i]
    adaptation_bay_de_noise = MAGaussianProcesses(model, initial_dataset, ub=ubx, lb=lbx, filter_data=True, neighbors_type='k_last')
    rto_bay_de_noise = RTOBayesian(model, plant, optimizer_bay_de, adaptation_bay_de_noise, iterations, db_file=DATABASE, name='MA-GP-Bayesian-DE+noise-datasets', noise=noise)

    adaptation_bay_nm_noise = MAGaussianProcesses(model, initial_dataset, ub=ubx, lb=lbx, filter_data=True, neighbors_type='k_last')
    rto_bay_nm_noise = RTOBayesian(model, plant, optimizer_bay_nm, adaptation_bay_nm_noise, iterations, db_file=DATABASE, name='MA-GP-Bayesian-NM+noise-datasets', noise=noise)
    
    rto_bay_de_noise.run(initial_dataset.u[-1])
    rto_bay_nm_noise.run(initial_dataset.u[-1])


