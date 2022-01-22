#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from sklearn.utils import Bunch
from os.path import join
import seaborn as sns

# plt.rcParams['axes.grid'] = True
# plt.rcParams['grid.linestyle'] = "dotted"
plt.rcParams["savefig.dpi"] = 100
plt.rcParams['lines.linewidth'] = 3

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('lines', markersize=SMALL_SIZE)

logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.ticker').disabled = True
logging.getLogger('matplotlib.colorbar').disabled = True
logging.basicConfig(level=logging.ERROR)

from rto.models.williams_otto import WilliamsOttoReactor, WilliamsOttoReactorSimplified
from rto.optimization.optimizer import ModelBasedOptimizer
from rto.models.semi_batch import SemiBatchReactor
from rto.optimization.bayesian import ModelBasedBayesianOptimizer
from rto.experiment.analysis import ExperimentAnalyzer

#%% CONSTANTS
MODEL_NAME = 'wo'
FIGURES_PATH = join("/mnt/d/textos-mestrado/dissertacao/figures", MODEL_NAME)
DATABASE = f"/mnt/d/rto_data/final-results-{MODEL_NAME}.db"

if(MODEL_NAME == 'wo'):
    # Our complete model will be called the "plant"
    plant = WilliamsOttoReactor()
    # And the uncertain is the "model"
    model = WilliamsOttoReactorSimplified()
    g0 = 0.12
    g1 = 0.08
    ubx = [6, 100]
    lbx = [3, 70]
    g = np.array([g0, g1])

    optimizer = ModelBasedOptimizer(ub=ubx, lb=lbx, g=g, solver={'name': 'de', 'params': {'strategy': 'best1bin'}}, backoff=0.0)
    f_plant, u_plant ,_ = optimizer.run(plant, [])
    f_model, u_model ,_ = optimizer.run(model, [])
    g_plant = plant.get_constraints(u_plant)

elif(MODEL_NAME == 'rap'):
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

    f_plant = -0.5085930760109818
    u_plant = [18.4427644, 0.00110823777, 227.792418]
    g_plant = plant.get_constraints(u_plant)

print(f'Plant: u*={u_plant}, f*={f_plant}, g*={g_plant}')
# %%
def plot_by_iteration(data, y, ylabel, title='', style=None, hue='run.status', xlabel='Iteration'):
        fig, ax = plt.subplots(figsize=(8, 6))
        results_palette = {
            'SQP/MA-GP': '#377eb8',
            'DE/MA-GP': '#e41a1c',
            'DE/MA-GP-EIC': '#4daf4a',
            'SQP/MA-GP-EIC': '#984ea3',
            'NM/MA-GP-EIC': '#ff7f00',
        } 
        
        # results_dashes = {
        #     'SQP/MA-GP': '',
        #     'DE/MA-GP': (4, 1.5),
        #     'DE/MA-GP-EIC': (1, 1),
        #     'SQP/MA-GP-EIC': (3, 1.25, 1.5, 1.25),
        #     'NM/MA-GP-EIC': (5, 1, 1, 1),
        # } 

        results_dashes = {
            'SQP/MA-GP': '',
            'DE/MA-GP': '',
            'DE/MA-GP-EIC': '',
            'SQP/MA-GP-EIC': '',
            'NM/MA-GP-EIC': '',
        } 

        results_markers = {
            'SQP/MA-GP': 'X',
            'DE/MA-GP': 's',
            'DE/MA-GP-EIC': 'd',
            'SQP/MA-GP-EIC': '*',
            'NM/MA-GP-EIC': 'o',
        }

        # [',', '.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
        

        # Set1
        #e41a1c
        #377eb8
        #4daf4a
        #984ea3 
        #ff7f00
        #ffff33
        #a65628
        #f781bf
        #999999

        g = sns.lineplot(data=data, y=y, x='iteration', hue=hue, style=style, ax=ax, palette=results_palette, dashes=results_dashes, markers=results_markers, seed=1234, legend=True, markersize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.get_legend().set_title('')
        ax.set_title(title)

        # handles, labels = ax.get_legend_handles_labels()

        # for lh in handles: 
        #     lh._sizes = [100] 
        
        # ax.legend(handles, labels)
        fig.tight_layout()
        
        ax.minorticks_on()
        #ax.xaxis.set_tick_params(which='minor', bottom=False)
        ax.grid(b=True, which='minor', color='gray', linestyle='dotted')
        ax.grid(b=True, which='major', color='gray', linestyle='-')

        # for xmaj in ax.xaxis.get_majorticklocs():
        #     ax.axvline(x=xmaj, ls='-', color='gray', alpha=0.5)
        # # for xmin in ax.xaxis.get_minorticklocs():
        # #     ax.axvline(x=xmin, ls='--')

        # for ymaj in ax.yaxis.get_majorticklocs():
        #     ax.axhline(y=ymaj, ls='-', color='gray', alpha=0.5)
        # for ymin in ax.yaxis.get_minorticklocs():
        #     ax.axhline(y=ymin, ls='.', color='gray', alpha=0.5)
        
        return ax, fig

# %%
# load the results
analyzer = ExperimentAnalyzer(DATABASE)
results_ma_de = analyzer.load('MA-GP-DE')
results_ma_sqp = analyzer.load('MA-GP-SQP')

results_processed_ma_de = analyzer.pre_process(results_ma_de, f_plant, u_plant, g_plant)
results_processed_ma_sqp = analyzer.pre_process(results_ma_sqp, f_plant, u_plant, g_plant)

results_processed_ma_de['Cenário'] = 'DE/MA-GP'
results_processed_ma_sqp['Cenário'] = 'SQP/MA-GP'

results_processed_ma_de['Amostra'] = results_processed_ma_de['run.status'].map(lambda x: 'Inicialização' if x=='initialization' else 'Malha-fechada')
results_processed_ma_sqp['Amostra'] = results_processed_ma_sqp['run.status'].map(lambda x: 'Inicialização' if x=='initialization' else 'Malha-fechada')

results_magp_validation = pd.concat([results_processed_ma_de, results_processed_ma_sqp], ignore_index=True)


# %% VALIDATION - MA-GP
_, figa = plot_by_iteration(results_magp_validation, 'dPhi', '$\Delta G_0$ (%)', 'Gap de otimalidade (objetivo)',style='Cenário', hue='Cenário', xlabel='Iteração')
_, figb = plot_by_iteration(results_magp_validation, 'du', '$\Delta \mathbf{u}$ (%)', 'Gap de otimalidade (entradas)',style='Cenário', hue='Cenário', xlabel='Iteração')

if FIGURES_PATH is not None:
    figa.savefig(join(FIGURES_PATH, 'magp_validation_gap_obj.png'))
    figb.savefig(join(FIGURES_PATH, 'magp_validation_gap_u.png'))

# %% FULL EXPERIMENT - MA-GP
# load the results
results_ma_de_noise_ds = analyzer.load('MA-GP-DE+noise-datasets')
results_ma_sqp_noise_ds = analyzer.load('MA-GP-SQP+noise-datasets')

results_processed_ma_de_noise_df = analyzer.pre_process(results_ma_de_noise_ds, f_plant, u_plant, g_plant)
results_processed_ma_sqp_noise_df = analyzer.pre_process(results_ma_sqp_noise_ds, f_plant, u_plant, g_plant)

results_processed_ma_de_noise_df['Cenário'] = 'DE/MA-GP'
results_processed_ma_sqp_noise_df['Cenário'] = 'SQP/MA-GP'

results_processed_ma_de_noise_df['Amostra'] = results_processed_ma_de_noise_df['run.status'].map(lambda x: 'Inicialização' if x=='initialization' else 'Malha-fechada')
results_processed_ma_sqp_noise_df['Amostra'] = results_processed_ma_sqp_noise_df['run.status'].map(lambda x: 'Inicialização' if x=='initialization' else 'Malha-fechada')

results_magp = pd.concat([results_processed_ma_de_noise_df, results_processed_ma_sqp_noise_df], ignore_index=True)

_, figa = plot_by_iteration(results_magp, 'dPhi', '$\Delta G_0$ (%)', '',style='Cenário', hue='Cenário', xlabel='Iteração')
_, figb = plot_by_iteration(results_magp, 'du', '$\Delta \mathbf{u}$ (%)', '',style='Cenário', hue='Cenário', xlabel='Iteração')
_, figc = plot_by_iteration(results_magp, 'dBest', '$\Delta G_0^{\star}$ (%)', '',style='Cenário', hue='Cenário', xlabel='Iteração')

if FIGURES_PATH is not None:
    figa.savefig(join(FIGURES_PATH, 'magp_gap_obj.png'))
    figb.savefig(join(FIGURES_PATH, 'magp_gap_u.png'))
    figb.savefig(join(FIGURES_PATH, 'magp_best_solution.png'))


# %% VALIDATION - MA-GP-EIC
# load the results
analyzer = ExperimentAnalyzer(DATABASE)

results_bay_nm = analyzer.load('MA-GP-Bayesian-NM')
results_bay_de = analyzer.load('MA-GP-Bayesian-DE')
results_bay_sqp = analyzer.load('MA-GP-Bayesian-SQP')

results_processed_bay_de = analyzer.pre_process(results_bay_de, f_plant, u_plant, g_plant)
results_processed_bay_sqp = analyzer.pre_process(results_bay_sqp, f_plant, u_plant, g_plant)
results_processed_bay_nm = analyzer.pre_process(results_bay_nm, f_plant, u_plant, g_plant)

results_processed_bay_de['Cenário'] = 'DE/MA-GP-EIC'
results_processed_bay_sqp['Cenário'] = 'SQP/MA-GP-EIC'
results_processed_bay_nm['Cenário'] = 'NM/MA-GP-EIC'

results_processed_bay_de['Amostra'] = results_processed_bay_de['run.status'].map(lambda x: 'Inicialização' if x=='initialization' else 'Malha-fechada')
results_processed_bay_sqp['Amostra'] = results_processed_bay_sqp['run.status'].map(lambda x: 'Inicialização' if x=='initialization' else 'Malha-fechada')
results_processed_bay_nm['Amostra'] = results_processed_bay_nm['run.status'].map(lambda x: 'Inicialização' if x=='initialization' else 'Malha-fechada')

results_magpeic_validation = pd.concat([results_processed_bay_de, results_processed_bay_sqp, results_processed_bay_nm], ignore_index=True)

_, figa = plot_by_iteration(results_magpeic_validation, 'dPhi', '$\Delta G_0$ (%)', '',style='Cenário', hue='Cenário', xlabel='Iteração')
_, figb = plot_by_iteration(results_magpeic_validation, 'du', '$\Delta \mathbf{u}$ (%)', '',style='Cenário', hue='Cenário', xlabel='Iteração')
_, figc = plot_by_iteration(results_magpeic_validation, 'dBest', '$\Delta G_0^{\star}$ (%)', '',style='Cenário', hue='Cenário', xlabel='Iteração')

if FIGURES_PATH is not None:
    figa.savefig(join(FIGURES_PATH, 'magpeic_validation_gap_obj.png'))
    figb.savefig(join(FIGURES_PATH, 'magpeic_validation_gap_u_eic.png'))
    figc.savefig(join(FIGURES_PATH, 'magpeic_validation_best_solution_eic.png'))

# %% ALL VALIDATION RESULTS 
results_all_validation = pd.concat([results_processed_ma_sqp, results_processed_ma_de, results_processed_bay_de, results_processed_bay_sqp, results_processed_bay_nm], ignore_index=True)

_, figa = plot_by_iteration(results_all_validation, 'dPhi', '$\Delta G_0$ (%)', '', style='Cenário', hue='Cenário', xlabel='Iteração')
_, figb = plot_by_iteration(results_all_validation, 'du', '$\Delta \mathbf{u}$ (%)', '', style='Cenário', hue='Cenário', xlabel='Iteração')
_, figc = plot_by_iteration(results_all_validation, 'dBest', '$\Delta G_0^{\star}$ (%)', '',style='Cenário', hue='Cenário', xlabel='Iteração')


if FIGURES_PATH is not None:
    figa.savefig(join(FIGURES_PATH, 'all_validation_gap_obj.png'))
    figb.savefig(join(FIGURES_PATH, 'all_validation_gap_u.png'))
    figc.savefig(join(FIGURES_PATH, 'all_validation_best_solution.png'))

# %%
def plot_decision_surface(fig, ax, xx, yy, z_f, z_c, title, contour_type='contourf'):
    ax.grid(b=True, which='major', color='gray', linestyle='dotted')

    if(z_f is not None):
        ax.contour(xx, yy, z_f, colors='red')

    if(contour_type=='contourf'):
        CS = ax.contourf(xx, yy, z_c, cmap='viridis')
    else:
        CS = ax.contour(xx, yy, z_c, cmap='viridis')
    fig.colorbar(CS, ax=ax)
    ax.set_xlabel('$F_b$')
    ax.set_ylabel('$T_r$')
    ax.set_title(title)

def get_grid_predictions(models, xx, yy):
    # unpack stuff
    fobj = models['f']
    scaler = models['gp_scaler']
    constraint0 = models['g_0']
    constraint1 = models['g_1']
    
    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    # horizontal stack vectors to create x1,x2 input for the model
    grid = np.hstack((r1,r2))
    # make predictions for the grid
    cost = np.array([model.get_objective(x) + fobj.predict(scaler.transform(x.reshape(1,-1))) for x in grid])
    g_0 = np.array([model.get_constraints(x)[0] + constraint0.predict(scaler.transform(x.reshape(1,-1))) for x in grid])
    g_1 = np.array([model.get_constraints(x)[1] + constraint1.predict(scaler.transform(x.reshape(1,-1))) for x in grid])

    # reshape the predictions back into a grid
    zz_cost = cost.reshape(xx.shape)
    zz_g0 = g_0.reshape(xx.shape)
    zz_g1 = g_1.reshape(xx.shape)

    return zz_cost, zz_g0, zz_g1

def plot_gp_surface(fig, ax, xx, yy, z_f, title):
    CS = ax.contour(xx, yy, z_f)
    fig.colorbar(CS, ax=ax)
    ax.set_xlabel('$F_b$')
    ax.set_ylabel('$T_r$')
    ax.set_title(title)

def plot_gp_predictions(fig, ax, i, gp_iterations, xx, yy, title, initial_data_size=5):
    f_gp, g0_gp, g1_gp = get_grid_predictions(gp_iterations[initial_data_size + i],xx,yy)
    g_gp = (g1_gp < g1)&(g0_gp < g0)
    plot_decision_surface(fig, ax, xx, yy, g_gp, f_gp, title)


# %%
def get_eic_grid(models, xx, yy, f_best):
    # unpack stuff
    fobj = models['f']
    scaler = models['gp_scaler']
    constraint0 = models['g_0']
    constraint1 = models['g_1']
    
    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    # horizontal stack vectors to create x1,x2 input for the model
    grid = np.hstack((r1,r2))
    # make predictions for the grid
    ei = []
    eic = []
    g0_probs = []
    g1_probs = []

    for x in grid:
        xs = scaler.transform(x.reshape(1,-1))
        model_obj = model.get_objective(x)
        model_g = model.get_constraints(x)

        adaptation = Bunch(modifiers=[fobj.predict(xs, return_std=True), constraint0.predict(xs, return_std=True), constraint1.predict(xs, return_std=True)])
        ei_f = ModelBasedBayesianOptimizer.ei_acquisition(model_obj, adaptation, f_best)

        ei.append(ei_f)
        probs = ModelBasedBayesianOptimizer.constraint_probability(g, model_g, adaptation)
        eic.append(ei_f*np.prod(probs))
        g0_probs.append(probs[0])
        g1_probs.append(probs[1])


    # reshape the predictions back into a grid
    zz_ei = np.array(ei).reshape(xx.shape)
    zz_g0 = np.array(g0_probs).reshape(xx.shape)
    zz_g1 = np.array(g1_probs).reshape(xx.shape)
    zz_eic = np.array(eic).reshape(xx.shape)

    return zz_ei, zz_g0, zz_g1, zz_eic

def plot_eic_grid(fig, ax, i, best_solutions, gp_iterations, xx, yy, initial_data_size=5):
    fbest = best_solutions.iloc[initial_data_size + i]
    f_gp, g0_gp, g1_gp, eic = get_eic_grid(gp_iterations[initial_data_size + i],xx,yy,fbest)
    plot_decision_surface(fig, ax[0], xx, yy, None, f_gp, 'EI')
    plot_decision_surface(fig, ax[1], xx, yy, None, g1_gp*g0_gp, '$P(G(x) <= g)$')
    plot_decision_surface(fig, ax[2], xx, yy, None, eic, 'EIC')

def plot_eic(i, results, gp_iterations, xx, yy, initial_data_size=5):
    fig, ax = plt.subplots(figsize=(10,8))
    fbest = results['best_plant_objective'].iloc[initial_data_size + i]
    _, _, _, eic = get_eic_grid(gp_iterations[initial_data_size + i],xx,yy,fbest)
    plot_decision_surface(fig, ax, xx, yy, None, eic, 'EIC')
    ax.scatter(u_plant[0], u_plant[1], c='r', marker='*', s=150, zorder=6)
    u_eic = results['u_opt'].iloc[initial_data_size + i]
    ax.scatter(u_eic[0], u_eic[1], c='r', marker='x', s=150, zorder=6)
    return fig, ax

def plot_eic_iterations(iterations, results, gp_models, initial_data_size=5):
    for i, itertrain in enumerate(iterations):
        fig, ax = plt.subplots(1,3,figsize=(20,6))
        plot_eic_grid(fig, ax, itertrain, results['best_plant_objective'], gp_models,xx,yy)

        u_eic = results['u_opt'].iloc[initial_data_size + itertrain]
        
        ax[0].scatter(u_plant[0], u_plant[1], c='r', marker='*', s=150, zorder=6)
        ax[1].scatter(u_plant[0], u_plant[1], c='r', marker='*', s=150, zorder=6)
        ax[2].scatter(u_plant[0], u_plant[1], c='r', marker='*', s=150, zorder=6)

        ax[1].scatter(u_eic[0], u_eic[1], c='r', marker='x', s=150, zorder=6)
        ax[2].scatter(u_eic[0], u_eic[1], c='r', marker='x', s=150, zorder=6)

        fig.suptitle(f'Iteração: {itertrain}')
        fig.show()

        if FIGURES_PATH is not None:
            fig.savefig(join(FIGURES_PATH, f'iter{itertrain}_dec_surf_eic.png'))


# %%
# generate the results
gp_results_bay_de = [analyzer.load_run_models(run_id) for run_id in results_processed_bay_de['run.id']]
gp_results_bay_sqp = [analyzer.load_run_models(run_id) for run_id in results_processed_bay_sqp['run.id']]

# %% DECISION SURFACES - EIC
if(MODEL_NAME == 'wo'):
    grid_size = 50
    u1 = np.linspace(3, 6, grid_size)
    u2 = np.linspace(70, 100, grid_size)
    xx, yy = np.meshgrid(u1, u2)
    # plot_eic_iterations([0], results_processed_bay_de, gp_results_bay_de)
    plot_eic_iterations([0], results_processed_bay_sqp, gp_results_bay_sqp)

    fig_eic, ax_eic = plot_eic(0, results_processed_bay_sqp, gp_results_bay_sqp, xx, yy)
    if FIGURES_PATH is not None:
        fig_eic.savefig(join(FIGURES_PATH, 'zoom_iter0_dec_surf_eic.png'))


# %% COMPLETE - MA-GP-EIC
analyzer = ExperimentAnalyzer(DATABASE)
results_bay_datasets_de = analyzer.load('MA-GP-Bayesian-DE+noise-datasets')
results_bay_datasets_nm = analyzer.load('MA-GP-Bayesian-NM+noise-datasets')

results_bay_processed_datasets_de = analyzer.pre_process(results_bay_datasets_de, f_plant, u_plant, g_plant)
results_bay_processed_datasets_nm = analyzer.pre_process(results_bay_datasets_nm, f_plant, u_plant, g_plant)

results_bay_processed_datasets_de['Cenário'] = 'DE/MA-GP-EIC'
results_bay_processed_datasets_nm['Cenário'] = 'NM/MA-GP-EIC'

results_bay_processed_datasets_de['Amostra'] = results_bay_processed_datasets_de['run.status'].map(lambda x: 'Inicialização' if x=='initialization' else 'Malha-fechada')
results_bay_processed_datasets_nm['Amostra'] = results_bay_processed_datasets_nm['run.status'].map(lambda x: 'Inicialização' if x=='initialization' else 'Malha-fechada')

results_magpeic = pd.concat([results_bay_processed_datasets_de, results_bay_processed_datasets_nm], ignore_index=True)
results_magpeic_magp = pd.concat([results_processed_ma_de_noise_df, results_bay_processed_datasets_de], ignore_index=True)
results_magpeic_magp_all = pd.concat([results_processed_ma_de_noise_df, results_bay_processed_datasets_de, results_bay_processed_datasets_nm], ignore_index=True)

# %%
_, figa = plot_by_iteration(results_magpeic, 'dPhi', '$\Delta G_0$ (%)', '',style='Cenário', hue='Cenário', xlabel='Iteração')
_, figb = plot_by_iteration(results_magpeic, 'du', '$\Delta \mathbf{u}$ (%)', '',style='Cenário', hue='Cenário', xlabel='Iteração')
_, figc = plot_by_iteration(results_magpeic, 'dBest', '$\Delta G_0^{\star}$ (%)', '',style='Cenário', hue='Cenário', xlabel='Iteração')

if FIGURES_PATH is not None:
    figa.savefig(join(FIGURES_PATH, 'magpeic_gap_obj.png'))
    figb.savefig(join(FIGURES_PATH, 'magpeic_gap_u_eic.png'))
    figc.savefig(join(FIGURES_PATH, 'magpeic_best_solution_eic.png'))

# %%
axa, figa = plot_by_iteration(results_magpeic, 'dg0', '$G_1^p$', '',style='Cenário', hue='Cenário', xlabel='Iteração')
axb, figb = plot_by_iteration(results_magpeic, 'dg1', '$G_2^p$', '',style='Cenário', hue='Cenário', xlabel='Iteração')

if FIGURES_PATH is not None:
    figa.savefig(join(FIGURES_PATH, 'magpeic_g0.png'))
    figb.savefig(join(FIGURES_PATH, 'magpeic_g1.png'))


# %%
_, figa = plot_by_iteration(results_magpeic_magp, 'dPhi', '$\Delta G_0$ (%)', '',style='Cenário', hue='Cenário', xlabel='Iteração')
_, figb = plot_by_iteration(results_magpeic_magp, 'du', '$\Delta \mathbf{u}$ (%)', '',style='Cenário', hue='Cenário', xlabel='Iteração')
_, figc = plot_by_iteration(results_magpeic_magp, 'dBest', '$\Delta G_0^{\star}$ (%)', '',style='Cenário', hue='Cenário', xlabel='Iteração')


if FIGURES_PATH is not None:
    figa.savefig(join(FIGURES_PATH, 'comparison_gap_obj.png'))
    figb.savefig(join(FIGURES_PATH, 'comparison_gap_u.png'))
    figc.savefig(join(FIGURES_PATH, 'comparison_best_solution.png'))

# %%
_, figa = plot_by_iteration(results_magpeic_magp_all, 'dPhi', '$\Delta G_0$ (%)', '',style='Cenário', hue='Cenário', xlabel='Iteração')
_, figb = plot_by_iteration(results_magpeic_magp_all, 'du', '$\Delta \mathbf{u}$ (%)', '',style='Cenário', hue='Cenário', xlabel='Iteração')
_, figc = plot_by_iteration(results_magpeic_magp_all, 'dBest', '$\Delta G_0^{\star}$ (%)', '',style='Cenário', hue='Cenário', xlabel='Iteração')

if FIGURES_PATH is not None:
    figa.savefig(join(FIGURES_PATH, 'comparison_gap_obj_all.png'))
    figb.savefig(join(FIGURES_PATH, 'comparison_gap_u_all.png'))
    figc.savefig(join(FIGURES_PATH, 'comparison_best_solution_all.png'))

# %%
axa, figa = plot_by_iteration(results_magpeic_magp, 'dg0', '$G_1^p$', '',style='Cenário', hue='Cenário', xlabel='Iteração')
axb, figb = plot_by_iteration(results_magpeic_magp, 'dg1', '$G_2^p$', '',style='Cenário', hue='Cenário', xlabel='Iteração')

if FIGURES_PATH is not None:
    figa.savefig(join(FIGURES_PATH, 'comparison_g0.png'))
    figb.savefig(join(FIGURES_PATH, 'comparison_g1.png'))

# %%
# SQP vs NM
results_sqp_nm = pd.concat([results_processed_ma_sqp_noise_df, results_bay_processed_datasets_nm], ignore_index=True)

_, figa = plot_by_iteration(results_sqp_nm, 'dPhi', '$\Delta G_0$ (%)', '',style='Cenário', hue='Cenário', xlabel='Iteração')
_, figb = plot_by_iteration(results_sqp_nm, 'du', '$\Delta \mathbf{u}$ (%)', '',style='Cenário', hue='Cenário', xlabel='Iteração')
_, figc = plot_by_iteration(results_sqp_nm, 'dBest', '$\Delta G_0^{\star}$ (%)', '',style='Cenário', hue='Cenário', xlabel='Iteração')

if FIGURES_PATH is not None:
    figa.savefig(join(FIGURES_PATH, 'comparison_gap_obj_sqpnm.png'))
    figb.savefig(join(FIGURES_PATH, 'comparison_gap_u_sqpnm.png'))
    figc.savefig(join(FIGURES_PATH, 'comparison_best_solution_sqpnm.png'))


# %% TIME ANALYSIS

def plot_execution_costs(data, suffix):
    figa, axa = plt.subplots(1, 1, figsize=(8, 6))
    figb, axb = plt.subplots(1, 1, figsize=(8, 6))
    
    data_plot = data[data['run.status'] != 'initialization']


    results_palette = {
            'SQP/MA-GP': '#377eb8',
            'DE/MA-GP': '#e41a1c',
            'DE/MA-GP-EIC': '#4daf4a',
            'SQP/MA-GP-EIC': '#984ea3',
            'NM/MA-GP-EIC': '#ff7f00',
    } 

    # axa.minorticks_on()
    # axa.xaxis.set_tick_params(which='minor', bottom=False)
    axa.set_axisbelow(True)

    # axa.grid(b=True, axis='y', which='minor', color='gray', linestyle='dotted')
    axa.grid(b=True, axis='y', which='major', color='gray', linestyle='-')

    # axb.minorticks_on()
    # axb.grid(b=True, axis='y', which='minor', color='gray', linestyle='dotted')
    axb.grid(b=True, axis='y', which='major', color='gray', linestyle='-')
    # axb.xaxis.set_tick_params(which='minor', bottom=False)
    axb.set_axisbelow(True)

    sns.boxplot(y='opt_time', data=data_plot, x='Cenário', showfliers=False, ax=axa, palette=results_palette, linewidth=1.5)
    axa.set_ylabel('Tempo (s)')
    axa.set_xlabel('')
    sns.boxplot(y='n_fev', data=data_plot, x='Cenário', showfliers=False, ax=axb, palette=results_palette, linewidth=1.5)
    axb.set_ylabel('# avaliações do modelo')
    axb.set_xlabel('')

    print(data_plot.groupby('Cenário')['opt_time'].agg(['mean','std','min','max']))

    if FIGURES_PATH is not None:
        figa.savefig(join(FIGURES_PATH, f'opt_time_{suffix}.png'))
        figb.savefig(join(FIGURES_PATH, f'n_fev_{suffix}.png'))


# %%
results_sqp_nm = pd.concat([results_processed_ma_sqp_noise_df, results_bay_processed_datasets_nm], ignore_index=True)

plot_execution_costs(results_magpeic_magp, 'comparison')
plot_execution_costs(results_magpeic, 'magpeic')
plot_execution_costs(results_magp, 'magp')
plot_execution_costs(results_sqp_nm, 'sqp_nm')
# %%

# %%
