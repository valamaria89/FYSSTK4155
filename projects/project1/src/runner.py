import numpy as np
from fysstatistics import Regressor
from sampler import Sampler
from numpy import ndarray
from itertools import product
from modelselection import kfold_indices
from typing import Tuple, List, Dict, Any, Optional
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import turbo
if "JPY_PARENT_PID" in os.environ:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


class Runner:
    def __init__(self, sampler: Sampler, regressor: Regressor,
                 parameter_range: Optional[ndarray] = None,
                 CV_k: int = 5, find_prediction_error: bool = False) -> None:
        self.Regressor = regressor
        self.parameter_range = parameter_range
        self.CV_k = CV_k
        self.sampler = sampler
        self.find_prediction_error = find_prediction_error
        self.x = None
        self.y = None
        self.z = None
        self.all_regressors: List[Regressor] = []

        self.df = pd.DataFrame()


    def sample(self, num_samples: int) -> None:
        if self.find_prediction_error:
            noise_sigma = self.sampler.sigma
            self.sampler.set_noise(0)
            # Is this wrong?
            self.X_pred, self.z_pred = self.sampler.sample(1000)
            (self.x, self.y), self.z_true = self.sampler.sample(num_samples)
            self.sampler.set_noise(noise_sigma)
            self.z = self.z_true + self.sampler.add_noise(self.z_true)
        else:
            (self.x, self.y), self.z = self.sampler.sample(num_samples)

    def run(self, max_order: Tuple[int, int], single: bool = False,
            use_interactions: bool = True):
        self.use_interactions = use_interactions
        modelspace = self.model_space(max_order, single)
        terms = self.highest_terms(modelspace)
        self.df = self.construct_dataframe(len(modelspace), terms)

        i = 0
        for model in tqdm(modelspace):
            self.train_test_model(model, i)
            i += 1

    def train_test_model(self, model, i: int) -> None:
        x, y, z = self.x, self.y, self.z
        reg = self.make_regressor(model)
        mse_train = []
        mse_test = []
        r2_train = []
        r2_test = []
        betas = []
        regs = []
        for train, test in kfold_indices(x, k=self.CV_k):
            regressor = reg(x[train], y[train], z[train])
            mse_train.append(regressor.mse())
            mse_test.append(regressor.mse([x[test], y[test]], z[test]))
            betas.append(regressor.betadict())
            regs.append(regressor)
            r2_train.append(regressor.r2())
            r2_test.append(regressor.r2([x[test], y[test]], z[test]))

        if self.parameter_range is not None:
            self.df.loc[i, 'parameter'] = model[2]
        self.df.loc[i, 'MSE train'] = np.mean(mse_train)
        self.df.loc[i, 'MSE test'] = np.mean(mse_test)
        self.df.loc[i, 'train sd'] = np.std(mse_train)
        self.df.loc[i, 'test sd'] = np.std(mse_test)
        self.df.loc[i, 'complexity'] = regressor.vandermonde.shape[1]
        self.df.loc[i, 'df'] = regressor.df()
        self.df.loc[i, 'max x'] = max(model[0][0])
        self.df.loc[i, 'max y'] = max(model[0][1])
        self.df.loc[i, 'r2 train'] = np.mean(r2_train)
        self.df.loc[i, 'r2 test'] = np.mean(r2_test)

        # Set the terms
        best = np.argmin(mse_train)
        for term, coeff in betas[best].items():
            self.df.loc[i, term] = coeff

        # Compute predicition error
        if self.find_prediction_error:
            mse = []
            for reg in regs:
                mse_ = reg.mse(self.X_pred, self.z_pred)
                mse.append(mse_)
            self.df.loc[i, 'MSE pred'] = np.mean(mse)
            self.df.loc[i, 'pred sd'] = np.std(mse)

        # Save the best model for later use
        self.all_regressors.append(regs[best])

    def highest_terms(self, modelspace: List[Any]) -> List[str]:
        highest = modelspace[-1]
        reg = self.make_regressor(highest)
        regressor = reg(self.x[0:2], self.y[0:2], self.z[0:2])
        return regressor.betadict().keys()

    def make_regressor(self, model) -> Any:
        def reg(x, y, z):
            if model[2] is not None:
                regressor = self.Regressor([x, y], z, parameter=model[2])
            else:
                regressor = self.Regressor([x, y], z)
            if self.use_interactions:
                regressor.fit(model[0], max_interaction=model[1])
            else:
                regressor.fit(model[0], interactions=False)
            return regressor
        return reg

    def model_space(self, max_order: Tuple[int, int],
                    single: bool = False) -> List[Tuple[List[int],
                                                  Optional[int],
                                                  Optional[float]]]:

        # We want to run a single model of given order
        if single:
            if self.use_interactions:
                max_interaction = max_order[0]*max_order[1]
                interactions = [[[*max_order], max_interaction]]
            else:
                interactions = [[[*max_order], None]]
        # We want all possible models up to given order
        else:
            XY = product(range(1, max_order[0]+1),
                         range(1, max_order[1]+1))
            interactions = []
            for x, y in XY:
                if self.use_interactions:
                    for max_interaction in list(range(1, x*y+1)):
                        interactions.append([[x, y], max_interaction])
                else:
                    interactions.append([[x, y], None])

        # Add the parameters to the model space
        space = []
        if self.parameter_range is not None:
            for parameter in self.parameter_range:
                for model in interactions:
                    space.append((*model, parameter))
        else:
            for model in interactions:
                space.append((*model, None))

        return space

    def best_model(self, what='test') -> Regressor:
        i = self.df[f'MSE {what}'].astype(float).argmin()
        mse = self.df[f'MSE {what}'].astype(float).min()
        se = self.df[f'{what} sd'][i]
        complexity = self.df['complexity'][i]
        high = mse + se
        best = i
        #print(best, low, high)
        for (i, row) in self.df.iterrows():
            if row['complexity'] < complexity:
                se_c = row[f'{what} sd']
                mse_c = row[f'MSE {what}']
                if mse_c - se_c < high:
                    best = i
                    complexity = row['complexity']
                    #print(f"{i} with {complexity}: {mse_c - se_c}, {mse_c + se_c}")
        return self.all_regressors[best]

    def construct_dataframe(self, num_iterations: int, all_terms: List[str]) -> pd.DataFrame:
        columns = ['parameter', 'MSE train', 'MSE test', 'train sd',
                   'test sd', 'complexity', 'df', 'max x', 'max y',
                   'r2 train', 'r2 test', *all_terms]
        if self.find_prediction_error:
            columns.append('MSE pred')
            columns.append('pred sd')

        df = pd.DataFrame(index=range(num_iterations), columns=columns)
        return df

    def coeff_plot(self, ax=None) -> Tuple[Any, Any]:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        term_cols = [term for term in self.df.columns if '^' in term]
        terms = [f'${term}$' for term in term_cols]
        array = np.asarray(self.df[term_cols])
        #cmaplist = cmap(np.arange(cmap.N))
        #cmap = mpl.colors.LinearSegmentedColormap.from_list('Discrete', cmaplist, cmap.N)
        #bounds = np.arange(0, array.shape[0])
        #norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        terms = terms*array.shape[0]
        c = np.tile(self.df['parameter'].to_numpy(), (array.shape[1], 1)).T
        scat = ax.scatter(array, terms, c=c, norm=matplotlib.colors.LogNorm(),
                          cmap='Greens', s=5)
        # for i in range(0, array.shape[0]):
            # color = [cmap(i/array.shape[0]), ]*array.shape[1]
            # ax.scatter(array[i, :], terms, c=color, s=10, alpha=0.5)
        #ax.set_xscale('symlog')
        ax.set_xlabel('Coefficient value')
        cb = fig.colorbar(scat)
        cb.outline.set_linewidth(0)
        cb.ax.set_ylabel('Parameter $\\lambda$')
        return fig, ax

    def coeff_evolution(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        term_cols = [term for term in self.df.columns if '^' in term]
        #terms = [f'${term}$' for term in term_cols]
        #terms = terms*array.shape[0]
        cmap = plt.get_cmap('Greens')
        terms = self.df[term_cols]
        param = self.df['parameter'].to_numpy()
        param = np.tile(param, (terms.shape[1], 1)).T
        #param = np.tile(self.df['parameter'].to_numpy(), (array.shape[1], 1)).T
        #c = np.tile(self.df[''].to_numpy(), (array.shape[1], 1)).T
        #lines = ax.plot(param, terms, c=self.df['df'].to_numpy())#, c=c, norm=matplotlib.colors.LogNorm(),
        #                  cmap='Greens', s=5)
        for i, term in enumerate(term_cols):
            ax.plot(self.parameter_range, self.df[term], label=f"${term}$",
                    c=cmap(i/len(term_cols)))
        return fig, ax

    def ci_plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None
        cmap = plt.get_cmap('Greens')
        c = cmap(0.5)
        c2 = cmap(0.8)
        model = self.best_model('test')
        betas = model.betadict()
        ci = model.ci(0.95)
        for i, ((term, coeff), (low, high)) in enumerate(zip(betas.items(), ci)):
            label = f"${term}$"
            ax.plot([low, high], [label, label], c=c, zorder=0)
            ax.scatter([betas[term]], [label], c=c2, s=5)
        #ax.set_yticks(np.arange(0, len(ci)))
        #ax.set_yticklabels(terms)
        ax.set_xlabel("Coefficient value")

        return fig, ax

    def mse_plot(self, ax=None) -> Tuple[Any, Any]:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        MSE_train = self.df['MSE train'].to_numpy(dtype=float)
        MSE_test = np.array(self.df['MSE test'].to_numpy(), dtype=float)
        test_sd = self.df['test sd'].to_numpy()
        test_sd = np.array(test_sd, dtype=float)
        train_sd = self.df['train sd'].to_numpy()
        train_sd = np.array(train_sd, dtype=float)
        trax, = ax.plot(self.parameter_range, MSE_train)
        #            yerr=train_sd, label='Train', elinewidth=0.5)
        tax, = ax.plot(self.parameter_range, MSE_test)
        ax.fill_between(self.parameter_range,
                        MSE_test - test_sd,
                        MSE_test + test_sd,
                        label='Test',
                        alpha=0.3, color=tax.get_color())
        ax.fill_between(self.parameter_range,
                        MSE_train - train_sd,
                        MSE_train + train_sd,
                        label='Train',
                        alpha=0.3, color=trax.get_color())

        testbest = self.best_model('test').parameter
        trainbest = self.best_model('train').parameter
        ylim = ax.get_ylim()
        ax.vlines(trainbest, *ax.get_ylim(), color=trax.get_c(),
                  linewidth=0.5, linestyle='--')
        ax.vlines(testbest, *ax.get_ylim(), color=tax.get_color(),
                  linewidth=0.5, linestyle='--')
        if self.find_prediction_error:
            predbest = self.best_model('pred').parameter
            MSE_pred = np.asarray(self.df['MSE pred'])
            pr, = ax.plot(self.parameter_range, MSE_pred, label='Pred')
            ax.vlines(predbest, *ylim, color=pr.get_color(),
                      linewidth=0.5, linestyle='--')
        ax.set_ylim(ylim)
        ax.set_xscale('log')
        ax.set_ylabel('MSE')
        fig.legend(loc=9, ncol=3, frameon=False)
        return fig, ax

    def compare_plot(self, ax=None, use_3d=False, which='test') -> Tuple[Any, Any]:
        if ax is None:
            fig = plt.figure()
            if use_3d:
                fact_ax = fig.add_subplot(1, 2, 1, projection='3d')
                model_ax = fig.add_subplot(1, 2, 2, projection='3d')
            else:
                fact_ax = fig.add_subplot(1, 2, 1)
                model_ax = fig.add_subplot(1, 2, 2)
        else:
            fig = ax[0].figure
            fact_ax = ax[0]
            model_ax = ax[1]

        (xp, yp), zp = self.sampler.population_sample()

        vmin = np.min(zp)
        vmax = np.max(zp)
        model = self.best_model(which)
        pred = model.predict([xp, yp])
        if self.sampler.type == ndarray:
            if use_3d:
                surface = fact_ax.plot_surface(xp, yp,
                                               zp, cmap='turbo',
                                               antialiased=False, linewidth=0,
                                               vmin=vmin, vmax=vmax)
                surface = model_ax.plot_surface(xp, yp,
                                                pred, cmap='turbo',
                                                antialiased=False, linewidth=0,
                                                vmin=vmin, vmax=vmax)
                model_ax.set_axis_off()
                fact_ax.set_axis_off()
                cb = fig.colorbar(surface)
                cb.outline.set_linewidth(0)
            else:
                fact_ax.imshow(zp.T, cmap='turbo', vmin=vmin, vmax=vmax)
                model_ax.imshow(pred.T, cmap='turbo', vmin=vmin, vmax=vmax)
        else:
            if use_3d:
                surface = fact_ax.plot_surface(xp, yp,
                                               zp, cmap='turbo',
                                               antialiased=False, linewidth=0,
                                               vmin=vmin, vmax=vmax)
                surface = model_ax.plot_surface(xp, yp,
                                                pred, cmap='turbo',
                                                antialiased=False, linewidth=0,
                                                vmin=vmin, vmax=vmax)
                model_ax.set_axis_off()
                fact_ax.set_axis_off()
                cb = fig.colorbar(surface)
                cb.outline.set_linewidth(0)
            else:
                fact_ax.pcolormesh(xp, yp, zp, cmap='turbo', vmin=vmin, vmax=vmax)
                model_ax.pcolormesh(xp, yp, pred, cmap='turbo', vmin=vmin, vmax=vmax)
                #model_ax.scatter(xp, yp, s=1)

        return fig, (fact_ax, model_ax)




