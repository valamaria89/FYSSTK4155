import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import os
from modelselection import kfold
from fysstatistics import Regressor
from typing import Callable, Optional, Tuple, Any
from sampler import Sampler
from sklearn.model_selection import train_test_split
if "JPY_PARENT_PID" in os.environ:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


class Ensemble:
    def __init__(self, sampler: Sampler,
                 num_members: int = 10, sample_size: int = 100,
                 order: int = 2) -> None:
        self.sampler = sampler
        self.num_members = num_members
        self.sample_size = sample_size
        self.order = order

    def run(self, Reg: Regressor) -> None:
        self.sampler.set_noise(0.0)
        X, Y = self.sampler(self.sample_size)
        Xtest, Ytest = self.sampler(self.sample_size)
        noise = np.random.normal(0, 0.1, (self.num_members, self.sample_size))
        train = np.tile(Y, (self.num_members, 1)) + noise
        dfs = self.order
        self.degrees_of_freedom = np.zeros(dfs)
        self.MSE_train = np.zeros((dfs, self.num_members))
        self.MSE_test = np.zeros_like(self.MSE_train)
        self.MSE_true = np.zeros_like(self.MSE_train)
        self.variance = np.zeros(dfs)
        self.bias = np.zeros_like(self.variance)
        progress = tqdm(total=dfs*self.num_members)
        for i, order in enumerate(range(1, self.order+1)):
            prediction = np.zeros_like(train)
            for n in range(self.num_members):
                regressor = Reg(X, train[n, :])
                regressor.fit([order, order], interactions=False)
                prediction[n, :] = regressor.predict(Xtest)
                self.MSE_train[i, n] = regressor.mse()
                self.MSE_true[i, n] = regressor.mse(X, Y)
                self.MSE_test[i, n] = regressor.mse(Xtest, Ytest)
                progress.update(1)

            mean = prediction.mean(axis=0)
            var = prediction.var(axis=0)
            squared_bias = (Ytest - mean)**2


            self.variance[i] = var.mean()
            self.bias[i] = squared_bias.mean()

            self.degrees_of_freedom[i] = order+order#regressor.df()


    def plot_train_test(self, ax: Optional[Any] = None) -> Tuple[Any, Any]:
        if ax is None:
            fig, ax = plt.subplots(1)
        else:
            fig = None

        dfs = np.tile(self.degrees_of_freedom, (self.num_members, 1)).T
        ax.plot(dfs, self.MSE_train, color=lighten_color('dodgerblue'),
                alpha=0.1, linewidth=0.5)
        ax.plot(dfs, self.MSE_test, '-', color=lighten_color('forestgreen'),
                alpha=0.1, linewidth=0.5)
        train_mean = np.mean(self.MSE_train, axis=1)
        test_mean = np.mean(self.MSE_test, axis=1)
        ax.plot(self.degrees_of_freedom, train_mean, '-',
                label='Train', color='dodgerblue', linewidth=0.4)
        ax.plot(self.degrees_of_freedom, test_mean,
                '-', label='Test', color='forestgreen', linewidth=0.4)

        ax.set_ylabel("MSE")
        ax.set_xlabel("Complexity")
        lgd = fig.legend(loc='lower left',# mode='expand', 
                         ncol=2,
                         bbox_to_anchor=(0.3, 1.02, 1, 0.2))

        return fig, ax


    def plot_decomposition(self, ax: Optional[Any] = None) -> Tuple[Any, Any]:
        if ax is None:
            fig, ax = plt.subplots(1)
        else:
            fig = None

        dfs = np.tile(self.degrees_of_freedom, (self.num_members, 1)).T
        #ax.plot(dfs, self.MSE_true, '-', color=lighten_color('forestgreen'),
        #        alpha=0.1, linewidth=0.5)
        train_mean = np.mean(self.MSE_train, axis=1)
        test_mean = np.mean(self.MSE_test, axis=1)
        true_mean = np.mean(self.MSE_true, axis=1)
        ax.plot(self.degrees_of_freedom, test_mean, '-',
                label='Test', color='forestgreen', linewidth=0.4)
        #ax.plot(self.degrees_of_freedom, test_mean,
        #        '-', label='Test', color='forestgreen', linewidth=0.4,
        #        alpha=0.5)
        #ax.plot(self.degrees_of_freedom, true_mean,
        #        '-', label='True', color='forestgreen', linewidth=0.8)
        #ax.plot(self.degrees_of_freedom, true_mean + 0.1**2,
        #        '--', label=r"$bias^2 + Var + \sigma_{\varepsilon}^2$",
        #        linewidth=0.4)
        ax.plot(self.degrees_of_freedom, self.bias, label='$bias^2$')
        ax.plot(self.degrees_of_freedom, self.variance, label='Var')
        ax.plot(self.degrees_of_freedom, self.bias + self.variance, 
                '--', label="$bias^2 + Var$")

        ax.set_ylabel("MSE")
        ax.set_xlabel("Complexity")
        fig.legend(loc='lower left', # mode='expand', 
                   ncol=2,
                   bbox_to_anchor=(-1.3, 1.02, 1, 0.2))

        return fig, ax



def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
