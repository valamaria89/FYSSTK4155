import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import os
from typing import Callable, Optional, Tuple, Any
from sklearn.model_selection import train_test_split
if "JPY_PARENT_PID" in os.environ:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


class Ensemble:
    def __init__(self, X, y, num_members: int = 10) -> None:
        self.X = X
        self.y = y
        self.num_members = num_members

    def run(self, estimator, hyperparameters) -> None:
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.1, random_state=1)
        start = 0
        X_ensemble, y_ensemble = [], []
        for i in range(self.num_members):
            stop = (i+1)*(len(X_train)//self.num_members)
            X_ensemble.append(X_train[start:stop])
            y_ensemble.append(y_train[start:stop])
            start = stop

        n_fits = len(hyperparameters)
        self.hp = hyperparameters
        self.MSE_train = np.zeros((n_fits, self.num_members))
        self.MSE_test = np.zeros_like(self.MSE_train)
        self.variance = np.zeros(n_fits)
        self.bias = np.zeros_like(self.variance)
        self.coeffs = []
        progress = tqdm(total=n_fits*self.num_members)
        for i, hp in enumerate(hyperparameters):
            prediction = np.zeros((self.num_members, len(y_test)))
            clf = estimator(hp)
            for n in range(self.num_members):
                clf = clf.fit(X_ensemble[n], y_ensemble[n])
                prediction[n, :] = clf.predict_proba(X_test)[:, 1]
                self.MSE_train[i, n] = clf.score(X_ensemble[n], y_ensemble[n])
                self.MSE_test[i, n] = clf.score(X_test, y_test)
                progress.update(1)

            if hasattr(clf, "coef_"):
                self.coeffs.append(clf.coef_.ravel().copy())

            mean = prediction.mean(axis=0)
            var = prediction.var(axis=0)
            squared_bias = (y_test - mean)**2

            self.variance[i] = var.mean()
            self.bias[i] = squared_bias.mean()


    def plot_train_test(self, ax: Optional[Any] = None) -> Tuple[Any, Any]:
        if ax is None:
            fig, ax = plt.subplots(1)
        else:
            fig = None

        hp = self.hp
        dfs = np.tile(hp, (self.num_members, 1)).T
        print(self.MSE_train.shape)
        ax.plot(dfs, self.MSE_train, color=lighten_color('dodgerblue'),
                alpha=0.5, linewidth=0.5)
        ax.plot(dfs, self.MSE_test, '-', color=lighten_color('forestgreen'),
                alpha=0.5, linewidth=0.5)
        train_mean = np.mean(self.MSE_train, axis=1)
        test_mean = np.mean(self.MSE_test, axis=1)
        ax.plot(hp, train_mean, '-',
                label='Train', color='dodgerblue', linewidth=0.4)
        ax.plot(hp, test_mean,
                '-', label='Test', color='forestgreen', linewidth=0.4)

        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Hyperparameter")
        ax.set_xscale('log')
        lgd = fig.legend(loc='lower left',# mode='expand', 
                         ncol=2,
                         bbox_to_anchor=(0.3, 1.02, 1, 0.2))

        return fig, ax


    def plot_decomposition(self, ax: Optional[Any] = None) -> Tuple[Any, Any]:
        if ax is None:
            fig, ax = plt.subplots(1)
        else:
            fig = None

        hp = self.hp
        dfs = np.tile(hp, (self.num_members, 1)).T
        #ax.plot(dfs, self.MSE_true, '-', color=lighten_color('forestgreen'),
        #        alpha=0.1, linewidth=0.5)
        train_mean = np.mean(self.MSE_train, axis=1)
        test_mean = np.mean(self.MSE_test, axis=1)
        #ax.plot(hp, test_mean, '-',
        #        label='Test', color='forestgreen', linewidth=0.4)
        #ax.plot(self.degrees_of_freedom, test_mean,
        #        '-', label='Test', color='forestgreen', linewidth=0.4,
        #        alpha=0.5)
        #ax.plot(self.degrees_of_freedom, true_mean,
        #        '-', label='True', color='forestgreen', linewidth=0.8)
        #ax.plot(self.degrees_of_freedom, true_mean + 0.1**2,
        #        '--', label=r"$bias^2 + Var + \sigma_{\varepsilon}^2$",
        #        linewidth=0.4)
        ax.plot(hp, self.bias, label='$bias^2$')
        ax.plot(hp, self.variance, label='$Var$')
        ax.plot(hp, self.bias + self.variance, 
                '--', label="$bias^2 + Var$")

        ax.set_ylabel("Proability Error")
        ax.set_xlabel("Hyperparameter")
        ax.set_xscale('log')
        #fig.legend()
        fig.legend(loc='lower left', # mode='expand', 
                   ncol=3,
                   bbox_to_anchor=(0.1, 1.02, 1, 0.2))

        return fig, ax

    def plot_coeffs(self):
        coeffs = np.asarray(self.coeffs)
        fig, ax = plt.subplots()
        ax.plot(self.hp, coeffs)
        ax.set_xlabel("Hyperparameter")
        ax.set_ylabel("Coefficient Value")
        ax.set_yscale("symlog")
        ax.set_xscale("log")
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
