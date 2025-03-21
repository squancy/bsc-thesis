import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pareto, norm
import random
import matplotlib
from mm_est import mm_est
from ml_est import ml_est
from par import generate_pareto_values

sns.set(style="whitegrid", palette="colorblind", font_scale=1.2)
sns.set_context("paper")
matplotlib.rcParams['font.family'] = 'serif'

def est(n, num_samples, alpha, c):
    """Computes Y_n for num_samples with n Pareto-distributed random variables."""
    Y_n = []
    alphas = {
        'MM': [],
        'ML': []
    }
    cs = {
        'MM': [],
        'ML': []
    }
    for _ in range(num_samples):
        pareto_values = generate_pareto_values(n, alpha, c)
        est_method_of_moments = mm_est(pareto_values)
        est_max_likelihood = ml_est(pareto_values)
        alphas['MM'].append(est_method_of_moments[0])
        alphas['ML'].append(est_max_likelihood[0])
        cs['MM'].append(est_method_of_moments[1])
        cs['ML'].append(est_max_likelihood[1])
        Y_n.append(np.mean(pareto_values))
    return Y_n, alphas, cs

def calc_normal_params(alpha, c):
    mean = c * alpha / (alpha - 1)
    variance = (c**2 * alpha) / ((alpha - 1)**2 * (alpha - 2)) if alpha > 2 else np.inf
    std_dev = np.sqrt(variance / n) if alpha > 2 else np.inf
    return mean, variance, std_dev

def plot_normal_dist(ax, Y_n, mean, std_dev, color, label):
    x = np.linspace(min(Y_n), max(Y_n), 1000)
    normal_density = norm.pdf(x, loc=mean, scale=std_dev)
    sns.lineplot(x=x, y=normal_density, color=color, lw=2, ax=ax, label=label, alpha=0.65)

def plot_histograms_with_seaborn(n, num_samples, subplot_rows, subplot_cols):
    """Plots histograms with normal distribution overlays in subplots using Seaborn."""
    fig, axes = plt.subplots(subplot_rows, subplot_cols, figsize=(12, 8))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        alpha = round(random.uniform(2.5, 7), 2) 
        c = round(random.uniform(2.8, 8), 2) 

        Y_n, alphas, cs = est(n, num_samples, alpha, c)
        mm_alpha, mm_c = round(np.mean(alphas['MM']), 2), round(np.mean(cs['MM']), 2)
        ml_alpha, ml_c = round(np.mean(alphas['ML']), 2), round(np.mean(cs['ML']), 2)

        mm_mean, mm_variance, mm_std_dev = calc_normal_params(mm_alpha, mm_c)
        ml_mean, ml_variance, ml_std_dev = calc_normal_params(ml_alpha, ml_c)

        data = {"Y_n": Y_n}

        sns.histplot(
            data=data,
            x="Y_n",
            bins=30,
            stat="density",
            kde=False,
            alpha=0.7,
            edgecolor="black",
            ax=ax,
            label="Histogram of $Y_n$"
        )
        
        if np.isfinite(mm_std_dev):
            plot_normal_dist(ax, Y_n, mm_mean, mm_std_dev, "red", "Normal Approximation (MM)")
        if np.isfinite(ml_std_dev):
            plot_normal_dist(ax, Y_n, ml_mean, ml_std_dev, "black", "Normal Approximation (ML)")

        ax.set_title(rf"$\alpha = {alpha}, c = {c}$", fontsize=12)
        ax.set_xlabel(r"$Y_n$", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("../plots/yn_sim.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    n = 10 ** 3 
    num_samples = 10 ** 4 
    subplot_rows = 2 
    subplot_cols = 3 

    plot_histograms_with_seaborn(n, num_samples, subplot_rows, subplot_cols)
