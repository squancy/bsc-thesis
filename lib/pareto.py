from scipy.stats import pareto
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from mm_est import mm_est
from par import generate_pareto_values

sns.set(style="whitegrid", palette="colorblind", font_scale=1.2)
sns.set_context("paper")
matplotlib.rcParams['font.family'] = 'serif'

def perform_experiment(N, repeat, cs, alphas):
    """
    Generates N i.i.d. Pareto distributed random variables, estimates its parameters c and alpha
    and counts the number of random variables that are smaller than c.
    Then repeats the experiment [repeat] times and calculates the estimated probability.
    """
    p = []
    for c, alpha in zip(cs, alphas):
        w = []
        for _ in range(repeat):
            X_i = generate_pareto_values(N, c, alpha)
            est_alpha, est_c = mm_est(X_i)
            w.append((X_i > est_c).sum() / N)
        p.append(sum(w) / 10**3)
    return p

def plot_prob(p, cs, alphas):
    """
    Plots the estimated probabilities calculated and saves the graph as an image
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    x_labels = [f"$({c}, {alpha})$" for c, alpha in zip(cs, alphas)]
    sns.lineplot(x=range(len(p)), y=p, ax=ax, marker="o", label=r"Estimated $\mathbb{P}$(X > c)", color='blue')

    ax.set_xlabel(r"(c, $\alpha$)", fontsize=16, labelpad=20)
    ax.set_ylabel(r"Estimated $\mathbb{P}(X > c)$", fontsize=16, labelpad=30)
    n = 7  # Keeps every 7th label for readability
    ax.set_xticks(range(0, len(x_labels), n))
    ax.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), n)], rotation=30, ha="right", fontsize=14)
    
    ax.set_title(r"Estimated Probability of X > c", fontsize=20)
    sns.despine()
    plt.tight_layout()

    plt.savefig("../plots/res.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    N = 10 ** 3
    repeat = 10 ** 3
    cs = [round(float(x), 2) for x in np.linspace(1, 50, num=100)]
    alphas = [round(float(x), 2) for x in np.linspace(1, 15, num=100)]

    p = perform_experiment(N, repeat, cs, alphas)
    plot_prob(p, cs, alphas)
