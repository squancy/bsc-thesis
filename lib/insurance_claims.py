import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mm_est import mm_est
from ml_est import ml_est

sns.set(style="whitegrid", palette="colorblind", font_scale=1.2)
sns.set_context("paper")

def pareto_pdf(x, alpha, c):
    return np.where(x >= c, (alpha * c**alpha) / x**(alpha + 1), 0)

def pareto_cdf(x, alpha, c):
    return np.where(x < c, 0, 1 - (c / x)**alpha)

def empirical_cdf(t, data):
    n = len(data)
    return np.array([np.sum(data <= ti) / n for ti in t])

def calc_mm_ml(claims, typ, x):
    alpha, c = mm_est(claims) if typ == 'mm' else ml_est(claims)
    pdf = pareto_pdf(x, alpha, c)
    cdf = pareto_cdf(x, alpha, c)
    return (pdf, cdf, alpha, c)

def plot_results(pdf_mm, cdf_mm, alpha_mm, c_mm, pdf_ml, cdf_ml, alpha_ml, c_ml, x):
    X_sorted = np.sort(claims)
    ecdf_y = empirical_cdf(X_sorted, X_sorted)
    op = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    sns.lineplot(x=x, y=pdf_mm, ax=axes[0], label=f'Pareto PDF (MM) (α={alpha_mm:.2f}, c={c_mm:.2f})', color='green')
    sns.lineplot(x=x, y=pdf_ml, ax=axes[0], label=f'Pareto PDF (ML) (α={alpha_ml:.2f}, c={c_ml:.2f})', color='blue')
    axes[0].axvline(x=c_mm, color='red', linestyle='--', label=f'Scale parameter (MM) c={c_mm:.2f}', alpha=op)
    axes[0].axvline(x=c_ml, color='black', linestyle='--', label=f'Scale parameter (ML) c={c_ml:.2f}', alpha=op)
    axes[0].set_title('Pareto Distribution - PDF', fontsize=16)
    axes[0].set_xlabel('x', fontsize=14)
    axes[0].set_ylabel('Density', fontsize=14)
    axes[0].legend()
    axes[0].grid(True)

    sns.lineplot(x=x, y=cdf_mm, ax=axes[1], label=f'Pareto CDF (MM) (α={alpha_mm:.2f}, c={c_mm:.2f})', color='green')
    sns.lineplot(x=x, y=cdf_ml, ax=axes[1], label=f'Pareto CDF (ML) (α={alpha_ml:.2f}, c={c_ml:.2f})', color='blue')

    sns.lineplot(x=X_sorted, y=ecdf_y, ax=axes[1], label='Empirical CDF', color='orange', linestyle='--')
    axes[1].axvline(x=c_mm, color='red', linestyle='--', label=f'Scale parameter (MM) c={c_mm:.2f}', alpha=op)
    axes[1].axvline(x=c_ml, color='black', linestyle='--', label=f'Scale parameter (ML) c={c_ml:.2f}', alpha=op)
    axes[1].set_title('Pareto Distribution - CDF', fontsize=16)
    axes[1].set_xlabel('x', fontsize=14)
    axes[1].set_ylabel('Cumulative Probability', fontsize=14)
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("../plots/insurance.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    cutoff = 5000
    df = pd.read_csv('../data/insurance_data.csv')
    claims = np.array(df['CLAIM_AMOUNT'])
    claims = claims[claims > cutoff]
    x = np.linspace(min(claims), np.max(claims), 1000)
    plot_results(*calc_mm_ml(claims, 'mm', x), *calc_mm_ml(claims, 'ml', x), x)
