import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto

file_path = '../data/insurance_data.csv'
df = pd.read_csv(file_path)

variable = 'CLAIM_AMOUNT'
data = df[variable].dropna()

n = len(data)
threshold = np.percentile(data, 90)
exceedances = data[data > threshold] - threshold
#exceedances = np.array(sorted(data, reverse=True)[:int(n ** 0.5)])
#exceedances = np.array(sorted(data, reverse=True)[:int(n ** (2/3) / np.log(np.log(n)))])

shape, loc, scale = genpareto.fit(exceedances)

x = np.linspace(0, exceedances.max(), 100)
pdf = genpareto.pdf(x, shape, loc=0, scale=scale)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(exceedances, bins=50, density=True, alpha=0.6, color='g')
plt.plot(x, pdf, 'r-', lw=2)
plt.title('Exceedances and Fitted GPD')
plt.xlabel('Exceedance over Threshold')
plt.ylabel('Density')

plt.subplot(1, 2, 2)
empirical = np.sort(exceedances)
theoretical = genpareto.ppf(np.linspace(0, 1, len(empirical)), shape, loc=0, scale=scale)
plt.scatter(theoretical, empirical, alpha=0.6)
plt.plot([empirical.min(), empirical.max()], [empirical.min(), empirical.max()], 'r--', lw=2)
plt.title('QQ Plot')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Empirical Quantiles')

plt.tight_layout()
plt.savefig("../plots/pot_test.png", dpi=200)
plt.show()

print(f"Fitted GPD Parameters:\nShape: {shape}, Scale: {scale}")
