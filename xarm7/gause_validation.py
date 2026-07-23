import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

data = np.array([
    1.25,2.25,-4.75,-2.75,-10,-0.5,-1.5,-4,2.75,-5,
    -0.5,1.25,2.75,2.25,-8,0,-0.25,-10.75,-4,-4.25,
    3.25,3.25,2.25,-1.75,2,1,0.5,-8.75,-5.25,-6.25,
    0.75,0.75,-4.75,0.25,0,0,-5.25,2.25,1.5,-5,
    1.5,-1.5,3.5,-8,-2.5,3.75,-3.25,2.75,1.25,-3.75,
    -2,-0.5,3.25,0.5,2,-2.25,0,4.75,-7.25,1.25,
    -3,3.5,0,0.5,-3.75,3.25,-2,-7,1.25,-2.5
])

mu = np.mean(data)
sigma = np.std(data, ddof=1)

print(f"平均: {mu:.3f}")
print(f"標準偏差: {sigma:.3f}")

# x軸
x = np.linspace(data.min() - 1, data.max() + 1, 300)

# 正規分布
y = norm.pdf(x, mu, sigma)

plt.figure(figsize=(8, 5))

# ヒストグラム
plt.hist(data, bins=10, density=True, alpha=0.6, edgecolor="black", label="Data")

# ガウス分布
plt.plot(x, y, linewidth=2, label=f"Normal fit\nμ={mu:.2f}, σ={sigma:.2f}")

plt.xlabel("Error")
plt.ylabel("Probability density")
plt.title("Histogram with Gaussian Fit")
plt.legend()
plt.grid(True)
plt.show()