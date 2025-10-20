import numpy as np
import matplotlib.pyplot as plt

##From Step 1
def simulate_galton_board(n, N):
    
    right_moves = np.random.binomial(n, 0.5, size=N)
    ##Counts how many balls had j right moves
    counts, _ = np.histogram(right_moves, bins=np.arange(n+2))
    return counts, right_moves

##From Step 2
from scipy.stats import binom

def binomial_distribution(n):
    k = np.arange(n+1)  #possible values of j (number of right moves)
    p_binom = binom.pmf(k, n, 0.5)
    return k, p_binom
    
##From Step 4
from scipy.stats import norm

def normal_approximation(n):
    k = np.arange(n+1)
    m = n / 2
    sigma = np.sqrt(n / 4)
    p_norm = norm.pdf(k, m, sigma)
    p_norm /= p_norm.sum()  # normalize to sum to 1
    return k, p_norm


##From Step 6
def compute_mse(empirical_counts, theoretical_probs, N):
    empirical_probs = empirical_counts / N
    mse = np.mean((empirical_probs - theoretical_probs)**2)
    return mse
    
##From Step 7
def mse_binomial_vs_normal(n, p=0.5):
    _, p_binom = binomial_distribution(n)
    _, p_norm = normal_approximation(n)
    mse = np.mean((p_binom - p_norm) ** 2)
    return mse
    
##Visualize
def plot_distributions(n, N, empirical_counts, theoretical_probs, mse):
    k = np.arange(n + 1)
    empirical_probs = empirical_counts / N
    plt.bar(k, empirical_probs, alpha=0.6, label='Simulation')
    plt.plot(k, theoretical_probs, 'o-', color='black', label='Binomial')
    _, p_norm = normal_approximation(n)
    plt.plot(k, p_norm, '-', color='red', label='Normal PDF')
    plt.title(f'n = {n}, N = {N}')
    plt.xlabel('Final position')
    plt.ylabel('Probability')
    plt.text(n * 0.6, max(empirical_probs) * 0.9, f"MSE = {mse:.6f}", fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    plt.legend()
    plt.grid(True)
    plt.show()

# Experiments: list with (n, N)
experiments = [(10, 1000), (20, 5000), (50, 10000)]

for n, N in experiments:
    counts, _ = simulate_galton_board(n, N)
    _, p_binom = binomial_distribution(n)
    mse_sim_vs_binom = compute_mse(counts, p_binom, N)
    mse_binom_vs_norm = mse_binomial_vs_normal(n)
    
    print(f"Experiment: n = {n}, N = {N}")
    print(f"MSE (Simulation vs Binomial): {mse_sim_vs_binom:.6f}")
    print(f"MSE (Binomial vs Normal): {mse_binom_vs_norm:.6f}")
    plot_distributions(n, N, counts, p_binom,mse_sim_vs_binom)



    