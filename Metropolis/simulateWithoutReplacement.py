import numpy as np
import matplotlib.pyplot as plt


def generate_time_series(n_steps=100, mu=0.0, sigma=1.0, noise_pool_size=500, seed=None, replace=False):
    """
    Simulates a time series where the noise is Gaussian-distributed
    and sampled either with or without replacement from a finite pool.

    Parameters:
    - n_steps: length of time series
    - mu: mean of the Gaussian noise
    - sigma: standard deviation of the Gaussian noise
    - noise_pool_size: size of the pre-generated noise pool
    - seed: random seed for reproducibility
    - replace: whether to sample with replacement
    """
    if seed is not None:
        np.random.seed(seed)

    if not replace and n_steps > noise_pool_size:
        raise ValueError("n_steps must be less than or equal to noise_pool_size for sampling without replacement")

    # Create a Gaussian noise pool
    noise_pool = np.random.normal(loc=mu, scale=sigma, size=noise_pool_size)

    # Sample noise with or without replacement
    noise = np.random.choice(noise_pool, size=n_steps, replace=replace)

    # Simulate time series (e.g., simple random walk)
    time_series = np.cumsum(noise)

    return time_series, noise


if __name__ == '__main__':
    # Parameters
    n = 3000
    mu = 0.1
    sigma = 5.0
    pool_size = 3000
    seed = 51273

    # Time series without replacement
    ts_no_replace, noise_no_replace = generate_time_series(n_steps=n, mu=mu, sigma=sigma,
                                                            noise_pool_size=pool_size, seed=seed, replace=False)

    # Time series with replacement
    ts_with_replace, noise_with_replace = generate_time_series(n_steps=n, mu=mu, sigma=sigma,
                                                                noise_pool_size=pool_size, seed=seed, replace=True)

    # Plot
    plt.figure(figsize=(14, 6))

    plt.subplot(2, 2, 1)
    plt.plot(ts_no_replace)
    plt.title("Time Series (No Replacement)")

    plt.subplot(2, 2, 2)
    plt.plot(ts_with_replace)
    plt.title("Time Series (With Replacement)")

    plt.subplot(2, 2, 3)
    plt.hist(noise_no_replace, bins=20, edgecolor='black', alpha=0.7)
    plt.title("Noise Histogram (No Replacement)")

    plt.subplot(2, 2, 4)
    plt.hist(noise_with_replace, bins=20, edgecolor='black', alpha=0.7)
    plt.title("Noise Histogram (With Replacement)")

    plt.tight_layout()
    plt.show()
    plt.show()
