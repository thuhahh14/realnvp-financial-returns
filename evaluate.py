import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, gaussian_kde, probplot


def ensure_output_dir(output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)


def plot_histogram_kde(returns, output_dir="outputs"):
    ensure_output_dir(output_dir)

    plt.figure(figsize=(8, 5))
    sns.histplot(returns, bins=60, stat="density", kde=True)
    plt.title("Histogram and KDE of SP500 Log-Returns")
    plt.xlabel("Log-return")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/histogram_kde.png", dpi=300)
    plt.close()


def plot_normal_comparison(returns, output_dir="outputs"):
    ensure_output_dir(output_dir)

    mu = np.mean(returns)
    sigma = np.std(returns)
    x = np.linspace(returns.min(), returns.max(), 1000)

    plt.figure(figsize=(8, 5))
    sns.histplot(returns, bins=60, stat="density", alpha=0.5, label="Real data")
    plt.plot(x, norm.pdf(x, mu, sigma), label="Normal distribution")
    plt.title("SP500 Log-Returns vs Normal Distribution")
    plt.xlabel("Log-return")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/normal_comparison.png", dpi=300)
    plt.close()


def plot_qq(returns, output_dir="outputs"):
    ensure_output_dir(output_dir)

    plt.figure(figsize=(5, 5))
    probplot(returns, dist="norm", plot=plt)
    plt.title("QQ Plot of SP500 Log-Returns")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/qq_plot.png", dpi=300)
    plt.close()


def plot_losses(train_losses, test_losses, output_dir="outputs"):
    ensure_output_dir(output_dir)

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train NLL")
    plt.plot(test_losses, label="Test NLL")
    plt.title("Training Curve of RealNVP")
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log-Likelihood")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_curve.png", dpi=300)
    plt.close()


def plot_density_comparison(real_flat, sample_flat, output_dir="outputs"):
    ensure_output_dir(output_dir)

    plt.figure(figsize=(8, 5))
    sns.kdeplot(real_flat, label="Real data", fill=True, alpha=0.4)
    sns.kdeplot(sample_flat, label="RealNVP samples", fill=True, alpha=0.4)
    plt.title("Density Comparison: Real Data vs RealNVP Samples")
    plt.xlabel("Log-return")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/density_comparison.png", dpi=300)
    plt.close()


def plot_time_series(real_flat, sample_flat, output_dir="outputs"):
    ensure_output_dir(output_dir)

    n = len(real_flat)
    plt.figure(figsize=(12, 5))
    plt.plot(real_flat, label="Real Data", color="steelblue", linewidth=0.8)
    plt.plot(sample_flat[:n], label="Generated Data", color="orange",
             linestyle="--", linewidth=1.2)
    plt.title("Real vs Generated Time Series")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/time_series.png", dpi=300)
    plt.close()


def plot_kde_comparison(real_flat, sample_flat, output_dir="outputs"):
    ensure_output_dir(output_dir)

    kde_real = gaussian_kde(real_flat)
    kde_sample = gaussian_kde(sample_flat)

    x_grid = np.linspace(
        min(real_flat.min(), sample_flat.min()),
        max(real_flat.max(), sample_flat.max()),
        1000
    )

    plt.figure(figsize=(8, 5))
    plt.plot(x_grid, kde_real(x_grid), label="KDE of real data")
    plt.plot(x_grid, kde_sample(x_grid), label="KDE of RealNVP samples")
    plt.title("KDE Comparison")
    plt.xlabel("Log-return")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/kde_comparison.png", dpi=300)
    plt.close()
