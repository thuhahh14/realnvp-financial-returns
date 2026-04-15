import torch
import numpy as np

from data_preprocessing import (
    load_fred_sp500,
    compute_log_return,
    create_sliding_windows,
    standardize_train_test
)
from model_realnvp import RealNVP
from train import train_realnvp
from evaluate import (
    plot_histogram_kde,
    plot_normal_comparison,
    plot_qq,
    plot_losses,
    plot_density_comparison,
    plot_kde_comparison,
    plot_time_series
)


def main():
    # =========================
    # 1. Load data
    # =========================
    csv_path = "data/SP500.csv"
    df = load_fred_sp500(csv_path)
    df = compute_log_return(df)

    returns = df["log_return"].values

    print("Loaded SP500 data successfully.")
    print("Number of return observations:", len(returns))
    print(df.head())

    # =========================
    # 2. Initial plots
    # =========================
    plot_histogram_kde(returns)
    plot_normal_comparison(returns)
    plot_qq(returns)

    # =========================
    # 3. Create sliding windows
    # =========================
    window_size = 5
    X = create_sliding_windows(returns, window_size=window_size)
    print("Windowed data shape:", X.shape)

    # =========================
    # 4. Train/test split + standardization
    # =========================
    X_train_std, X_test_std, mean, std = standardize_train_test(X, test_size=0.2)

    X_train_tensor = torch.tensor(X_train_std, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_std, dtype=torch.float32)

    # =========================
    # 5. Build model
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = RealNVP(dim=window_size, n_coupling=6, hidden_dim=64)

    # =========================
    # 6. Train
    # =========================
    train_losses, test_losses = train_realnvp(
        model=model,
        X_train_tensor=X_train_tensor,
        X_test_tensor=X_test_tensor,
        epochs=300,
        lr=1e-3,
        device=device
    )

    plot_losses(train_losses, test_losses)

    # =========================
    # 7. Evaluate
    # =========================
    model.eval()
    with torch.no_grad():
        X_test_tensor_device = X_test_tensor.to(device)
        test_nll = -model.log_prob(X_test_tensor_device).mean().item()

        samples_std = model.sample(2000, device=device).cpu().numpy()

    samples = samples_std * std + mean

    real_flat = X[-len(X_test_std):].reshape(-1)
    sample_flat = samples.reshape(-1)

    plot_density_comparison(real_flat, sample_flat)
    plot_kde_comparison(real_flat, sample_flat)
    plot_time_series(real_flat, sample_flat)

    print(f"Final Test Negative Log-Likelihood: {test_nll:.6f}")
    print("All figures have been saved in the outputs/ folder.")


if __name__ == "__main__":
    main()
