import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

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


# ─────────────────────────────────────────────
# HELPER: Rolling Probabilistic Forecast
# ─────────────────────────────────────────────
def rolling_forecast(model, n_test, mean, std, n_samples=500, device="cpu"):
    """
    Sinh n_samples từ phân phối đã học của RealNVP,
    sau đó tổng hợp thành median forecast + prediction interval.

    Lưu ý: RealNVP trong bài này là unconditional flow —
    forecast mang tính probabilistic (phân phối), không phải point forecast.
    """
    model.eval()
    forecasts_median = []
    forecasts_q05    = []
    forecasts_q95    = []

    with torch.no_grad():
        for _ in range(n_test):
            # Sample n_samples điểm từ phân phối đã học
            samples_i = model.sample(n_samples, device=device).cpu().numpy()

            # Destandardize về không gian log-return gốc
            samples_i = samples_i * std + mean

            # Lấy chiều cuối của mỗi cửa sổ làm "dự báo t+1"
            pred = samples_i[:, -1]

            forecasts_median.append(np.median(pred))
            forecasts_q05.append(np.percentile(pred, 5))
            forecasts_q95.append(np.percentile(pred, 95))

    return (
        np.array(forecasts_median),
        np.array(forecasts_q05),
        np.array(forecasts_q95)
    )


# ─────────────────────────────────────────────
# HELPER: Plot Forecast with Prediction Interval
# ─────────────────────────────────────────────
def plot_forecast_interval(y_true, median_fc, q05_fc, q95_fc, save_path="outputs/forecast_interval.png"):
    import matplotlib.pyplot as plt
    import os
    os.makedirs("outputs", exist_ok=True)

    x = np.arange(len(y_true))

    plt.figure(figsize=(14, 5))
    plt.plot(x, y_true,      color="steelblue",  label="Actual log-return",    linewidth=1.2)
    plt.plot(x, median_fc,   color="darkorange",  label="Forecast (median)",    linewidth=1.2, linestyle="--")
    plt.fill_between(x, q05_fc, q95_fc, alpha=0.25, color="orange", label="90% Prediction Interval")

    plt.title("RealNVP Probabilistic Forecast vs Actual (Test Set)")
    plt.xlabel("Time step (test period)")
    plt.ylabel("Log-return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved forecast interval plot → {save_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():

    # ── 1. Load data ──────────────────────────
    csv_path = "data/SP500.csv"
    df = load_fred_sp500(csv_path)
    df = compute_log_return(df)
    returns = df["log_return"].values

    print("Loaded SP500 data successfully.")
    print("Number of return observations:", len(returns))
    print(df.head())

    # ── 2. Initial EDA plots ──────────────────
    plot_histogram_kde(returns)
    plot_normal_comparison(returns)
    plot_qq(returns)

    # ── 3. Sliding windows ────────────────────
    window_size = 5
    X = create_sliding_windows(returns, window_size=window_size)
    print("Windowed data shape:", X.shape)

    # ── 4. Train/test split + standardize ─────
    X_train_std, X_test_std, mean, std = standardize_train_test(X, test_size=0.2)

    X_train_tensor = torch.tensor(X_train_std, dtype=torch.float32)
    X_test_tensor  = torch.tensor(X_test_std,  dtype=torch.float32)

    # ── 5. Build model ────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = RealNVP(dim=window_size, n_coupling=6, hidden_dim=64)

    # ── 6. Train ──────────────────────────────
    train_losses, test_losses = train_realnvp(
        model=model,
        X_train_tensor=X_train_tensor,
        X_test_tensor=X_test_tensor,
        epochs=300,
        lr=1e-3,
        device=device
    )
    plot_losses(train_losses, test_losses)

    # ── 7. Density evaluation ─────────────────
    model.eval()
    with torch.no_grad():
        X_test_device = X_test_tensor.to(device)
        test_nll = -model.log_prob(X_test_device).mean().item()
        samples_std = model.sample(2000, device=device).cpu().numpy()

    samples = samples_std * std + mean

    real_flat   = X[-len(X_test_std):].reshape(-1)
    sample_flat = samples.reshape(-1)

    plot_density_comparison(real_flat, sample_flat)
    plot_kde_comparison(real_flat, sample_flat)
    plot_time_series(real_flat, sample_flat)

    print(f"\nFinal Test Negative Log-Likelihood: {test_nll:.6f}")

    # ── 8. Rolling Probabilistic Forecast ─────
    print("\nRunning rolling probabilistic forecast...")

    n_test = len(X_test_std)

    median_fc, q05_fc, q95_fc = rolling_forecast(
        model=model,
        n_test=n_test,
        mean=mean,
        std=std,
        n_samples=500,
        device=device
    )

    # Ground truth: giá trị thực (chiều cuối của mỗi cửa sổ test)
    y_true = X[-n_test:][:, -1]   # vẫn ở scale gốc (chưa standardize)

    # ── 9. Forecast Metrics ───────────────────
    mae      = mean_absolute_error(y_true, median_fc)
    rmse     = np.sqrt(mean_squared_error(y_true, median_fc))
    coverage = np.mean((y_true >= q05_fc) & (y_true <= q95_fc))

    print("\n" + "="*45)
    print("         FORECAST EVALUATION SUMMARY")
    print("="*45)
    print(f"  MAE              : {mae:.6f}")
    print(f"  RMSE             : {rmse:.6f}")
    print(f"  90% Coverage     : {coverage*100:.2f}%")
    print(f"  Test NLL         : {test_nll:.6f}")
    print("="*45)
    print("\nNote: RealNVP is an unconditional density estimator.")
    print("Forecast = median of 500 samples drawn from learned distribution.")
    print("Suitable for risk distribution analysis, NOT point forecasting.")

    # ── 10. Plot forecast interval ────────────
    plot_forecast_interval(y_true, median_fc, q05_fc, q95_fc)

    print("\nAll figures have been saved in the outputs/ folder.")


if __name__ == "__main__":
    main()
