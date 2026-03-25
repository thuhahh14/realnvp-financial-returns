# realnvp-financial-returns
# RealNVP for Financial Time Series Density Estimation

## Project topic
Nghiên cứu khả năng ước lượng mật độ xác suất của chuỗi lợi suất tài chính bằng mô hình RealNVP.

## Data source
S&P 500 (SP500) from FRED:  
https://fred.stlouisfed.org/series/SP500

The downloaded CSV file should be placed in:

data/SP500.csv

## Main steps
1. Load SP500 data from CSV
2. Compute log-return
3. Visualize Histogram, KDE, Normal comparison, QQ plot
4. Create sliding windows
5. Train RealNVP model
6. Evaluate with Negative Log-Likelihood
7. Compare density of real data and generated samples

## File structure
- `main.py`: run the whole pipeline
- `data_preprocessing.py`: load data and create features
- `model_realnvp.py`: define RealNVP model
- `train.py`: training loop
- `evaluate.py`: plotting and evaluation
- `requirements.txt`: dependencies

## Run
```bash
python main.py

# 3. requirements.txt

Tạo file `requirements.txt`:

```txt
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
torch
