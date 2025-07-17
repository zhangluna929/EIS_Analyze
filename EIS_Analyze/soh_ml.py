import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error


def load_data(batch_csv: Path, capacity_csv: Path | None):
    df = pd.read_csv(batch_csv)
    if capacity_csv is not None:
        cap_df = pd.read_csv(capacity_csv)
        df = df.merge(cap_df, on='cycle_number', how='inner')
    if 'capacity' not in df.columns:
        raise ValueError('capacity column not found; provide capacity CSV with cycle_number,capacity')
    return df


def train_model(df: pd.DataFrame, target: str = 'capacity'):
    feature_cols = [c for c in df.columns if c.startswith('p')]
    X = df[feature_cols].values
    y = df[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = GradientBoostingRegressor(random_state=0)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'Cross-val MAE: {-scores.mean():.3f} ± {scores.std():.3f}')
    print(f'Test R2: {r2_score(y_test, y_pred):.3f}   MAE: {mean_absolute_error(y_test, y_pred):.3f}')
    return model, feature_cols, X_test, y_test, y_pred


def plot_pred(y_test, y_pred):
    plt.figure(figsize=(5,5))
    plt.scatter(y_test, y_pred, c='b')
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, 'k--')
    plt.xlabel('Actual capacity')
    plt.ylabel('Predicted capacity')
    plt.title('SOH Prediction')
    plt.tight_layout()
    plt.savefig('soh_pred.png', dpi=300)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train ML model to predict battery SOH from EIS parameters')
    parser.add_argument('--batch', default='batch_fit_results.csv', help='CSV with fitted parameters')
    parser.add_argument('--capacity', default=None, help='CSV with cycle_number,capacity (optional if already in batch)')
    parser.add_argument('--model-out', default='soh_model.pkl', help='Output model file')
    args = parser.parse_args()

    df = load_data(Path(args.batch), Path(args.capacity) if args.capacity else None)
    model, feats, Xt, yt, yp = train_model(df)
    joblib.dump({'model': model, 'features': feats}, args.model_out)
    print(f'Model saved to {args.model_out}')
    plot_pred(yt, yp)


if __name__ == '__main__':
    main() 