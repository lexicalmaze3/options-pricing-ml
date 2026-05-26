import sys
import os
import json
import pickle

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(__file__))
from models.mlp import OptionsMLP
from models.transformer import OptionsTransformer
from data.generate import bs_european, binomial_american

SEED = 42
BATCH_SIZE = 8192
BASELINE_BATCH = 50_000


# ── data ─────────────────────────────────────────────────────────────────────

def load_test_set():
    df = pd.read_csv('data/options_dataset.csv')
    df['type_encoded'] = (df['type'] == 'call').astype(float)
    df['style_encoded'] = (df['style'] == 'american').astype(float)

    feature_cols = ['spot', 'strike', 'tte', 'rate', 'vol', 'dividend', 'type_encoded', 'style_encoded']
    X = df[feature_cols].values.astype(np.float32)
    y = df['price'].values.astype(np.float32)
    style = df['style_encoded'].values

    X_tmp, X_test, y_tmp, y_test, _, s_test = train_test_split(
        X, y, style, test_size=0.10, random_state=SEED, stratify=style
    )
    _, X_val, _, _ = train_test_split(X_tmp, y_tmp, test_size=0.1111, random_state=SEED, stratify=_)

    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    X_test_sc = scaler.transform(X_test).astype(np.float32)

    # Rebuild test DataFrame with original (unscaled) features
    test_df = df.iloc[df.index[~df.index.isin(
        pd.read_csv('data/options_dataset.csv').index  # placeholder — use mask approach below
    )]]
    return X_test, X_test_sc, y_test, s_test, df


def build_test_df():
    """Recreate the exact test split and return a DataFrame with all original columns."""
    df = pd.read_csv('data/options_dataset.csv')
    df['type_encoded'] = (df['type'] == 'call').astype(float)
    df['style_encoded'] = (df['style'] == 'american').astype(float)

    feature_cols = ['spot', 'strike', 'tte', 'rate', 'vol', 'dividend', 'type_encoded', 'style_encoded']
    X = df[feature_cols].values.astype(np.float32)
    y = df['price'].values.astype(np.float32)
    style = df['style_encoded'].values
    idx = np.arange(len(df))

    idx_tmp, idx_test, _, _, s_tmp, _ = train_test_split(
        idx, y, style, test_size=0.10, random_state=SEED, stratify=style
    )
    _, _, y_tmp = y[idx_tmp], None, y[idx_tmp]
    s_tmp_arr = style[idx_tmp]
    idx_train, idx_val = train_test_split(
        idx_tmp, test_size=0.1111, random_state=SEED, stratify=s_tmp_arr
    )

    test_df = df.iloc[idx_test].copy().reset_index(drop=True)
    X_test = X[idx_test]
    y_test = y[idx_test]

    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    X_test_sc = scaler.transform(X_test).astype(np.float32)

    return test_df, X_test_sc, y_test


# ── inference ─────────────────────────────────────────────────────────────────

def predict(model, X_sc, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_sc), BATCH_SIZE):
            xb = torch.from_numpy(X_sc[i:i + BATCH_SIZE]).to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)


def baseline_prices(df):
    spots     = df['spot'].values
    strikes   = df['strike'].values
    ttes      = df['tte'].values
    rates     = df['rate'].values
    vols      = df['vol'].values
    dividends = df['dividend'].values
    opt_types = df['type'].values
    styles    = df['style'].values

    prices = np.zeros(len(df))

    eu = styles == 'european'
    if eu.any():
        prices[eu] = bs_european(spots[eu], strikes[eu], ttes[eu],
                                 rates[eu], vols[eu], dividends[eu], opt_types[eu])

    am_idx = np.where(~eu)[0]
    n_am = len(am_idx)
    for start in range(0, n_am, BASELINE_BATCH):
        end = min(start + BASELINE_BATCH, n_am)
        idx = am_idx[start:end]
        prices[idx] = binomial_american(spots[idx], strikes[idx], ttes[idx],
                                        rates[idx], vols[idx], dividends[idx], opt_types[idx])
    return prices


# ── metrics ───────────────────────────────────────────────────────────────────

def metrics(actual, predicted):
    err = predicted - actual
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    r2   = float(r2_score(actual, predicted))
    maxe = float(np.max(np.abs(err)))
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'max_error': maxe}


def breakdown(df, actual, predicted, label):
    result = {}
    err = np.abs(predicted - actual)

    moneyness = df['strike'].values / df['spot'].values
    df = df.copy()
    df['_moneyness'] = moneyness
    df['_err'] = err

    for col, cats in [
        ('style', ['european', 'american']),
        ('type',  ['call', 'put']),
    ]:
        result[col] = {}
        for cat in cats:
            mask = df[col].values == cat
            if mask.sum() == 0:
                continue
            result[col][cat] = metrics(actual[mask], predicted[mask])

    result['moneyness'] = {}
    for cat, lo, hi in [('OTM', 1.1, np.inf), ('ATM', 0.9, 1.1), ('ITM', 0.0, 0.9)]:
        mask = (moneyness > lo) & (moneyness <= hi) if lo > 0 else (moneyness >= lo) & (moneyness < hi)
        if cat == 'OTM':
            mask = moneyness > 1.1
        elif cat == 'ATM':
            mask = (moneyness >= 0.9) & (moneyness <= 1.1)
        else:
            mask = moneyness < 0.9
        if mask.sum() == 0:
            continue
        result['moneyness'][cat] = metrics(actual[mask], predicted[mask])

    result['expiry'] = {}
    for cat, lo, hi in [('short', 0, 30), ('medium', 30, 180), ('long', 180, 9999)]:
        mask = (df['tte'].values >= lo) & (df['tte'].values < hi)
        if mask.sum() == 0:
            continue
        result['expiry'][cat] = metrics(actual[mask], predicted[mask])

    return result


# ── plots ─────────────────────────────────────────────────────────────────────

STYLE_COLORS = {'european': '#4C72B0', 'american': '#DD8452'}


def plot_scatter(actual, predicted, styles, title, path, n_sample=5000):
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(actual), size=min(n_sample, len(actual)), replace=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    for style_val, color in STYLE_COLORS.items():
        mask = styles[idx] == style_val
        ax.scatter(actual[idx][mask], predicted[idx][mask],
                   alpha=0.35, s=8, color=color, label=style_val)
    lo, hi = actual[idx].min(), actual[idx].max()
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1, label='perfect')
    ax.set_xlabel('Actual price ($)')
    ax.set_ylabel('Predicted price ($)')
    ax.set_title(title)
    ax.legend(markerscale=2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_error_dist(actuals, preds_dict, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {'MLP': '#4C72B0', 'Transformer': '#DD8452', 'Baseline': '#55A868'}
    for name, pred in preds_dict.items():
        err = pred - actuals
        ax.hist(err, bins=200, range=(-20, 20), alpha=0.5,
                density=True, label=name, color=colors[name])
    ax.axvline(0, color='black', lw=1, ls='--')
    ax.set_xlabel('Prediction error ($)')
    ax.set_ylabel('Density')
    ax.set_title('Error distribution (clipped to ±$20)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_vol_smile(df, actual, preds_mlp, preds_baseline, path):
    moneyness = df['strike'].values / df['spot'].values
    edges = np.linspace(0.7, 1.3, 11)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mlp_mae, base_mae = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (moneyness >= lo) & (moneyness < hi)
        if mask.sum() == 0:
            mlp_mae.append(np.nan); base_mae.append(np.nan)
            continue
        mlp_mae.append(np.mean(np.abs(preds_mlp[mask] - actual[mask])))
        base_mae.append(np.mean(np.abs(preds_baseline[mask] - actual[mask])))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(centers, mlp_mae,  'o-', color='#4C72B0', label='MLP')
    ax.plot(centers, base_mae, 's--', color='#55A868', label='Baseline')
    ax.axvline(1.0, color='grey', lw=0.8, ls=':')
    ax.set_xlabel('Strike / Spot (moneyness)')
    ax.set_ylabel('Mean Absolute Error ($)')
    ax.set_title('MAE by moneyness bucket (vol smile proxy)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_early_exercise(df, preds_mlp, preds_baseline, path):
    am = df['style'].values == 'american'
    tte_am    = df['tte'].values[am]
    mlp_am    = preds_mlp[am]
    bs_am     = preds_baseline[am]
    premium   = mlp_am - bs_am

    moneyness = (df['strike'].values / df['spot'].values)[am]
    mon_cats  = np.where(moneyness > 1.1, 'OTM',
                np.where(moneyness < 0.9, 'ITM', 'ATM'))

    rng = np.random.default_rng(SEED)
    n   = min(8000, len(tte_am))
    idx = rng.choice(len(tte_am), size=n, replace=False)

    cat_colors = {'OTM': '#4C72B0', 'ATM': '#55A868', 'ITM': '#DD8452'}
    fig, ax = plt.subplots(figsize=(8, 5))
    for cat, color in cat_colors.items():
        mask = mon_cats[idx] == cat
        ax.scatter(tte_am[idx][mask], premium[idx][mask],
                   alpha=0.25, s=6, color=color, label=cat)
    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.set_xlabel('Time to expiry (days)')
    ax.set_ylabel('MLP price − Baseline price ($)')
    ax.set_title('Early exercise premium learned by MLP (American options)')
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading test set...")
    test_df, X_test_sc, y_test = build_test_df()
    print(f"Test set: {len(test_df):,} rows")

    print("Loading models...")
    mlp = OptionsMLP()
    mlp.load_state_dict(torch.load('models/checkpoints/mlp_best.pt', weights_only=False)['model_state_dict'])
    mlp = mlp.to(device)

    tfm = OptionsTransformer()
    tfm.load_state_dict(torch.load('models/checkpoints/transformer_best.pt', weights_only=False)['model_state_dict'])
    tfm = tfm.to(device)

    print("Running MLP inference...")
    preds_mlp = predict(mlp, X_test_sc, device)

    print("Running Transformer inference...")
    preds_tfm = predict(tfm, X_test_sc, device)

    print("Computing baseline prices (BS + binomial)...")
    preds_base = baseline_prices(test_df)

    actual = y_test

    print("Computing metrics...")
    all_metrics = {
        'MLP':         {'overall': metrics(actual, preds_mlp),
                        'breakdown': breakdown(test_df, actual, preds_mlp, 'MLP')},
        'Transformer': {'overall': metrics(actual, preds_tfm),
                        'breakdown': breakdown(test_df, actual, preds_tfm, 'Transformer')},
        'Baseline':    {'overall': metrics(actual, preds_base),
                        'breakdown': breakdown(test_df, actual, preds_base, 'Baseline')},
    }

    os.makedirs('results', exist_ok=True)
    with open('results/metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print("Saved results/metrics.json")

    os.makedirs('results/plots', exist_ok=True)

    print("Generating plots...")
    styles = test_df['style'].values

    plot_scatter(actual, preds_mlp, styles,
                 'MLP: Predicted vs Actual', 'results/plots/scatter_mlp.png')
    plot_scatter(actual, preds_tfm, styles,
                 'Transformer: Predicted vs Actual', 'results/plots/scatter_transformer.png')
    plot_error_dist(actual, {'MLP': preds_mlp, 'Transformer': preds_tfm, 'Baseline': preds_base},
                    'results/plots/error_dist.png')
    plot_vol_smile(test_df, actual, preds_mlp, preds_base,
                   'results/plots/vol_smile.png')
    plot_early_exercise(test_df, preds_mlp, preds_base,
                        'results/plots/early_exercise_premium.png')
    print("Saved 5 plots to results/plots/")

    # ── summary table ─────────────────────────────────────────────────────────
    print()
    print("Model       | MAE    | RMSE   | R²")
    print("------------|--------|--------|--------")
    for name in ('MLP', 'Transformer', 'Baseline'):
        m = all_metrics[name]['overall']
        print(f"{name:<11} | ${m['mae']:.3f} | ${m['rmse']:.3f} | {m['r2']:.6f}")

    print()
    print("── Breakdown by style ──")
    for name in ('MLP', 'Transformer', 'Baseline'):
        bd = all_metrics[name]['breakdown']['style']
        eu = bd.get('european', {}); am = bd.get('american', {})
        print(f"  {name}: European MAE=${eu.get('mae',0):.3f}  American MAE=${am.get('mae',0):.3f}")

    print()
    print("── Breakdown by moneyness ──")
    for name in ('MLP', 'Transformer', 'Baseline'):
        bd = all_metrics[name]['breakdown']['moneyness']
        row = '  ' + name + ': ' + '  '.join(
            f"{k} MAE=${v['mae']:.3f}" for k, v in bd.items()
        )
        print(row)

    print()
    print("── Breakdown by expiry ──")
    for name in ('MLP', 'Transformer', 'Baseline'):
        bd = all_metrics[name]['breakdown']['expiry']
        row = '  ' + name + ': ' + '  '.join(
            f"{k} MAE=${v['mae']:.3f}" for k, v in bd.items()
        )
        print(row)


if __name__ == '__main__':
    main()
