import math
import random
import sys
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from models.mlp import OptionsMLP
from models.transformer import OptionsTransformer

SEED = 42
BATCH_SIZE = 4096
LR = 1e-3
WEIGHT_DECAY = 1e-4
T_MAX = 50
PATIENCE = 10
MAX_EPOCHS = 200


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data():
    df = pd.read_csv('data/options_dataset.csv')
    df['type_encoded'] = (df['type'] == 'call').astype(float)
    df['style_encoded'] = (df['style'] == 'american').astype(float)

    feature_cols = ['spot', 'strike', 'tte', 'rate', 'vol', 'dividend', 'type_encoded', 'style_encoded']
    X = df[feature_cols].values.astype(np.float32)
    y = df['price'].values.astype(np.float32)
    style = df['style_encoded'].values

    X_tmp, X_test, y_tmp, y_test, s_tmp, _ = train_test_split(
        X, y, style, test_size=0.10, random_state=SEED, stratify=style
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.1111, random_state=SEED, stratify=s_tmp
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    os.makedirs('models', exist_ok=True)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    def to_ds(X, y):
        return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

    return (
        to_ds(X_train, y_train),
        to_ds(X_val, y_val),
        to_ds(X_test, y_test),
        len(X_train), len(X_val), len(X_test),
    )


def train_model(name, model, train_ds, val_ds, n_train, n_val, device, checkpoint_path, log_path):
    set_seed(SEED)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    log_rows = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    print(f"\n--- Training {name} ---")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_b)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                val_loss += criterion(model(X_b), y_b).item() * len(X_b)
        val_loss /= n_val

        scheduler.step()
        log_rows.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
        print(f"Epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_loss': val_loss}, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print(f"Best epoch: {best_epoch}  val_loss={best_val_loss:.4f}")
    return best_epoch, best_val_loss


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs('models/checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    train_ds, val_ds, _, n_train, n_val, _ = load_data()
    print(f"Split — train: {n_train:,}  val: {n_val:,}")

    mlp_epoch, mlp_val = train_model(
        'MLP', OptionsMLP(),
        train_ds, val_ds, n_train, n_val, device,
        'models/checkpoints/mlp_best.pt',
        'results/training_log_mlp.csv',
    )

    tfm_epoch, tfm_val = train_model(
        'Transformer', OptionsTransformer(),
        train_ds, val_ds, n_train, n_val, device,
        'models/checkpoints/transformer_best.pt',
        'results/training_log_transformer.csv',
    )

    print("\nModel        | Best Epoch | Val RMSE")
    print("-------------|------------|----------")
    print(f"MLP          | {mlp_epoch:<10} | ${math.sqrt(mlp_val):.2f}")
    print(f"Transformer  | {tfm_epoch:<10} | ${math.sqrt(tfm_val):.2f}")


if __name__ == '__main__':
    main()
