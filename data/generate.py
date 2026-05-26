import numpy as np
import pandas as pd
from scipy.stats import norm

SEED = 42
N_ROWS = 500_000
TREE_STEPS = 100
BATCH_SIZE = 50_000


def generate_params(rng):
    spot = rng.uniform(50, 500, N_ROWS)
    strike = spot * rng.uniform(0.7, 1.3, N_ROWS)
    tte = rng.uniform(7, 365, N_ROWS)
    rate = rng.uniform(0.0, 0.10, N_ROWS)
    vol = rng.uniform(0.05, 0.80, N_ROWS)
    dividend = rng.uniform(0.0, 0.05, N_ROWS)
    opt_type = rng.choice(['call', 'put'], N_ROWS)
    style = rng.choice(['european', 'american'], N_ROWS)
    return spot, strike, tte, rate, vol, dividend, opt_type, style


def bs_european(spots, strikes, ttes, rates, vols, dividends, opt_types):
    T = ttes / 365.0
    n = len(spots)
    prices = np.zeros(n)
    is_call = opt_types == 'call'

    mask_T0 = T == 0
    if mask_T0.any():
        s, k = spots[mask_T0], strikes[mask_T0]
        prices[mask_T0] = np.where(is_call[mask_T0], np.maximum(s - k, 0), np.maximum(k - s, 0))

    mask_v0 = (vols == 0) & ~mask_T0
    if mask_v0.any():
        s, k = spots[mask_v0], strikes[mask_v0]
        t, r, q = T[mask_v0], rates[mask_v0], dividends[mask_v0]
        pv_s = s * np.exp(-q * t)
        pv_k = k * np.exp(-r * t)
        prices[mask_v0] = np.where(is_call[mask_v0], np.maximum(pv_s - pv_k, 0), np.maximum(pv_k - pv_s, 0))

    mask = ~mask_T0 & ~mask_v0
    if mask.any():
        s, k = spots[mask], strikes[mask]
        t, r, v, q = T[mask], rates[mask], vols[mask], dividends[mask]
        sqrt_t = np.sqrt(t)
        d1 = (np.log(s / k) + (r - q + 0.5 * v ** 2) * t) / (v * sqrt_t)
        d2 = d1 - v * sqrt_t
        call = s * np.exp(-q * t) * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
        put = k * np.exp(-r * t) * norm.cdf(-d2) - s * np.exp(-q * t) * norm.cdf(-d1)
        prices[mask] = np.where(is_call[mask], call, put)

    return prices


def binomial_american(spots, strikes, ttes, rates, vols, dividends, opt_types):
    n = len(spots)
    N = TREE_STEPS
    T = ttes / 365.0
    dt = T / N

    u = np.exp(vols * np.sqrt(dt))
    disc = np.exp(-rates * dt)
    p = np.clip((np.exp((rates - dividends) * dt) - (1.0 / u)) / (u - 1.0 / u), 0.0, 1.0)
    q_prob = 1.0 - p
    is_call = opt_types == 'call'

    log_u = np.log(u)
    log_spots = np.log(spots)

    j = np.arange(N + 1)
    log_S_final = log_spots[np.newaxis, :] + (2 * j[:, np.newaxis] - N) * log_u[np.newaxis, :]
    S = np.exp(log_S_final)

    payoff_call = np.maximum(S - strikes[np.newaxis, :], 0.0)
    payoff_put = np.maximum(strikes[np.newaxis, :] - S, 0.0)
    V = np.where(is_call[np.newaxis, :], payoff_call, payoff_put)

    for step in range(N - 1, -1, -1):
        S = S[:step + 1] * u[np.newaxis, :]
        V_cont = disc[np.newaxis, :] * (
            p[np.newaxis, :] * V[1:step + 2] + q_prob[np.newaxis, :] * V[:step + 1]
        )
        intrinsic = np.where(
            is_call[np.newaxis, :],
            np.maximum(S - strikes[np.newaxis, :], 0.0),
            np.maximum(strikes[np.newaxis, :] - S, 0.0),
        )
        V = np.maximum(V_cont, intrinsic)

    return V[0]


def main():
    rng = np.random.default_rng(SEED)
    spot, strike, tte, rate, vol, dividend, opt_type, style = generate_params(rng)

    prices = np.zeros(N_ROWS)

    eu_mask = style == 'european'
    if eu_mask.any():
        print(f"Pricing {eu_mask.sum():,} European options (Black-Scholes)...")
        prices[eu_mask] = bs_european(
            spot[eu_mask], strike[eu_mask], tte[eu_mask],
            rate[eu_mask], vol[eu_mask], dividend[eu_mask], opt_type[eu_mask],
        )

    am_mask = ~eu_mask
    if am_mask.any():
        n_am = am_mask.sum()
        print(f"Pricing {n_am:,} American options (binomial tree, {TREE_STEPS} steps)...")
        am_idx = np.where(am_mask)[0]
        am_prices = np.zeros(n_am)
        for start in range(0, n_am, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_am)
            idx = am_idx[start:end]
            am_prices[start:end] = binomial_american(
                spot[idx], strike[idx], tte[idx],
                rate[idx], vol[idx], dividend[idx], opt_type[idx],
            )
            print(f"  batch {start // BATCH_SIZE + 1}/{(n_am + BATCH_SIZE - 1) // BATCH_SIZE} done")
        prices[am_mask] = am_prices

    df = pd.DataFrame({
        'spot': spot,
        'strike': strike,
        'tte': tte,
        'rate': rate,
        'vol': vol,
        'dividend': dividend,
        'type': opt_type,
        'style': style,
        'price': prices,
    })

    df.to_csv('data/options_dataset.csv', index=False)

    n_eu = eu_mask.sum()
    n_am = am_mask.sum()
    n_call = (opt_type == 'call').sum()
    n_put = (opt_type == 'put').sum()
    n_bad = (prices <= 0).sum()

    print(f"\nTotal rows: {N_ROWS:,}")
    print(f"European: {n_eu:,}  American: {n_am:,}")
    print(f"Calls: {n_call:,}  Puts: {n_put:,}")
    print(f"Price stats — mean: {prices.mean():.4f}  min: {prices.min():.4f}  max: {prices.max():.4f}  std: {prices.std():.4f}")
    if n_bad:
        print(f"WARNING: {n_bad:,} rows with price <= 0")


if __name__ == '__main__':
    main()
