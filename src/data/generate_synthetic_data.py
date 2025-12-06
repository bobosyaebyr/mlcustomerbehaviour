

from pathlib import Path
import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "age",
    "income",
    "tenure_months",
    "transactions_last_30d",
    "amount_last_30d",
    "days_since_last_purchase",
    "is_active_app_user",
    "support_tickets_last_90d",
    "discount_rate_mean",
    "segment_code",
]

TARGET_COLUMN = "target_purchase_next_30d"


def generate_synthetic_data(
    n_samples: int = 50000,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    age = rng.integers(18, 70, size=n_samples)
    income = rng.normal(50000, 15000, size=n_samples).clip(15000, 200000)
    tenure_months = rng.integers(1, 120, size=n_samples)
    transactions_last_30d = rng.poisson(lam=2, size=n_samples)
    amount_last_30d = (
        transactions_last_30d * rng.normal(40, 20, size=n_samples).clip(5, 500)
    )
    days_since_last_purchase = rng.integers(0, 180, size=n_samples)
    is_active_app_user = rng.integers(0, 2, size=n_samples)
    support_tickets_last_90d = rng.poisson(lam=0.3, size=n_samples)
    discount_rate_mean = rng.uniform(0, 0.3, size=n_samples)
    segment_code = rng.integers(0, 5, size=n_samples)

    # "Скрытая" формула вероятности покупки
    logit = (
        -2.0
        + 0.015 * transactions_last_30d
        + 0.00003 * amount_last_30d
        - 0.01 * days_since_last_purchase
        + 0.8 * is_active_app_user
        - 0.2 * support_tickets_last_90d
        + 0.5 * discount_rate_mean
        + 0.005 * tenure_months
    )

    probs = 1 / (1 + np.exp(-logit))
    target = rng.binomial(1, probs)

    df = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "tenure_months": tenure_months,
            "transactions_last_30d": transactions_last_30d,
            "amount_last_30d": amount_last_30d,
            "days_since_last_purchase": days_since_last_purchase,
            "is_active_app_user": is_active_app_user,
            "support_tickets_last_90d": support_tickets_last_90d,
            "discount_rate_mean": discount_rate_mean,
            "segment_code": segment_code,
            TARGET_COLUMN: target,
        }
    )

    return df


def main():
    base_dir = Path(__file__).resolve().parents[2]
    raw_dir = base_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_data()
    output_path = raw_dir / "customers_transactions.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved synthetic data to {output_path} with shape {df.shape}")


if __name__ == "__main__":
    main()
