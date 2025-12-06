# src/training/train_model.py

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


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


def build_model(input_dim: int) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    base_dir = Path(__file__).resolve().parents[2]
    data_path = base_dir / "data" / "raw" / "customers_transactions.csv"
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    print(f"Loaded data: {df.shape}")

    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Нормализация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # PCA (уменьшение размерности, оставляем 95% дисперсии)
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)

    input_dim = X_train_pca.shape[1]
    print(f"Input dimension after PCA: {input_dim}")

    model = build_model(input_dim)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train_pca,
        y_train,
        validation_data=(X_val_pca, y_val),
        epochs=50,
        batch_size=256,
        callbacks=callbacks,
        verbose=1,
    )

    # Оценка
    y_proba = model.predict(X_val_pca).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    print("\nClassification report:")
    print(classification_report(y_val, y_pred))

    roc_auc = roc_auc_score(y_val, y_proba)
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Сохранение артефактов
    joblib.dump(scaler, models_dir / "scaler.joblib")
    joblib.dump(pca, models_dir / "pca.joblib")
    model.save(models_dir / "tf_model.keras")

    # Сохраним список фич на будущее
    import json

    with open(models_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLUMNS, f, ensure_ascii=False, indent=2)

    print(f"Artifacts saved to {models_dir}")


if __name__ == "__main__":
    main()
