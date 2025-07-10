from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
import numpy as np
import pandas as pd
from functools import reduce


df = pd.read_csv("dataset1_complet.csv")
df_jpm = df[df["company"] == "JP Morgan Chase"].copy()
df_jpm.head()


df_jpm["delta_profitability"] = df_jpm["score_profitability_local"].diff()
df_jpm["delta_liquidity"] = df_jpm["score_liquidity_local"].diff()
df_jpm["delta_solvency"] = df_jpm["score_solvency_local"].diff()
df_jpm["delta_leverage"] = df_jpm["score_leverage_adjusted_local"].diff()

df_jpm.fillna(0, inplace=True)



def train_autoencoder(X_train, encoding_dim=2, epochs=100, batch_size=8, X_val=None):
    input_dim = X_train.shape[1]
    input_layer = Input(shape=(input_dim,))

    encoded = Dense(encoding_dim * 2, activation='relu')(input_layer)
    encoded = Dense(encoding_dim, activation='relu')(encoded)

    decoded = Dense(encoding_dim * 2, activation='relu')(encoded)
    output_layer = Dense(input_dim, activation='linear')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    autoencoder.fit(X_train, X_train,
                    validation_data=(X_val, X_val) if X_val is not None else None,
                    epochs=epochs, batch_size=batch_size, verbose=0)

    return autoencoder

def get_healthy_periods(df, score_name, delta_name, p_low=0.10, p_high=0.90):
    healthy_mask = (
        df[score_name].between(df[score_name].quantile(p_low), df[score_name].quantile(p_high)) &
        df[delta_name].between(df[delta_name].quantile(p_low), df[delta_name].quantile(p_high))
        #df["revenue_growth"].between(-rg_threshold, rg_threshold) don't use it for the moment as threshold
    )
    return df[healthy_mask].copy()

def run_anomaly_pipeline(df_jpm):
    features_by_score = {
        "profitability": ["score_profitability_local", "delta_profitability"],
        "liquidity": ["score_liquidity_local", "delta_liquidity"],
        "solvency": ["score_solvency_local", "delta_solvency"],
        "leverage": ["score_leverage_adjusted_local", "delta_leverage"]
    }

    df_errors_all = []

    for score, (score_col, delta_col) in features_by_score.items():
        print(f"\n indicators : {score}")


        df_healthy = get_healthy_periods(df_jpm, score_col, delta_col)
        print(f" {len(df_healthy)} health period identified.")


        scaler = StandardScaler()
        X_train = scaler.fit_transform(df_healthy[[score_col, delta_col]])
        X_full = scaler.transform(df_jpm[[score_col, delta_col]])

        X_tr, X_val = train_test_split(X_train, test_size=0.2, random_state=42)

        model = train_autoencoder(X_tr, encoding_dim=2, X_val=X_val)
        model.save(f"autoencoder_model_{score}.h5", save_format='h5' )

        # rebuild for all priods
        X_pred = model.predict(X_full)
        mse = np.mean(np.square(X_full - X_pred), axis=1)

        # rebuild in health period
        X_val_pred = model.predict(X_val)
        mse_val = np.mean(np.square(X_val - X_val_pred), axis=1)

        # threshold 95 percentil of helathy period
        mse_train = np.mean(np.square(X_train - model.predict(X_train)), axis=1)
        threshold = np.percentile(mse_val, 95)
        print(f" Seuil d’anomalie (95e percentile) pour {score} : {threshold:.4f}")

        #  anomaly detection
        is_anomaly = mse > threshold
        delta = df_jpm[delta_col].values
        anomaly_type = np.where(is_anomaly & (delta > 0), "positive",
                         np.where(is_anomaly & (delta < 0), "negative", "none"))

        df_errors = pd.DataFrame({
            "date": df_jpm["date"].values,
            "company": df_jpm["company"].values,
            score_col: df_jpm[score_col].values,
            delta_col: df_jpm[delta_col].values,
            f"reconstruction_error_{score}": mse,
            f"is_anomaly_{score}": is_anomaly,
            f"anomaly_type_{score}": anomaly_type,
            f"threshold_{score}": [threshold] * len(df_jpm)
        })

        df_errors_all.append(df_errors)

    if not df_errors_all:
        print(" Aucun score n’a pu être traité, DataFrame final vide.")
        return pd.DataFrame()

    # Fusion sur 'date' et 'company'
    df_errors_merged = reduce(lambda left, right: pd.merge(left, right, on=["date", "company"]), df_errors_all)
    print("\n Fusion des erreurs terminée.")
    return df_errors_merged


df_errors_merged = run_anomaly_pipeline(df_jpm)
# Enregistre le résultat final
df_errors_merged.to_csv("../app_streamlit/data/anomaly_results_jpm.csv", index=False)
