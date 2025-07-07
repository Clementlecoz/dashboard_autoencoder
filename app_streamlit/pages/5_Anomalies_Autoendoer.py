import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

st.title("🧠 Anomalies détectées par AutoEncoder")

# === 🔁 Fichiers des anomalies disponibles ===
BANK_FILES = {
    "JP Morgan Chase": "anomaly_results_jpm.csv",
    "Banco Santander": "anomaly_results_banco.csv",
    "BNP Paribas": "anomaly_results_bnp_paribas.csv",
    "Crédit Agricole": "anomaly_results_crédit_agricole.csv",
    "HSBC": "anomaly_results_hsbc.csv"
}

# === 📅 Événements spécifiques par banque ===
BANK_EVENTS = {
    "JP Morgan Chase": pd.DataFrame([
        {"date":"2012-07-01", "event":"London Whale (loss ~$6B)", "target":"profitability"},
        {"date":"2013-12-01", "event":"$13B DOJ subprime", "target":"profitability/solvency/leverage/liquidity"},
        {"date":"2019-09-17", "event":"Tensions repo US", "target":"liquidity"},
        {"date":"2020-03-31", "event":"COVID-19", "target":"profitability/liquidity/growth"},
        {"date":"2021-01-15", "event":"Record $12.1B Q4 profit", "target":"profitability"},
        {"date":"2023-03-15", "event":"SVB crisis", "target":"solvency/liquidity"},
        {"date":"2025-01-14", "event":"Whistleblower", "target":"solvency/profitability/leverage"},
        {"date":"2025-06-21", "event":"Directive FDIC", "target":"liquidity"},
    ])
    # Ajoute ici les autres banques si tu veux afficher leurs événements également
}

# === ⬇️ Sélection banque
bank_name = st.selectbox("Choisir une banque :", list(BANK_FILES.keys()))

# === Chargement des données anomalies
@st.cache_data
def load_anomaly_data(bank_file):
    file_path = os.path.join("data", bank_file)
    if not os.path.exists(file_path):
        st.error(f"⚠️ Le fichier '{bank_file}' est introuvable.")
        return None
    df = pd.read_csv(file_path, parse_dates=["date"])
    return df

df = load_anomaly_data(BANK_FILES[bank_name])
if df is None:
    st.stop()

# === Fonction de classification good/bad
def classify_anomalies(df, score_name, delta_col):
    error_col = f"reconstruction_error_{score_name}"
    anomaly_col = f"is_anomaly_{score_name}"
    nature_col = f"anomaly_nature_{score_name}"
    if delta_col not in df.columns or anomaly_col not in df.columns:
        return df
    df[nature_col] = "none"
    df.loc[df[anomaly_col] & (df[delta_col] > 0), nature_col] = "bad"
    df.loc[df[anomaly_col] & (df[delta_col] < 0), nature_col] = "good"
    return df

# === Classification des anomalies pour chaque indicateur
scores = ["profitability", "liquidity", "solvency", "leverage"]
for score in scores:
    df = classify_anomalies(df, score, f"delta_{score}")

# === Sélection de l'indicateur
selected_score = st.selectbox("Choisir un indicateur :", scores)

# === Affichage graphique
def plot_anomalies(df, score_name, company):
    err_col = f"reconstruction_error_{score_name}"
    anomaly_nature_col = f"anomaly_nature_{score_name}"
    threshold_col = f"threshold_{score_name}"

    df_c = df[df["company"].str.contains(company, case=False, na=False)].copy()
    df_c.sort_values("date", inplace=True)

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(df_c["date"], df_c[err_col], label="Reconstruction Error", color="gray")

    if threshold_col in df_c.columns:
        ax1.axhline(df_c[threshold_col].iloc[0], color="blue", linestyle="--", label="Threshold")

    if anomaly_nature_col in df_c.columns:
        ax1.scatter(df_c["date"][df_c[anomaly_nature_col] == "good"],
                    df_c[err_col][df_c[anomaly_nature_col] == "good"],
                    color="green", label="Good", zorder=5)
        ax1.scatter(df_c["date"][df_c[anomaly_nature_col] == "bad"],
                    df_c[err_col][df_c[anomaly_nature_col] == "bad"],
                    color="red", label="Bad", zorder=5)

    # Ajout des événements s'ils existent
    events_df = BANK_EVENTS.get(company)
    if events_df is not None:
        events_df = events_df[pd.to_datetime(events_df["date"]) >= df_c["date"].min()]
        events_df["date"] = pd.to_datetime(events_df["date"])
        for _, row in events_df[events_df["target"].str.contains(score_name.lower())].iterrows():
            ax1.axvline(row["date"], color='purple', linestyle=':', alpha=0.7)
            ax1.text(row["date"], ax1.get_ylim()[1]*0.95, row["event"],
                     rotation=90, fontsize=8, color='purple', verticalalignment='top')

    ax1.set_title(f"Anomalies AutoEncoder - {score_name.capitalize()} ({company})")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Reconstruction Error")
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    ax1.grid(True)
    ax1.legend()
    st.pyplot(fig)

# === Affichage du graphique final
plot_anomalies(df, selected_score, bank_name)
