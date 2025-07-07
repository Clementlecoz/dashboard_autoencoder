import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os


from datetime import timedelta

def detect_anomaly_clusters_by_proximity_fixed(dates, min_anomalies=3, max_total_days=185):
    if not dates:
        return []

    dates = sorted(dates)
    clusters = []
    i = 0
    while i < len(dates):
        cluster_found = False
        for j in range(i + min_anomalies - 1, len(dates)):
            duration = (dates[j] - dates[i]).days
            if duration <= max_total_days:
                if j - i + 1 >= min_anomalies:
                    clusters.append((dates[i] - timedelta(days=5), dates[j] + timedelta(days=5)))
                    i = j + 1  # saut au prochain groupe possible
                    cluster_found = True
                    break
            else:
                break
        if not cluster_found:
            i += 1
    return clusters





st.title("🧠 Anomalies détectées par AutoEncoder")

# === 🔁 Fichiers des anomalies disponibles ===
BANK_FILES = {
    "JP Morgan Chase": "anomaly_results_jpm.csv",
    "Banco Santander": "anomaly_results_banco.csv",
    "BNP Paribas": "anomaly_results_bnp.csv",
    "Crédit Agricole": "anomaly_results_ca.csv",
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
    ]),
    # Ajoute ici les autres banques si tu veux afficher leurs événements également

    "HSBC" : pd.DataFrame([
    {"date": "2012-12-11", "event": "Money laundering fine ($1.9B)", "target": "solvency/profitability/leverage"},
    {"date": "2015-02-09", "event": "Swiss Leaks scandal (tax evasion)", "target": "solvency/profitability"},
    {"date": "2016-06-24", "event": "Brexit vote", "target": "liquidity/growth"},
    {"date": "2020-03-31", "event": "COVID-19", "target": "profitability/liquidity/growth"},
    {"date": "2021-02-23", "event": "Pivot to Asia strategy & branch closures", "target": "growth/profitability"},
    {"date": "2022-10-28", "event": "Q3 profit drop due to inflation & UK market turmoil", "target": "profitability/liquidity"},
    {"date": "2023-03-13", "event": "Acquires SVB UK during crisis", "target": "liquidity/growth"},
    {"date": "2023-11-01", "event": "Major layoffs plan (2,000 jobs)", "target": "profitability"},
    {"date": "2024-04-30", "event": "Exit from Canadian operations completed", "target": "growth/profitability"},
    {"date": "2025-05-15", "event": "New climate risk regulation impact", "target": "solvency/leverage/growth"},
]),

    "Crédit Agricole" : pd.DataFrame([
    {"date": "2012-10-01", "event": "Vente d’Emporiki Bank (Grèce)", "target": "solvency/profitability"},
    {"date": "2014-02-14", "event": "Résultats annuels : retour aux bénéfices", "target": "profitability/growth"},
    {"date": "2016-08-01", "event": "Réorganisation interne (simplification structure cotée)", "target": "leverage/profitability"},
    {"date": "2020-03-31", "event": "COVID-19", "target": "profitability/liquidity/growth"},
    {"date": "2021-05-06", "event": "Résultats record T1 2021", "target": "profitability"},
    {"date": "2022-07-28", "event": "Impact guerre Ukraine & inflation", "target": "solvency/liquidity"},
    {"date": "2023-03-15", "event": "Tensions bancaires post-SVB", "target": "solvency/liquidity"},
    {"date": "2024-01-18", "event": "Acquisition d'Olinn (leasing & IT)", "target": "growth/leverage"},
    {"date": "2024-10-01", "event": "Investissements IA et numérique", "target": "growth"},
    {"date": "2025-04-20", "event": "Directive européenne sur fonds propres", "target": "solvency/leverage"},
]),

    "BNP Paribas" : pd.DataFrame([
    {"date": "2014-07-01", "event": "Amende record $8.9B pour violation d'embargos US", "target": "solvency/profitability"},
    {"date": "2015-12-01", "event": "Réduction d'exposition aux matières premières", "target": "leverage/liquidity"},
    {"date": "2017-10-05", "event": "Plan stratégique 2020 (transformation digitale)", "target": "growth/profitability"},
    {"date": "2020-03-31", "event": "COVID-19", "target": "profitability/liquidity/growth"},
    {"date": "2021-05-18", "event": "Résultats solides post-pandémie", "target": "profitability"},
    {"date": "2022-01-01", "event": "Vente Bank of the West (États-Unis)", "target": "liquidity/solvency"},
    {"date": "2023-03-15", "event": "Turbulences bancaires (SVB, Crédit Suisse)", "target": "solvency/liquidity"},
    {"date": "2023-12-12", "event": "Plan 2025 : accélération investissements IA et ESG", "target": "growth/leverage"},
    {"date": "2024-05-06", "event": "Acquisition Floa Bank (BNP PF)", "target": "growth/profitability"},
    {"date": "2025-03-25", "event": "Stress tests BCE : exigences renforcées", "target": "solvency/leverage"},
]), 

    "Banco Santander" : pd.DataFrame([
    {"date": "2012-11-01", "event": "Résultats affectés par crise zone euro", "target": "profitability/solvency"},
    {"date": "2015-01-08", "event": "Augmentation de capital de €7.5B", "target": "solvency/leverage"},
    {"date": "2017-06-07", "event": "Acquisition Banco Popular pour €1", "target": "growth/solvency/liquidity"},
    {"date": "2019-07-30", "event": "Résultats affectés par Brexit & taux négatifs", "target": "profitability"},
    {"date": "2019-10-31", "event": "COVID-19", "target": "profitability/liquidity/growth"},
    {"date": "2021-02-03", "event": "Perte annuelle historique 2020", "target": "profitability"},
    {"date": "2022-02-02", "event": "Reprise solide : bénéfice net record", "target": "profitability"},
    {"date": "2023-03-15", "event": "Tensions bancaires globales post-SVB", "target": "solvency/liquidity"},
    {"date": "2024-09-10", "event": "Investissements dans la fintech Ebury", "target": "growth/leverage"},
    {"date": "2025-04-02", "event": "Directive BCE sur fonds propres minimums", "target": "solvency/leverage"},
])

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


    # Regroupement visuel par type d’anomalie : 3+ dans la même année
    if anomaly_nature_col in df_c.columns:
        bad_anomalies = df_c[df_c[anomaly_nature_col] == "bad"]
        bad_anomalies_dates = list(bad_anomalies["date"])
        # DEBUG : Affiche les dates bad pour vérification
        # Affichage debug
        
        clusters_bad = detect_anomaly_clusters_by_proximity_fixed(bad_anomalies_dates, min_anomalies=3, max_total_days=185)
        
        for start, end in clusters_bad:
            ax1.axvspan(start, end, color='red', alpha=0.1, zorder=2.5)
            ax1.text(start + (end - start) / 2,
                    5.5,  # ↩️ Alt temporaire au lieu de get_ylim()[1] * 0.85
                    "⚠️ Cluster",
                    ha='center', fontsize=9, color='darkred', alpha=0.6)

        good_anomalies = df_c[df_c[anomaly_nature_col] == "good"]
        good_anomalies_dates = list(good_anomalies["date"])
        clusters_good = detect_anomaly_clusters_by_proximity_fixed(good_anomalies_dates, min_anomalies=3)

        for start, end in clusters_good:
            ax1.axvspan(start, end, color='green', alpha=0.1, zorder=0)
            ax1.text(start + (end - start) / 2,
                    10,
                    "✔️ Cluster",
                    ha='center', fontsize=9, color='darkgreen', alpha=0.6)

    st.pyplot(fig)

# === Affichage du graphique final
plot_anomalies(df, selected_score, bank_name)

st.markdown("---")
st.subheader("📋 Synthèse des anomalies détectées")

# 1. Extraction des anomalies sélectionnées
anomaly_col = f"is_anomaly_{selected_score}"
nature_col = f"anomaly_nature_{selected_score}"
delta_col = f"delta_{selected_score}"
err_col = f"reconstruction_error_{selected_score}"

df_anomalies = df[df[anomaly_col]].copy()
df_anomalies = df_anomalies[df_anomalies["company"].str.contains(bank_name, case=False, na=False)]

# 2. Ajout des colonnes essentielles
df_display = pd.DataFrame({
    "Date": df_anomalies["date"],
    "Erreur de reconstruction": df_anomalies[err_col],
    "Variation de score": df_anomalies[delta_col],
    "Type d'anomalie": df_anomalies[nature_col]
})

# 3. Fonction pour associer un événement proche
def match_event(date, events_df):
    if events_df is None:
        return ""
    events_df["date"] = pd.to_datetime(events_df["date"])
    matched = events_df[(events_df["date"] >= date - pd.Timedelta(days=30)) &
                        (events_df["date"] <= date + pd.Timedelta(days=30))]
    if not matched.empty:
        return " / ".join(matched["event"].tolist())
    return ""

# 4. Application à la banque courante
if bank_name in BANK_EVENTS:
    df_display["Événement associé"] = df_display["Date"].apply(lambda d: match_event(d, BANK_EVENTS[bank_name]))
else:
    df_display["Événement associé"] = ""

# 5. Interprétation automatique
def interpret_anomaly(row):
    if row["Type d'anomalie"] == "bad" and row["Variation de score"] > 0:
        return "⚠️ Dégradation brutale"
    elif row["Type d'anomalie"] == "good" and row["Variation de score"] < 0:
        return "✔️ Amélioration soudaine"
    elif abs(row["Variation de score"]) < 0.01:
        return "❓ Stable (possible biais)"
    else:
        return "🔍 Changement atypique"

df_display["Interprétation"] = df_display.apply(interpret_anomaly, axis=1)

# 6. Filtre par type d’anomalie
nature_filter = st.radio("Filtrer les anomalies :", ["Toutes", "Bad", "Good"], horizontal=True)
if nature_filter == "Bad":
    df_display = df_display[df_display["Type d'anomalie"] == "bad"]
elif nature_filter == "Good":
    df_display = df_display[df_display["Type d'anomalie"] == "good"]

# 7. Affichage final
st.dataframe(
    df_display.sort_values("Date"),
    use_container_width=True
)
