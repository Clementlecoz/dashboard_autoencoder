import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

st.title(" Anomalies detected by AutoEncoder")

st.markdown("""
Welcome to the **Anomaly Detection Dashboard**.

This tool helps you explore **anomalies detected by an AutoEncoder model** for several major banks.""")
with st.expander("How to use this page"):
    st.markdown("""
                
    - Select a bank and an indicator :      
            
        Profitability (Earnings Strength)

        Liquidity (Cash Cushion)

        Solvency (Long-Term Strength)

        Leverage (Debt Load).


    - Visualize anomalies and related events over time :
                
    🔴Red dots = potential negative anomalies  

    🟢Green dots = potential positive anomalies
                
    (--🟣)Purple lines = significant news/events related to the bank

    - Optionally : Overlay macroeconomic indicators for additional context.  
    These indicators (e.g., GDP growth, inflation and interest rate ) reflect the economic situation of the countries where the bank operates.  
    Comparing them with anomalies can help explain some abnormal behaviors — for example, a drop in GDP growth might justify weaker profitability or liquidity.

    
    """)

st.markdown("""

Use the table below the chart to explore individual anomaly details and related events.

        """)




BANK_FILES = {
    "JP Morgan Chase": "anomaly_results_jpm.csv",
    "Banco Santander": "anomaly_results_banco.csv",
    "BNP Paribas": "anomaly_results_bnp_paribas.csv",
    "Crédit Agricole": "anomaly_results_crédit_agricole.csv",
    "HSBC": "anomaly_results_hsbc.csv"
}



BANK_EVENTS = {
    # JP Morgan Chase
    "JP Morgan Chase": pd.DataFrame([
        {"date":"2012-07-01", "event":"London Whale", "target":"profitability"},
        {"date":"2013-07-01", "event":"$13B DOJ subprime", "target":"profitability/solvency/leverage/liquidity"},
        {"date":"2019-09-17", "event":"Tensions repo US", "target":"liquidity"},
        {"date":"2020-03-31", "event":"COVID-19", "target":"profitability/liquidity/growth"},
        {"date":"2021-01-15", "event":"Record $12.1B Q4 profit", "target":"profitability"},
        {"date":"2023-03-15", "event":"SVB crisis", "target":"solvency/liquidity"},
        {"date":"2025-01-14", "event":"Whistleblower", "target":"solvency/profitability/leverage"},
        {"date":"2025-06-21", "event":"Directive FDIC", "target":"liquidity"},
    ]),
    
    # Banco Santander
    "Banco Santander": pd.DataFrame([
        {"date":"2008-10-01", "event":"Acquisition of Sovereign Bank", "target":"growth"},
        {"date":"2012-06-01", "event":"Spain banking crisis bailout", "target":"solvency/liquidity"},
        {"date":"2015-01-08", "event":"€7.5B capital increase", "target":"solvency/leverage"},
        {"date":"2017-06-07", "event":"Acquisition of Banco Popular", "target":"growth/solvency"},
        {"date":"2020-03-31", "event":"COVID-19 provisions", "target":"profitability/solvency"},
        {"date":"2023-03-15", "event":"SVB crisis exposure concerns", "target":"liquidity"},
    ]),
    
    # BNP Paribas
    "BNP Paribas": pd.DataFrame([
        {"date":"2008-10-01", "event":"Financial crisis impact", "target":"liquidity/solvency"},
        {"date":"2014-06-30", "event":"$8.9B US sanctions fine", "target":"profitability/solvency"},
        {"date":"2016-07-01", "event":"Restructuring of Corporate & Institutional Banking", "target":"profitability"},
        {"date":"2020-03-31", "event":"COVID-19 provisions", "target":"profitability/liquidity"},
        {"date":"2022-01-01", "event":"Sale of US retail bank", "target":"liquidity/profitability"},
        {"date":"2023-03-15", "event":"SVB crisis market volatility", "target":"liquidity"},
    ]),
    
    # Crédit Agricole
    "Crédit Agricole": pd.DataFrame([
        {"date":"2008-10-01", "event":"Global financial crisis", "target":"solvency/liquidity"},
        {"date":"2011-11-01", "event":"Greek debt exposure losses", "target":"solvency/profitability"},
        {"date":"2016-02-01", "event":"Exit from Emporiki Bank Greece", "target":"growth/solvency"},
        {"date":"2020-03-31", "event":"COVID-19 provisions", "target":"profitability/liquidity"},
        {"date":"2021-06-01", "event":"Acquisition of Creval in Italy", "target":"growth"},
        {"date":"2023-03-15", "event":"SVB crisis impact", "target":"liquidity"},
    ]),
    
    # HSBC
    "HSBC": pd.DataFrame([
        {"date":"2008-03-01", "event":"Subprime crisis losses", "target":"profitability/solvency"},
        {"date":"2012-12-11", "event":"$1.9B US money laundering fine", "target":"profitability/solvency"},
        {"date":"2015-06-01", "event":"Global restructuring announced", "target":"profitability/leverage"},
        {"date":"2020-03-31", "event":"COVID-19 provisions", "target":"profitability/liquidity"},
        {"date":"2021-02-23", "event":"Pivot to Asia strategy", "target":"growth/solvency"},
        {"date":"2023-03-15", "event":"SVB crisis UK acquisition", "target":"liquidity/growth"},
    ])
}

scores = ["profitability", "liquidity", "solvency", "leverage"]


bank_name = st.selectbox("Select a bank :", list(BANK_FILES.keys()))


@st.cache_data
def load_anomaly_data(bank_file):
    file_path = os.path.join("data", bank_file)
    if not os.path.exists(file_path):
        st.error(f" folder '{bank_file}' not available")
        return None
    df = pd.read_csv(file_path, parse_dates=["date"])
    return df

df = load_anomaly_data(BANK_FILES[bank_name])
if df is None:
    st.stop()



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


scores = ["profitability", "liquidity", "solvency", "leverage"]
for score in scores:
    df = classify_anomalies(df, score, f"delta_{score}")



selected_score = st.selectbox(" Select an indicator :", scores)


@st.cache_data
def load_macro_data():
    file_path = os.path.join( "dataset1_complet.csv")
    if not os.path.exists(file_path):
        st.error(" Fichier macroéconomique introuvable.")
        return None
    df_macro = pd.read_csv(file_path, parse_dates=["date"])
    return df_macro

macro_df = load_macro_data()
if macro_df is not None:
    
    macro_bank_df = macro_df[macro_df["company"].str.contains(bank_name, case=False, na=False)].copy()
    
    if not macro_bank_df.empty:
        allowed_macro_vars = ["inflation_YoY", "gdp_growth_rate", "interest_rate"]

        available_macro_vars = [var for var in allowed_macro_vars if var in macro_bank_df.columns]

        if available_macro_vars:
            selected_macro_var = st.selectbox(
                "Select macroeconomic variable :",
                options=["Nothing"] + available_macro_vars,
                index=0
            )
    else:
        st.info("Aucune donnée macroéconomique trouvée pour cette banque.")
        selected_macro_var = "Nothing"
else:
    selected_macro_var = "Nothing"


def plot_anomalies(df, score_name, company, macro_df=None, macro_var=None):
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

    # Plot macroeconomic variable if selected
    if macro_df is not None and macro_var in macro_df.columns:
        macro_df.sort_values("date", inplace=True)
        ax2 = ax1.twinx()
        ax2.plot(
            macro_df["date"],
            macro_df[macro_var],
            color="orange",
            linestyle="dashed",
            label=f"Macro: {macro_var}"
        )
        ax2.set_ylabel("Macro Variable")

    
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

    # Combine legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    if macro_df is not None and macro_var in macro_df.columns:
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2)
    else:
        ax1.legend()

    ax1.grid(True)
    st.pyplot(fig)




plot_anomalies(df, selected_score, bank_name, macro_bank_df if selected_macro_var != "Aucune" else None, selected_macro_var)


# === Table of anomalies with event context ===
st.subheader("Anomaly Table with Events")

# Filter anomalies of selected score
anomaly_col = f"is_anomaly_{selected_score}"
delta_col = f"delta_{selected_score}"
nature_col = f"anomaly_nature_{selected_score}"

# Create filtered anomaly DataFrame
df_anomalies = df[
    (df[anomaly_col]) & 
    (df["company"].str.contains(bank_name, case=False, na=False))
].copy()

# Add anomaly type filter
selected_nature = st.selectbox("Filter anomaly type:", ["all", "good", "bad"])
if selected_nature != "all":
    df_anomalies = df_anomalies[df_anomalies[nature_col] == selected_nature]

# Add nearby events (±1 quarter = ~90 days)
events_df = BANK_EVENTS.get(bank_name)
if events_df is not None:
    events_df["date"] = pd.to_datetime(events_df["date"])
    events_df_score = events_df[events_df["target"].str.contains(selected_score.lower(), na=False)]

    from datetime import timedelta

    def find_nearby_events(anomaly_date, events_df, tolerance=timedelta(days=90)):
        nearby_events = events_df[
            (events_df["date"] >= anomaly_date - tolerance) & 
            (events_df["date"] <= anomaly_date + tolerance)
        ]
        return ", ".join(nearby_events["event"]) if not nearby_events.empty else ""

    df_anomalies["event"] = df_anomalies["date"].apply(lambda d: find_nearby_events(d, events_df_score))
else:
    df_anomalies["event"] = ""

# Simplify column names for display
columns_to_display = ["date", f"reconstruction_error_{selected_score}", delta_col, nature_col, "event"]
df_anomalies_display = df_anomalies[columns_to_display].rename(columns={
    "date": "Date",
    f"reconstruction_error_{selected_score}": "Error",
    delta_col: "Δ Score",
    nature_col: "Type",
    "event": "Nearby Event"
})

# Format the date (YYYY-MM-DD only, remove hour)
df_anomalies_display["Date"] = pd.to_datetime(df_anomalies_display["Date"]).dt.strftime("%Y-%m-%d")

# Display the simplified table
st.dataframe(df_anomalies_display.sort_values("Date"), use_container_width=True)
