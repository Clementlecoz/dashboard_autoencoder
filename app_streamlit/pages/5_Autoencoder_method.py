import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Understanding AutoEncoders", layout="wide")

st.title("Understanding Anomaly Detection with AutoEncoders")

# === 1. Why Use AI for Anomaly Detection? ===
st.header("1. Why Use AI for Anomaly Detection?")
st.markdown("""
In financial analysis, it’s crucial not only to spot large variations in individual indicators
but also to detect subtle patterns that might signal emerging risks.

**Traditional thresholds** (like percentile-based alerts) are excellent for finding obvious shocks or extreme values.
However, they sometimes miss more complex combinations of changes across multiple indicators.

To go further, we combine traditional methods with **Artificial Intelligence (AI)** — specifically
anomaly detection models. We use past periods of stable financial performance (called
**healthy periods**) to teach our AI models what “normal” financial behavior looks like.


""")

# === 2. What is an AutoEncoder? ===
st.header("2. What is an AutoEncoder?")
st.markdown("""
Our AI anomaly detection uses **AutoEncoders**, a type of neural network trained to
reconstruct its own input data.

The idea is simple:
- **Encoder**: compresses financial indicators into a simplified, lower-dimensional space.
- **Decoder**: tries to rebuild the original data from that compressed version.

If the data reflects normal financial conditions, the reconstruction is accurate,
and the **reconstruction error** is low.  
If the data is unusual or risky, the model fails to reconstruct it well,
and the reconstruction error becomes high — signaling a possible anomaly.

This makes AutoEncoders powerful for discovering:
- Unusual individual values
- Unexpected combinations of multiple indicators
- Sharp changes from one period to the next (thanks to delta features)

We also include **temporal deltas** in the model: for each indicator,
we compute how much it has changed compared to the previous quarter.
This helps detect abrupt shifts that may not stand out
if we look only at static values.
""")



# === 3. How Do We Define Healthy Periods? ===
st.header("3. How Do We Define Healthy Periods?")
st.markdown("""
Before we train our AutoEncoders, we need to define what “normal” financial behavior looks like.

We call these stable times **healthy periods**.

We identify healthy periods by applying two filters:
- We keep only quarters where the financial score stays **between the 10th and 90th percentiles** (as we have done with Health scoring).
- We apply the same filter to the **deltas** (the change in scores compared to the previous quarter).

By doing this, we exclude periods affected by crises, unusual events, or extreme performance.
This ensures that our AI models learn only from typical financial behavior,
making them much better at spotting what’s truly unusual later on.


""")

# === 4. How to read the anomaly charts ===
st.header("4. How to read the anomaly charts?")
st.markdown("""
            
Once our AutoEncoder is trained on healthy periods, we apply it to the **entire time series** of financial data.

For each quarter, the model tries to reconstruct the original financial scores and their deltas.
The difference between the actual values and the reconstructed values is called the **reconstruction error**.

- A **low error** means the data looks similar to healthy periods.
- A **high error** suggests the data is unusual compared to what the model learned as “normal.”

This reconstruction error becomes the key metric we plot to detect anomalies.
s

- **Y-axis**: reconstruction error
- **Red points**: negative anomalies (unexpected drop or loss)
- **Green points**: positive anomalies (unexpected improvement)
- **Blue line**: automatic anomaly threshold
- **Shaded areas**: clusters of nearby anomalies
- **Purple lines**: known macroeconomic or business events

 Sudden spikes often reflect important external events (COVID, SVB crisis, etc.)
""")

# === Example chart ===
x = np.arange(10)
y = np.random.normal(2, 0.5, size=10)
y[3] = 6
y[7] = 0.5
fig, ax = plt.subplots()
ax.plot(x, y, label="Reconstruction Error", color="gray")
ax.axhline(4, color="blue", linestyle="--", label="Threshold")
ax.scatter([3], [6], color="red", label="Negative Anomaly")
ax.scatter([7], [0.5], color="green", label="Positive Anomaly")
ax.fill_between([2, 4], 0, 7, color='red', alpha=0.1)
ax.text(3, 6.5, " Cluster", ha="center", color="darkred")
ax.legend()
st.pyplot(fig)

# === 5. Interpretation Tips ===
st.header("5. How to interpret anomalies?")
st.markdown("""
Here are some useful clues:
-  Gradually increasing error → emerging risk or latent instability
-  Sudden spike → external shock or event
-  Multiple positive anomalies → underestimated potential or good surprise
-  Anomaly without any known event → worth investigating

**Clusters** of anomalies often indicate structural stress or opportunity periods.
""")

