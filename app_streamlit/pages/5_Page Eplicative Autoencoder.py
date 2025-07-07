import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Understanding AutoEncoders", layout="wide")

st.title("🧠 Understanding Anomaly Detection with AutoEncoders")

# === 1. What is an AutoEncoder? ===
st.header("1. What is an AutoEncoder?")
st.markdown("""
An **AutoEncoder** is a neural network trained to **reconstruct its own input**.

It has two parts:
- 🔽 **Encoder**: compresses financial data into a simplified form.
- 🔼 **Decoder**: attempts to rebuild the original data from the compressed form.

➡️ When the data is **normal**, the AutoEncoder reconstructs it well.  
➡️ When the data is **abnormal**, the reconstruction fails → high **reconstruction error**, which is a signal of an **anomaly**.
""")

st.image(
    "https://upload.wikimedia.org/wikipedia/commons/6/60/Autoencoder_structure.png",
    caption="Simplified AutoEncoder structure",
    use_container_width=True
)

# === 2. How are anomalies detected? ===
st.header("2. How are anomalies detected?")
st.markdown(r"""
After training the model on a **healthy period**, we evaluate its performance across the entire timeline.

We compute the **reconstruction error** for each time point:

\[
\text{Error} = \lVert \text{Input} - \text{Reconstructed Output} \rVert^2
\]

An anomaly threshold is defined automatically:
- 📉 It corresponds to the **95th percentile** of the error **on the healthy period**
- ➕ This means only the 5% worst reconstructions (even in healthy times) trigger an alert
""")

# === 3. How is the healthy period selected? ===
st.header("3. What is considered a healthy financial period?")
st.markdown("""
Before training, we **automatically select a stable period** based on the score you want to monitor.

✅ **Selection criteria** for each indicator:
- The score is **not too high or too low** → between the **10th and 90th percentiles**
- The **variation** of the score is also moderate → between the **10th and 90th percentiles**

👉 This helps exclude known crises, shocks, or surprising performance from the training set.
""")

# === 4. How to read the anomaly charts ===
st.header("4. How to read the anomaly charts?")
st.markdown("""
- **Y-axis**: reconstruction error
- **Red points**: negative anomalies (unexpected drop or loss)
- **Green points**: positive anomalies (unexpected improvement)
- **Blue line**: automatic anomaly threshold
- **Shaded areas**: clusters of nearby anomalies
- **Purple lines**: known macroeconomic or business events

👉 Sudden spikes often reflect important external events (COVID, SVB crisis, etc.)
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
ax.text(3, 6.5, "⚠️ Cluster", ha="center", color="darkred")
ax.legend()
st.pyplot(fig)

# === 5. Interpretation Tips ===
st.header("5. How to interpret anomalies?")
st.markdown("""
Here are some useful clues:
- 📈 Gradually increasing error → emerging risk or latent instability
- 🔺 Sudden spike → external shock or event
- 🟩 Multiple positive anomalies → underestimated potential or good surprise
- ❓ Anomaly without any known event → worth investigating

**Clusters** of anomalies often indicate structural stress or opportunity periods.
""")

# === 6. Final note ===
st.success("🎯 This page helps you understand anomaly detection with AutoEncoders, even without technical background.")
