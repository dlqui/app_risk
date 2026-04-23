
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
import pandas as pd
import plotly.express as px


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Diabetes Surveillance System",
    layout="wide"
)

st.markdown(
    """
    <style>
        body { background-color: #0E1117; color: #EAEAEA; }
        .stApp { background-color: #0E1117; }
        h1, h2, h3 { color: #FFFFFF; font-family: Arial; }
        .stMetric {
            background-color: #161B22;
            border-radius: 10px;
            padding: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Diabetes Risk Surveillance System")

tab1, tab2 = st.tabs(["Individual Risk Prediction", "Population Risk Map"])


# =========================
# MODEL DEFINITION
# =========================
class DiabetesDNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()


INPUT_DIM = 10


# =========================
# LOAD MODEL + SCALER (CACHED)
# =========================
@st.cache_resource
def load_model():
    model = DiabetesDNN(INPUT_DIM)
    model.load_state_dict(
        torch.load("diabetes_dnn_a.pt", map_location=torch.device("cpu"))
    )
    model.eval()
    return model


@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")


model = load_model()
scaler = load_scaler()


# ==========================================================
# TAB 1: INDIVIDUAL PREDICTION
# ==========================================================
with tab1:

    st.subheader("Individual Risk Estimation")

    col1, col2 = st.columns(2)

    with col1:
        asthma = st.selectbox("Asthma", [0, 1])
        kidney = st.selectbox("Kidney disease", [0, 1])
        arthritis = st.selectbox("Arthritis", [0, 1])
        education = st.slider("Education level", 1, 6, 3)
        income = st.slider("Income level", 1, 5, 3)

    with col2:
        age = st.slider("Age group", 1, 6, 3)
        sex = st.selectbox("Sex", [0, 1])
        bmi = st.slider("BMI", 10.0, 60.0, 27.0)
        smoking = st.selectbox("Smoking", [0, 1])
        physical_activity = st.selectbox("Physical activity", [0, 1])

    x = np.array([[
        asthma, kidney, arthritis,
        education, income,
        age, sex, bmi,
        smoking, physical_activity
    ]])

    x_scaled = scaler.transform(x)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    if st.button("Run Prediction"):

        with torch.no_grad():
            prob = torch.sigmoid(model(x_tensor)).item()

        colA, colB, colC = st.columns(3)

        colA.metric("Risk Score", f"{prob:.3f}")
        colB.metric(
            "Classification",
            "High" if prob > 0.7 else "Medium" if prob > 0.3 else "Low"
        )
        colC.metric("Decision Threshold", "0.50")

        fig = px.bar(
            x=["No Risk", "Risk"],
            y=[1 - prob, prob],
            color=["No Risk", "Risk"],
            color_discrete_map={
                "No Risk": "#2ecc71",
                "Risk": "#e74c3c"
            }
        )

        fig.update_layout(
            height=350,
            yaxis_range=[0, 1],
            showlegend=False,
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117"
        )

        st.plotly_chart(fig, use_container_width=True)


# ==========================================================
# TAB 2: POPULATION MAP
# ==========================================================
with tab2:

    st.subheader("State-Level Risk Simulation")

    states = [
        "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
        "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
        "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
        "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
        "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"
    ]

    np.random.seed(42)

    def simulate():
        return np.array([
            np.random.binomial(1, 0.2),
            np.random.binomial(1, 0.1),
            np.random.binomial(1, 0.25),
            np.random.randint(1, 6),
            np.random.randint(1, 5),
            np.random.randint(1, 6),
            np.random.binomial(1, 0.5),
            np.random.normal(28, 4),
            np.random.binomial(1, 0.3),
            np.random.binomial(1, 0.4)
        ])

    results = []

    for s in states:
        x = simulate().reshape(1, -1)
        x_scaled = scaler.transform(x)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

        with torch.no_grad():
            prob = torch.sigmoid(model(x_tensor)).item()

        results.append([s, prob])

    df = pd.DataFrame(results, columns=["state", "risk"])

    fig = px.choropleth(
        df,
        locations="state",
        locationmode="USA-states",
        color="risk",
        scope="usa",
        range_color=(0, 1),
        color_continuous_scale=[
            [0.0, "#2ecc71"],
            [0.5, "#f1c40f"],
            [1.0, "#e74c3c"]
        ],
        title="Simulated Diabetes Risk by State"
    )

    fig.update_layout(
        height=650,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df.sort_values("risk", ascending=False))