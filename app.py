import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
import pandas as pd
import plotly.express as px
import os



st.set_page_config(
    page_title="Diabetes Surveillance System",
    layout="wide"
)

st.markdown("""
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
""", unsafe_allow_html=True)

st.title("Diabetes Risk Surveillance System")



tab1, tab2 = st.tabs(["Individual Risk Prediction", "Population Risk Map"])



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



@st.cache_resource
def load_model():
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, "diabetes_dnn_a.pt")

    model = DiabetesDNN(INPUT_DIM)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


@st.cache_resource
def load_scaler():
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, "scaler.pkl")
    return joblib.load(path)


model = load_model()
scaler = load_scaler()



with tab1:

    st.subheader("Individual Risk Estimation")

    col1, col2 = st.columns(2)

    with col1:
        asthma = st.selectbox(
            "Asthma Diagnosis",
            ["No", "Yes"],
            help="Has a doctor ever diagnosed asthma?"
        )
        asthma = 1 if asthma == "Yes" else 0

        kidney = st.selectbox(
            "Kidney Disease",
            ["No", "Yes"],
            help="Chronic kidney disease diagnosis"
        )
        kidney = 1 if kidney == "Yes" else 0

        arthritis = st.selectbox(
            "Arthritis",
            ["No", "Yes"],
            help="Includes arthritis, gout, lupus, or fibromyalgia"
        )
        arthritis = 1 if arthritis == "Yes" else 0

        education = st.selectbox(
            "Education Level",
            [
                "1 - No schooling",
                "2 - Elementary",
                "3 - Some high school",
                "4 - High school graduate",
                "5 - Some college",
                "6 - College graduate"
            ]
        )
        education = int(education.split(" - ")[0])

        income = st.selectbox(
            "Income Level",
            [
                "1 - Very low",
                "2 - Low",
                "3 - Middle",
                "4 - High",
                "5 - Very high"
            ]
        )
        income = int(income.split(" - ")[0])

    with col2:
        age = st.selectbox(
            "Age Group",
            [
                "1 - 18–24",
                "2 - 25–34",
                "3 - 35–44",
                "4 - 45–54",
                "5 - 55–64",
                "6 - 65+"
            ]
        )
        age = int(age.split(" - ")[0])

        sex_label = st.selectbox("Sex", ["Female", "Male"])
        sex = 0 if sex_label == "Female" else 1

        bmi = st.slider(
            "Body Mass Index (BMI)",
            10.0, 60.0, 27.0,
            help="BMI = weight (kg) / height (m²)"
        )

        smoking = st.selectbox(
            "Smoking Status",
            ["Non-smoker", "Smoker"],
            help="Has smoked at least 100 cigarettes in lifetime"
        )
        smoking = 1 if smoking == "Smoker" else 0

        physical_activity = st.selectbox(
            "Physical Activity",
            ["Active", "Inactive"],
            help="Any physical activity in the past 30 days"
        )
        physical_activity = 1 if physical_activity == "Active" else 0


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
            "Risk Category",
            "High" if prob > 0.7 else "Moderate" if prob > 0.3 else "Low"
        )
        colC.metric("Model Threshold", "0.50")

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
