
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="Bike Rental Prediction",
    page_icon="🚲",
    layout="wide"
)

# ----------------------------------
# LOAD SMALLER TRAINED MODEL
# ----------------------------------

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    model = joblib.load(
        BASE_DIR / "bike_rental_model_small.pkl"
    )
    features = joblib.load(
        BASE_DIR / "feature_names.pkl"
    )
    return model, features

model, features = load_model()

# ----------------------------------
# CUSTOM UI DESIGN
# ----------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #f5f7fc;
    color: #172b4d;
}

h1, h2, h3 {
    color: #15375d;
}

[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #e0e6ef;
    padding: 20px;
    border-radius: 12px;
}

div.stFormSubmitButton > button {
    background-color: #1769aa;
    color: white;
    border-radius: 10px;
    font-weight: bold;
    border: none;
}

div.stFormSubmitButton > button:hover {
    background-color: #104b80;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------
# SIDEBAR
# ----------------------------------

with st.sidebar:

    st.title("🚲 Bike Rental AI")
    st.caption("Machine Learning Assignment 3")

    st.divider()

    st.subheader("Model Information")

    st.write("**Algorithm:** Extra Trees Regressor")
    st.write("**Validation:** 10-Fold CV")
    st.write("**Mean RMSE:** 41.89")
    st.write("**Model Size:** 5.06 MB")

    st.divider()

    st.info(
        "This system predicts hourly bike rentals "
        "using historical weather and time information."
    )

    st.caption(
        "Dataset: UCI Bike Sharing (2011-2012)"
    )

# ----------------------------------
# MAIN DASHBOARD
# ----------------------------------

st.title("🚲 Bike Rental Prediction Dashboard")

st.write(
    "An intelligent machine learning system "
    "for predicting hourly bike rental demand."
)

st.divider()

# Performance Metrics

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Dataset Records",
        "17,379"
    )

with c2:
    st.metric(
        "Mean RMSE",
        "41.89"
    )

with c3:
    st.metric(
        "Cross-Validation",
        "10 Folds"
    )

st.divider()

# ----------------------------------
# USER INPUT FORM
# ----------------------------------

st.header("📋 Prediction Settings")

with st.form("bike_prediction_form"):

    st.subheader("📅 Date and Time")

    col1, col2, col3 = st.columns(3)

    with col1:

        season_name = st.selectbox(
            "Season",
            ["Winter", "Spring", "Summer", "Fall"]
        )

        season = {
            "Winter": 1,
            "Spring": 2,
            "Summer": 3,
            "Fall": 4
        }[season_name]

        yr = st.selectbox(
            "Historical Year",
            [0, 1],
            format_func=lambda x:
                "2011" if x == 0 else "2012"
        )

    with col2:

        mnth = st.slider(
            "Month",
            min_value=1,
            max_value=12,
            value=6
        )

        hr = st.slider(
            "Hour of Day",
            min_value=0,
            max_value=23,
            value=12
        )

    with col3:

        weekday = st.selectbox(
            "Day of Week",
            list(range(7)),
            format_func=lambda x: [
                "Sunday",
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday"
            ][x]
        )

        holiday = st.selectbox(
            "Holiday",
            [0, 1],
            format_func=lambda x:
                "Yes" if x == 1 else "No"
        )

    # Automatically calculate working day
    workingday = int(
        holiday == 0 and weekday not in [0, 6]
    )

    st.divider()

    st.subheader("🌤️ Weather Conditions")

    col4, col5 = st.columns(2)

    with col4:

        weathersit = st.selectbox(
            "Weather Situation",
            [1, 2, 3, 4],
            format_func=lambda x: {
                1: "Clear / Partly Cloudy",
                2: "Mist / Cloudy",
                3: "Light Rain / Snow",
                4: "Heavy Rain / Snow"
            }[x]
        )

        temp = st.slider(
            "Normalized Temperature",
            0.0, 1.0, 0.50
        )

        atemp = st.slider(
            "Normalized Feeling Temperature",
            0.0, 1.0, 0.50
        )

    with col5:

        hum = st.slider(
            "Normalized Humidity",
            0.0, 1.0, 0.50
        )

        windspeed = st.slider(
            "Normalized Wind Speed",
            0.0, 1.0, 0.20
        )

    submitted = st.form_submit_button(
        "🚀 Predict Bike Rentals",
        use_container_width=True
    )

# ----------------------------------
# MAKE PREDICTION
# ----------------------------------

if submitted:

    input_data = pd.DataFrame([{
        "season": season,
        "yr": yr,
        "mnth": mnth,
        "hr": hr,
        "holiday": holiday,
        "weekday": weekday,
        "workingday": workingday,
        "weathersit": weathersit,
        "temp": temp,
        "atemp": atemp,
        "hum": hum,
        "windspeed": windspeed
    }])

    # Match training feature order
    input_data = input_data[features]

    prediction = model.predict(input_data)[0]

    prediction = max(0, round(prediction))

    st.divider()

    st.header("📊 Prediction Result")

    st.success(
        "Prediction completed successfully!"
    )

    st.metric(
        "Estimated Hourly Bike Rentals",
        f"{prediction:,} bikes"
    )

    st.progress(
        min(prediction / 1000, 1.0)
    )

    st.caption(
        "Visualization scale: 0 to 1,000 bikes. "
        "This is not a model confidence score."
    )

    with st.expander("View Input Data"):
        st.dataframe(
            input_data,
            use_container_width=True
        )

# ----------------------------------
# MODEL PERFORMANCE INFORMATION
# ----------------------------------

st.divider()

st.subheader("📈 Model Performance")

performance = pd.DataFrame({
    "Model": [
        "Original Extra Trees",
        "Optimized Extra Trees"
    ],
    "Mean RMSE": [
        40.75,
        41.89
    ]
})

st.bar_chart(
    performance.set_index("Model")
)

st.caption(
    "Lower RMSE means better predictive accuracy. "
    "The optimized model is used for deployment."
)

# ----------------------------------
# FOOTER
# ----------------------------------

st.divider()

st.caption(
    "Machine Learning Assignment 3 | "
    "Bike Rental Prediction System | "
    "UCI Bike Sharing Dataset | "
    "Extra Trees Regressor"
)
