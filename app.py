
import streamlit as st
import pandas as pd
import joblib

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Bike Rental Prediction",
    page_icon="🚲",
    layout="wide"
)

# --------------------------------
# LOAD TRAINED MODEL
# --------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("bike_rental_model.pkl")
    features = joblib.load("feature_names.pkl")
    return model, features

model, features = load_model()

# --------------------------------
# CUSTOM STYLING
# --------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #f5f7fc;
}

h1, h2, h3 {
    color: #16365c;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background-color: #1769aa;
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: bold;
    padding: 12px;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background-color: #104b80;
    color: white;
}

[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #e0e6ef;
    padding: 18px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------
# SIDEBAR
# --------------------------------

with st.sidebar:
    st.title("🚲 Bike Rental AI")
    st.caption("Machine Learning Assignment 3")

    st.divider()

    st.subheader("Model Information")
    st.write("**Algorithm:** Extra Trees Regressor")
    st.write("**Cross-Validation:** 10-Fold")
    st.write("**Mean RMSE:** 40.75")
    st.write("**Dataset:** UCI Bike Sharing")

    st.divider()

    st.info(
        "This application predicts hourly bike rental "
        "demand using historical weather and time data."
    )

# --------------------------------
# MAIN DASHBOARD
# --------------------------------

st.title("🚲 Bike Rental Prediction Dashboard")

st.write(
    "An intelligent machine learning system for "
    "predicting hourly bike rental demand."
)

st.divider()

# Performance cards
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset Records", "17,379")

with col2:
    st.metric("Best Model RMSE", "40.75")

with col3:
    st.metric("Cross-Validation", "10 Folds")

st.divider()

# --------------------------------
# INPUT FORM
# --------------------------------

st.header("📋 Prediction Settings")

with st.form("bike_prediction_form"):

    st.subheader("📅 Date and Time Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        season_label = st.selectbox(
            "Season",
            ["Spring", "Summer", "Fall", "Winter"]
        )

        season = {
            "Spring": 1,
            "Summer": 2,
            "Fall": 3,
            "Winter": 4
        }[season_label]

        yr = st.selectbox(
            "Historical Year",
            [0, 1],
            format_func=lambda x: "2011" if x == 0 else "2012"
        )

    with col2:
        mnth = st.slider("Month", 1, 12, 6)
        hr = st.slider("Hour of Day", 0, 23, 12)

    with col3:
        weekday = st.selectbox(
            "Day of Week",
            list(range(7)),
            format_func=lambda x: [
                "Sunday", "Monday", "Tuesday",
                "Wednesday", "Thursday", "Friday",
                "Saturday"
            ][x]
        )

        holiday = st.selectbox(
            "Holiday",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    workingday = int(
        holiday == 0 and weekday not in [0, 6]
    )

    st.divider()

    st.subheader("🌤️ Weather Information")

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

# --------------------------------
# PREDICTION
# --------------------------------

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

    # Match the model's original feature order
    input_data = input_data[features]

    prediction = model.predict(input_data)[0]
    prediction = max(0, round(prediction))

    st.divider()

    st.header("📊 Prediction Result")

    st.success("Prediction completed successfully!")

    st.metric(
        label="Estimated Hourly Bike Rentals",
        value=f"{prediction:,} bikes"
    )

    st.progress(min(prediction / 1000, 1.0))

    st.caption(
        "Visual scale: 0 to 1,000 rentals. "
        "This is not a model confidence score."
    )

    with st.expander("View Prediction Input Data"):
        st.dataframe(input_data, use_container_width=True)

# --------------------------------
# FOOTER
# --------------------------------

st.divider()

st.caption(
    "Bike Rental Prediction System | "
    "Machine Learning Assignment 3 | "
    "Extra Trees Regressor | UCI Bike Sharing Dataset"
)
