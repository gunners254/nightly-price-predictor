import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Nightly Price Predictor", layout="centered")

@st.cache_resource
def load_assets():
    model = joblib.load("xgb_house_price_model_clean.pkl")
    expected_cols = joblib.load("expected_cols.pkl")
    return model, expected_cols

model, EXPECTED_COLS = load_assets()

st.title("Nightly Price Predictor")
st.write("Enter listing details to estimate the nightly price.")

# --- Inputs (keep it simple for non-technical users) ---
latitude = st.number_input("Latitude", value=42.66191, format="%.6f")
longitude = st.number_input("Longitude", value=-73.797441, format="%.6f")

neighbourhood_cleansed = st.text_input("Neighbourhood (cleansed)", value="FOURTEENTH WARD")
property_type = st.text_input("Property type", value="Entire rental unit")
room_type = st.text_input("Room type", value="Entire home/apt")

bedrooms = st.number_input("Bedrooms", value=2.0, min_value=0.0, step=1.0)
bathrooms = st.number_input("Bathrooms", value=1.0, min_value=0.0, step=0.5)
beds = st.number_input("Beds", value=2.0, min_value=0.0, step=1.0)
accommodates = st.number_input("Accommodates", value=4, min_value=1, step=1)

minimum_nights = st.number_input("Minimum nights", value=1, min_value=1, step=1)
maximum_nights = st.number_input("Maximum nights", value=365, min_value=1, step=1)

instant_bookable = st.selectbox("Instant bookable", ["f", "t"], index=0)
amenities_count = st.number_input("Amenities count", value=10, min_value=0, step=1)

def predict_price(feature_dict):
    row = {c: np.nan for c in EXPECTED_COLS}
    row.update(feature_dict)
    df_new = pd.DataFrame([row], columns=EXPECTED_COLS)
    pred_log = model.predict(df_new)[0]
    return float(np.expm1(pred_log))

if st.button("Predict"):
    house_input = {
        "latitude": latitude,
        "longitude": longitude,
        "neighbourhood_cleansed": neighbourhood_cleansed,
        "property_type": property_type,
        "room_type": room_type,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "beds": beds,
        "accommodates": accommodates,
        "minimum_nights": minimum_nights,
        "maximum_nights": maximum_nights,
        "instant_bookable": instant_bookable,
        "amenities_count": amenities_count,
    }

    price = predict_price(house_input)
    st.success(f"Predicted nightly price: ${price:,.2f}")
