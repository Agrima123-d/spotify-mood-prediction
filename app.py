import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("mood_model.pkl")

# Load scaler if you used one
use_scaler = False
try:
    scaler = joblib.load("scaler.pkl")
    use_scaler = True
except:
    pass

st.set_page_config(page_title="Spotify Mood Predictor", page_icon="🎵")

st.title("🎵 Spotify Mood Prediction App")
st.write("Enter Spotify audio features to predict the mood.")

danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
energy = st.slider("Energy", 0.0, 1.0, 0.5)
loudness = st.slider("Loudness", -60.0, 5.0, -10.0)
speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1)
acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
liveness = st.slider("Liveness", 0.0, 1.0, 0.2)
valence = st.slider("Valence", 0.0, 1.0, 0.5)
tempo = st.slider("Tempo", 0.0, 250.0, 120.0)

if st.button("Predict Mood"):
    input_df = pd.DataFrame([{
        "danceability": danceability,
        "energy": energy,
        "loudness": loudness,
        "speechiness": speechiness,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "valence": valence,
        "tempo": tempo
    }])

    if use_scaler:
        input_data = scaler.transform(input_df)
        prediction = model.predict(input_data)
    else:
        prediction = model.predict(input_df)

    st.success(f"Predicted Mood: {prediction[0]}")