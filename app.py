import streamlit as st
import joblib
import pandas as pd

# Load the Trained Model & Encoders
random_forest_model = joblib.load("models/model.joblib")
type_encoder = joblib.load("models/type_encoder.joblib")
label_encoder = joblib.load("models/label_encoder.joblib")

# Page config
st.set_page_config(page_title="Tool Wear Prediction", layout="wide")

# Hide Streamlit menu/header/footer
# st.markdown("""
#     <style>
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
#     header {visibility: hidden;}
#     </style>
# """, unsafe_allow_html=True)

# Now your app UI starts
st.title("🛠️ Tool Wear and Fault Prediction System")
st.markdown("Enter the following values to predict potential failure:")


# Getting the input data from the User
# Create columns
col1, col2 = st.columns(2)

with col1:
    selected_type = st.selectbox('Select a Type', ['Low', 'Medium', 'High'])

    #type_map = {"Low": "L", "Medium": "M", "High": "H"}
    #selected_type = int(oe.transform([[type_map[selected_type]]])[0][0])

    type_map = {"Low": "L", "Medium": "M", "High": "H"}
    selected_type = type_encoder.transform([[type_map[selected_type]]])[0][0]


with col2:
    air_temperature = st.text_input('Air temperature [K]')

with col1:
    process_temperature = st.text_input('Process temperature [K]')

with col2:
    rotational_speed = st.text_input('Rotational speed [rpm]')

with col1:
    torque = st.text_input('Torque [Nm]')

with col2:
    tool_wear = st.text_input('Tool wear [min]')


# Creating a button for Prediction
if st.button('Predict Failure'):
    try:
        # Convert inputs to float
        input_features = [
            float(air_temperature),
            float(process_temperature),
            float(rotational_speed),
            float(torque),
            float(tool_wear)
        ]
        
        # Include the Mapped Type as the first feature
        input_features = [selected_type] + input_features

        # Make prediction
        predicted_encoded = random_forest_model.predict([input_features])

        # Decode prediction to actual label
        predicted_label = label_encoder.inverse_transform(predicted_encoded)[0]

        if predicted_label == "No Failure":
            st.success(f"🔍 Prediction: ✅ {predicted_label}")
        else:
            st.error(f"🔍 Prediction: ⚠️ {predicted_label}")
    
    except ValueError:
        st.warning("🚨 Please enter valid numerical inputs.")


# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This system predicts tool wear failure based on sensor inputs. "
        "It uses a Random Forest model trained on predictive maintenance data."
    )
    st.subheader("📄 Dataset CSV")

    csv_file_path = "data/predictive_maintenance.csv" 

    try:
        # Load CSV
        df = pd.read_csv(csv_file_path)

        # Preview CSV in sidebar
        with st.expander("View CSV File"):
            st.dataframe(df)

        # Download CSV
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download CSV File",
            data=csv_data,
            file_name="predictive_maintenance.csv",
            mime="text/csv"
        )

    except FileNotFoundError:
        st.warning("CSV file not found.")
