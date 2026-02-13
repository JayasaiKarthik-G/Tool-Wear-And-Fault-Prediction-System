import streamlit as st
import joblib


# Load the Trained Model & Encoder
rfc = joblib.load('model.joblib')
oe = joblib.load("type_encoder.joblib")
le = joblib.load('label_encoder.joblib')  

st.set_page_config(page_title="Tool Wear Prediction", layout="wide")
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
    selected_type = oe.transform([[type_map[selected_type]]])[0][0]


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
        pred_encoded = rfc.predict([input_features])

        # Decode prediction to actual label
        pred_label = le.inverse_transform(pred_encoded)[0]

        if pred_label == 'No Failure':
            st.success(f"🔍 Prediction: ✅ {pred_label}")
        else:
            st.error(f"🔍 Prediction: ⚠️ {pred_label}")
    
    except ValueError:
        st.warning("🚨 Please enter valid numerical inputs.")


# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This system predicts tool wear failure based on sensor inputs. "
        "It uses a Random Forest model trained on predictive maintenance data."
    )
