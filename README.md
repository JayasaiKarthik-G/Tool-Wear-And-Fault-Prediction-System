# 🛠️ Tool Wear and Fault Prediction System

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.1.1-brightgreen.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.26.0-blue.svg)

---

# 📌 Overview

The **Tool Wear and Fault Prediction System** is a Machine Learning web application that predicts whether an industrial machine is likely to experience a failure based on real-time operating parameters.

The application is built using **Python**, **Streamlit**, and **Scikit-Learn**. A trained **Random Forest Classifier** analyzes machine sensor values and predicts the machine condition, helping users identify potential failures before they occur.

This project demonstrates the practical implementation of **Predictive Maintenance**, where machine learning techniques are used to reduce downtime, improve equipment reliability, and support preventive maintenance decisions.

---

# 🚀 Features

- Predict machine failure instantly
- User-friendly Streamlit interface
- Random Forest Machine Learning model
- Encoded categorical feature support
- Download dataset directly from the application
- Interactive dataset preview
- Clean and responsive UI
- Fast prediction with trained model
- Easy deployment using Streamlit Cloud

---

# 🧠 Machine Learning Model

| Model | Random Forest Classifier |
|--------|--------------------------|
| Problem Type | Classification |
| Target | Machine Failure |
| Framework | Scikit-Learn |

The model predicts whether a machine is likely to fail based on machine operating conditions.

---

# 📂 Project Structure

```
Tool-Wear-And-Fault-Prediction-System/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── data/
│   └── predictive_maintenance.csv
│
├── models/
│   ├── model.joblib
│   ├── type_encoder.joblib
│   └── label_encoder.joblib
│
├── notebooks/
│   └── Tool_Wear_And_Fault_Prediction.ipynb
│
└── .gitignore
```
---

## 📓 Google Colab Notebook

Explore the complete model development process, including data preprocessing, model training, and evaluation.

Click [here](notebooks/Tool_Wear_And_Fault_Prediction.ipynb) to open the notebook.

---

## 🌐 Live Demo

You can access the deployed application here:

https://tool-wear-and-fault-prediction-system.onrender.com/

---

## 📊 Dataset Information

This project uses the **Machine Predictive Maintenance Classification** dataset containing industrial machine operating parameters.

**Kaggle Dataset:**
https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification


## Dataset Description

The dataset consists of the following features:
- `UID`: Unique identifier ranging from 1 to 10000.
- `productID`: Product quality variant with letters L, M, or H, and a variant-specific serial number.
- `air temperature [K]`: Generated using a random walk process, later normalized to a standard deviation of 2 K around 300 K.
- `process temperature [K]`: Generated using a random walk process, normalized to a standard deviation of 1 K, added to the air temperature plus 10 K.
- `rotational speed [rpm]`: Calculated from power of 2860 W, overlaid with normally distributed noise.
- `torque [Nm]`: Torque values are normally distributed around 40 Nm with an Ïƒ = 10 Nm and no negative values.
- `tool wear [min]`: The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process.
- `machine failure`: A label that indicates whether the machine has failed in this particular data point for any of the following failure modes.

### Input Features

- Machine Type
- Air Temperature (K)
- Process Temperature (K)
- Rotational Speed (RPM)
- Torque (Nm)
- Tool Wear (Minutes)

### Output

- No Failure
- Specific Failure

The dataset contains sensor values collected from machines under different operating conditions and is used to train the prediction model.

---

# ⚙️ Technologies Used

- Python 3.11
- Streamlit
- Scikit-Learn
- Pandas
- NumPy
- Joblib

---

# 📦 Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install streamlit==1.28.0
pip install scikit-learn==1.3.2
pip install pandas==2.1.1
pip install numpy==1.26.0
pip install joblib==1.3.2
```

---

# 💻 Installation Guide

## Step 1

Clone the repository

```bash
git clone https://github.com/JayasaiKarthik-G/Tool-Wear-And-Fault-Prediction-System.git
```

---

## Step 2

Navigate into the project directory

```bash
cd Tool-Wear-and-Fault-Prediction-System
```

---

## Step 3 (Optional): Create and Activate a Virtual Environment

Create a virtual environment:


### Windows (Command Prompt)

```bash
python -m venv venv

venv\Scripts\activate
```

### Windows (PowerShell)

```bash
python -m venv venv

venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

Once the virtual environment is activated, your terminal prompt will display `(venv)`.

Example:

```bash
> (venv) C:\Users\YourName\Tool-Wear-and-Fault-Prediction-System>
```

---

## Step 4

Install dependencies

```bash
pip install -r requirements.txt
```
---

## Step 5

Train the Machine Learning Model

```bash
python train_model.py
```
Model and encoders saved successfully!
Generated files:

models/
├── model.joblib
├── type_encoder.joblib
└── label_encoder.joblib

---

# ▶️ Running the Application

Run the Streamlit application:

```bash
streamlit run app.py
```

After execution, Streamlit will automatically open the application in your default browser.

If it doesn't open automatically, visit:

```bash
http://localhost:8501
```
To stop a running Streamlit server, use one of these methods:

```bash
Ctrl + C
```
---

# 🖥️ Application Workflow

### Step 1

Launch the application using Streamlit.

---

### Step 2

Select the machine type:

- Low
- Medium
- High

---

### Step 3

Enter machine operating values:

- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

---

### Step 4

Click

```
Predict Failure
```

---

### Step 5

The trained Random Forest model processes the inputs.

---

### Step 6

Prediction is displayed as:

✅ No Failure

or

⚠️ Failure

---

# 📸 User Interface

The application provides:

- Clean dashboard
- Interactive form
- Instant prediction
- Dataset preview
- Dataset download option
- Sidebar information

---

# 🔍 Input Parameters

| Feature | Description |
|----------|-------------|
| Type | Machine Quality (Low, Medium, High) |
| Air Temperature | Temperature of surrounding air (Kelvin) |
| Process Temperature | Internal machine process temperature |
| Rotational Speed | Machine speed (RPM) |
| Torque | Torque generated by machine |
| Tool Wear | Tool wear duration (minutes) |

---

# 📈 Prediction Output

The model predicts one of the following:

### ✅ No Failure

Machine is operating normally.

### ⚠️ Failure

Machine is likely to experience failure and may require maintenance.

---

# 📄 Dataset Preview

The application includes:

- CSV preview
- Scrollable data table
- Download CSV button

This allows users to inspect the dataset directly from the web application.

---

# 📚 Model Files

The application loads the following trained files:

```
model.joblib
```

Random Forest trained model.

```
type_encoder.joblib
```

Encodes machine type.

```
label_encoder.joblib
```

Converts prediction labels into readable output.

---

# 🧪 Jupyter Notebook

The project includes:

```
Machine_Predictive_Maintenance_Classification.ipynb
```

The notebook contains:

- Data preprocessing
- Feature engineering
- Label encoding
- Model training
- Model evaluation
- Model saving
- Prediction testing

## 📓 Running the Jupyter Notebook (Optional)

If you want to explore the data preprocessing, model training, and evaluation steps, you can run the Jupyter Notebook included in this project.

### Install Jupyter Notebook

```bash
pip install notebook ipykernel
```

or

```bash
pip install jupyter
```

### Open the Notebook in VS Code

1. Install the **Python** and **Jupyter** extensions in Visual Studio Code.
2. Open the project folder in VS Code.
3. Open `Machine_Predictive_Maintenance_Classification.ipynb`.
4. Select the Python interpreter (kernel).
5. Click **Run All** or execute cells individually.

> **Note:** This step is optional. The Streamlit application (`app.py`) runs independently and does not require the Jupyter Notebook.

---

# 🛡️ Error Handling

The application validates user inputs.

If invalid values are entered, it displays:

```
Please enter valid numerical inputs.
```

preventing incorrect predictions.

---

# 🎯 Use Cases

- Predictive Maintenance
- Manufacturing Industry
- Industrial IoT
- Machine Health Monitoring
- Equipment Failure Detection
- Smart Factory Solutions
- Educational Machine Learning Projects

---

# 🌟 Future Improvements

- Deploy on Streamlit Cloud
- Add prediction probability
- Visualize feature importance
- Upload CSV for batch predictions
- Support multiple ML models
- Interactive charts
- Historical prediction logs
- API integration
- Dark mode UI
- User authentication

---

# 📝 Requirements

```
Python >= 3.11.9

Streamlit == 1.28.0

Scikit-Learn == 1.3.2

Pandas == 2.1.1

NumPy == 1.26.0

Joblib == 1.3.2
```

---


# 👨‍💻 Author

**Gadekari Jayasai Karthik**

Java Full Stack Developer

- 🌐 Portfolio: https://jayasai-karthik.vercel.app
- 💼 LinkedIn: https://www.linkedin.com/in/gadekari-jayasai-karthik/
- 💻 GitHub: https://github.com/JayasaiKarthik-G

---

# ⭐ Support

If you found this project helpful, please consider giving it a ⭐ on GitHub.

Your support motivates future open-source development.

---