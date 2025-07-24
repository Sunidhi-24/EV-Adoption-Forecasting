# ⚡ EV-Adoption-Forecasting

## 🎯 Goal  
Build a regression model that predicts future Electric Vehicle (EV) adoption using historical registration data. This aids urban planners in anticipating infrastructure needs like charging stations.

---

## 📂 Dataset

**Source:** Kaggle - *Electric Vehicle Population Size 2024*

**Key Features:**

- **Date:** Registration count date (range: 2017-01-31 to 2024-02-29)
- **County:** Geographic region within a state where the vehicle is registered (mainly Washington)
- **State:** U.S. state related to the registration record
- **Vehicle Primary Use:** Vehicle purpose (Passenger – 83%, Truck – 17%)
- **Battery Electric Vehicles (BEVs):** Fully electric vehicle count
- **Plug-In Hybrid Electric Vehicles (PHEVs):** Partially electric vehicle count
- **Electric Vehicle (EV) Total:** Sum of BEVs and PHEVs
- **Non-Electric Vehicle Total:** Count of all other non-electric vehicles
- **Total Vehicles:** Combined EVs and non-EVs
- **Percent Electric Vehicles:** % of EVs among total vehicles

---

## 🛠️ Work Done

- ✅ **Preprocessing**
  - Formatted date column
  - Handled missing values
  - Encoded categorical features (e.g., County, State, Use)

- 🧹 **Data Cleaning**
  - Standardized EV total and percentage columns
  - Created `preprocessed_ev_data.csv` with clean and ready-to-train data

- 🤖 **Model Training**
  - Used **RandomForestRegressor** for prediction
  - Target: `Electric Vehicle (EV) Total`

- 📊 **Model Evaluation**
  - Metrics used:  
    - Mean Absolute Error (MAE)  
    - Mean Squared Error (MSE)  
    - R-squared (R²)
  - Plotted actual vs. predicted EV counts

- 💾 **Model Export**
  - Trained model saved as `ev_demand.pkl`
  - Label encoders saved as `encoders.pkl`

- 🧠 **Prediction Function**
  - Custom function accepts new user input and predicts future EV adoption

---

## 📁 Repository Structure

| File                          | Description                                                |
|-------------------------------|------------------------------------------------------------|
| `main.ipynb`                  | Jupyter notebook with full pipeline (EDA → Modeling)       |
| `dataset.csv`                 | Raw Kaggle dataset                                         |
| `preprocessed_ev_data.csv`   | Cleaned and encoded dataset ready for model training       |
| `forecasting_ev_model.pkl`   | Trained Random Forest model (same as `ev_demand.pkl`)      |
| `encoders.pkl`               | Encoded mappings for County, State, etc.                   |
| `ev_demand.pkl`              | Serialized trained model for inference                     |

---

## 🔧 Use Cases

- 📍 **Urban Planning:**  
  Helps city planners determine where to install new EV charging stations

- ⚡ **Power Grid Management:**  
  Assists energy companies in projecting load requirements due to EV growth

- 📈 **Policy Making:**  
  Supports government agencies in planning green transportation incentives

- 🏢 **Automotive Industry Forecasting:**  
  Offers insights for manufacturers and dealerships about future EV demand

---

## 🚀 Future Work

- 📡 Integrate real-time data via APIs for live predictions
- 📊 Build an interactive dashboard using Streamlit or Flask
- 🌍 Expand the model to include socio-economic and climate variables
- 🔍 Evaluate other regression models (XGBoost, LightGBM, etc.) for comparison
- 💬 Add a chatbot or voice assistant interface for querying forecasts
