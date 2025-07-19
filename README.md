# EV-Adoption-Forecasting

## Goal
Build a regression model that predicts future Electric Vehicle (EV) adoption using historical registration data. This aids urban planners in anticipating infrastructure needs like charging stations.

## Dataset
- **Source:** [Kaggle - Electric Vehicle Population Size 2024](https://www.kaggle.com/datasets/sahirmaharajj/electric-vehicle-population-size-2024/data)  
- **Key Features:**  
  - Date: Counts of registered vehicles are taken on this day (the end of this month). - 2017-01-31 2024-02-29
  - County: This is the geographic region of a state that a vehicle's owner is listed to reside within. Vehicles registered in Washington
  - State: This is the geographic region of the country associated with the record. These addresses may be located in other
  - Vehicle Primary Use: This describes the primary intended use of the vehicle.(Passenger-83%, Truck-17%)
  - Battery Electric Vehicles (BEVs): The count of vehicles that are known to be propelled solely by an energy derived from an onboard electric battery.
  - Plug-In Hybrid Electric Vehicles (PHEVs): The count of vehicles that are known to be propelled from energy partially sourced from an onboard electric battery
  - Electric Vehicle (EV) Total: The sum of Battery Electric Vehicles (BEVs) and Plug-in Hybrid Electric Vehicles (PHEVs).
  - Non-Electric Vehicle Total: The count of vehicles that are not electric vehicles.
  - Total Vehicles: All powered vehicles registered in the county. This includes electric vehicles.
  - Percent Electric Vehicles: Comparison of electric vehicles versus their non-electric counterparts.

## 🛠️ Work Done
- Performed preprocessing, including date formatting, missing value handling, and categorical feature encoding.
- Cleaned and standardized the EV total count column for consistent numerical analysis.
- Trained a machine learning regression model using RandomForestRegressor to forecast EV adoption.
- Evaluated model performance using key metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²).
- Visualized actual vs. predicted values to assess prediction accuracy.
- Serialized and exported the trained model (`ev_demand.pkl`) and associated label encoders (`encoders.pkl`) for future inference.
- Developed a custom prediction function that accepts user inputs and returns forecasted EV demand.
