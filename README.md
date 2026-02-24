🌋 Balmatt Seismic Warning System & Dashboard
This repository contains a Hybrid Seismic Warning System designed to predict significant seismic events (Magnitude > 0.5) within a 48-hour window. It uses a combination of Rule-Based thresholds (for high precision) and Machine Learning anomaly detection based on a sliding 3-day window of operational metrics.

Included is a Streamlit dashboard (dashboard.py) for visualizing operational data and real-time seismic risk warnings.

⚠️ Important Data Disclaimer: Dummy Dataset
The data provided in this repository is entirely synthetic (dummy data). Because the original operational and seismic data belongs to my university and is strictly confidential, it cannot be shared publicly.

To ensure this code can be run, reviewed, and tested by the open-source community, I have generated dummy datasets that perfectly mirror the original structure, column names, and data types:

operational_metrics.csv: Simulated hourly injection/production flow, pressure, and temperature.

seismic_events_1.csv: Simulated seismic events with randomized magnitudes and GIS locations.

risk_log.csv: Simulated traffic-light warning logs (Green, Yellow, Red).

Note: Because the data is randomly generated, the predictive accuracy or physical correlation of the machine learning models trained on this specific dataset will not reflect real-world physics. It is provided purely to demonstrate the data engineering pipeline and software architecture.

🚀 How to Run (Step-by-Step)
If you are using GitHub Codespaces or a local terminal, follow these steps in order:

1. Install Dependencies
Bash
pip install -r requirements.txt
2. Setup the Environment
The system requires a folder to store the trained model "brain."

Bash
mkdir -p trained_models
3. Train the Hybrid Model
You must train the model on the dummy data first to generate the prediction file.

Bash
python train_hybrid_3day.py
4. Launch the Dashboard
Run the Streamlit application to view the results.

Bash
python -m streamlit run dashboard.py
🛠️ Project Structure
dashboard.py: The main Streamlit application for monitoring risks.

train_hybrid_3day.py: Trains the sliding 3-day window Random Forest model.

predictor_3day.py: The inference script used by the dashboard to make predictions.

check_missing.py: A utility script to analyze operational datasets for shutdown periods.

requirements.txt: List of required Python libraries.

🔧 Technical Fixes Applied
To ensure the system runs correctly with dummy data and modern Python environments, the following logic was applied:

Robust Date Parsing: Added dayfirst=True and errors='coerce' to pd.to_datetime functions to handle international date formats correctly.

Local Pathing: Updated file paths to reference the root directory directly, ensuring compatibility with GitHub's flat-file structure.
