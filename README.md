# BikeFlow Seoul

## Project overview

BikeFlow is a machine learning project that explores and predicts hourly bike rental demand in Seoul using historical rental, weather and calendar data.

The project includes exploratory data analysis, statistical hypothesis testing, external API integration, model comparison and an interactive Streamlit application.

## Data sources

- UCI Machine Learning Repository – Seoul Bike Sharing Demand Dataset
- Open-Meteo Historical Weather API – cloud cover data
- 8,760 hourly observations from December 2017 to November 2018

## Model performance

Random Forest was selected as the final model after comparison with Linear Regression.

| Test metric | Random Forest |
|---|---:|
| MAE | 219.52 bikes/hour |
| RMSE | 312.52 bikes/hour |
| R² | 0.73 |

## Installation

Developed and tested on macOS using Python 3.12.15.

Clone the repository:

```bash
git clone https://github.com/ChandaNovent/BikeFlow.git
cd BikeFlow
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate the environment on macOS:

```bash
source .venv/bin/activate
```

Or on Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application allows users to explore historical rental patterns and predict hourly bike rentals based on selected weather and calendar conditions.

## Run the analysis

Open `01_bikeflow_analysis.ipynb` in VS Code or Jupyter Notebook and run all cells.

The enriched dataset and trained model are included in the repository. No live API connection is required.

## Limitations

The model is based on one year of historical data and may underestimate peak rental demand. Predictions represent estimated rentals rather than real-time forecasts.

## Acknowledgements

- UCI Machine Learning Repository
- Open-Meteo
- Seoul skyline photograph by Clark Gu on Unsplash
