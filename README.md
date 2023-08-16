# Energy Consumption Data Analysis Repository

This repository contains resources and tools for analyzing energy consumption datasets.

## 📄 Overview

- `Data_analysis_tasks_with_python.pdf`: A guide to tasks associated with this project.
- `electricity_consumption.ipynb`: Jupyter Notebook for visualizing and analyzing electricity consumption data.
- `energy_analysis.py`: Python script for energy data processing.
- `daily_lp_electricity_consumption.png`: A visual representation of daily load profile electricity consumption.
- `requirements.txt`: Required libraries and dependencies for running the project.

## 📁 Directory Structure

- `data`: Contains the datasets used for the project.
  - `preprocessed`: Contains preprocessed data files.
    - `lp.csv`: Load profile data in CSV format.
    - `lp_table.csv`: Table representation of the load profile data.
  - `raw`: Contains the raw datasets.
- `plots`: Contains visualizations derived from the data.
  - `AT`: (Contents to be detailed as per requirement.)
  - `lp`: Visualizations related to load profile.
    - Monthly breakdown visualizations for energy consumption for each month of 2019.
    - Summarized visualizations: daily, monthly, weekly, and yearly energy consumptions.
  - `profil`: (Contents to be detailed as per requirement.)

## 🛠 Getting Started

1. Clone the repository:
        - git clone https://github.com/Anas20001/Energy_consumption

2. Navigate to the project directory:
        - cd energy_consumption

3. Install the required packages:
        - pip install -r requirements.txt

4. Run the analysis script or open the Jupyter notebook to view visualizations. 

## 🚀 How to Run the Script

To successfully execute the `energy_analysis.py` script, follow the steps below:

1. **Set Up the Environment:**  
   Ensure you have Python installed on your system. This project is tested with Python 3.9, but other versions might be compatible.

2. **Activate a Virtual Environment (Optional, but Recommended):**  
   Before installing the dependencies, it's a good practice to use a virtual environment to avoid any package conflicts. 
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
  
## 🌐 Running the Streamlit App

To run the Streamlit app (`app.py`), follow these steps:

1. Ensure you have already installed the required packages from the `requirements.txt`.
   
2. Navigate to the project directory if you haven't already:

```bash 
cd energy_consumption
```
3. Run the Streamlit app:
```bash
streamlit run app.py
```
4. A new browser window or tab should open displaying the Streamlit app. If not, you can manually open the provided link in your preferred browser.


