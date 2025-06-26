# NYC Taxi Rides Data Analysis

> “The goal is to turn data into information, and information into insight.”  
> — Carly Fiorina

## Overview

This project provides an exploratory data analysis (EDA) of NYC taxi rides using Python and NumPy. The analysis uncovers patterns in taxi usage, passenger behavior, and operational metrics that can inform business decisions and urban planning.

## Features

- **Data Loading:** Efficiently loads and inspects a large NYC taxi dataset.
- **Data Cleaning:** Handles missing values and ensures data integrity.
- **Statistical Analysis:** Computes key metrics such as average speed, ride counts by month, and tip distributions.
- **Visualization:** Visualizes trends and distributions for deeper insights.
- **Actionable Insights:** Summarizes findings for practical use.

## Dataset

The dataset (`nyc_taxis.csv`) contains information about NYC taxi rides, including:
- Pickup and drop-off times and locations
- Trip distance and duration
- Fare and tip amounts
- Payment types

## Analysis Highlights

- **Average Speed of Taxis:**  
  Calculated using trip distance and duration, considering only valid trips.

- **Rides in February:**  
  Counts the number of rides that occurred in the month of February.

- **Rides with Tips Greater Than $50:**  
  Identifies high-tip rides for further business insights.

- **Drop-offs at NYC Airport:**  
  Quantifies the number of rides ending at NYC airports.

## Data Visualization

To better understand the data, we included the following visualizations:

- **Distribution of Taxi Speeds:**
  - A histogram showing the distribution of calculated taxi speeds (in mph).
  - Outliers and unrealistic values (e.g., speeds outside 0–100 mph) are filtered out for clarity.
  - This helps reveal the most common speed ranges and spot any anomalies in the data.

- **Number of Rides by Month:**
  - A bar chart displaying the number of taxi rides for each month.
  - This visualization highlights seasonal trends and peaks in taxi usage throughout the year.

## How to Run

1. Clone this repository.
2. Ensure you have Python 3.x, NumPy, and Matplotlib installed.
3. Place `nyc_taxis.csv` in the project directory.
4. Open and run `CabsDataAnalysis.ipynb` in Jupyter Notebook or JupyterLab.

## Example Usage

```python
import numpy as np
taxi = np.genfromtxt('nyc_taxis.csv', delimiter=',', skip_header=1)
# Further analysis as shown in the notebook
```

## Results

- The average speed of NYC taxis is approximately **32 mph**.
- There were over **13,000 rides** in February.
- **16 rides** had tips greater than $50.
- Over **11,800 rides** ended at NYC airports.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements or new features.

## License

This project is licensed under the MIT License.

---

*For more details, see the full analysis in `CabsDataAnalysis.ipynb`.*
