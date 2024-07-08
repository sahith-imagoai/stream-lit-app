# Mycotoxin Streamlit Applications

## Overview
This repository contains two Streamlit applications:
1. **Experiment Tracking**: For tracking and visualizing mycotoxin experiments for all the commodities.
2. **Mycotoxin Analysis**: For analyzing mycotoxin levels in Corn.


## Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/shweta-imagoai/streamlit-apps
    cd mycotoxin-apps or cdd experiment-tracking
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  
    ```

3. Install the required packages:
    - Python 3.8
    ```bash
    pip install -r requirements.txt
    ```

## Experiment Tracking

### Usage
To run the Experiment Tracking application, navigate to the `experiment-tracking` directory and start the Streamlit app:
```bash
cd experiment-tracking
streamlit run main.py
```
Link to the application - [Experiment Tracking App](https://stream-lit-app-asteibmlfzhzz47h3nngtq.streamlit.app/)
## Mycotoxin Analysis

### Usage
To run the Mycotoxin Analysis application, navigate to the mycotoxin-analysis directory and start the Streamlit app:

``` bash
cd mycotoxin-analysis
streamlit run main.py
```
Link to the application - [Mycotoxin Analysis App](https://mycotoxin-analysis.streamlit.app/)

## How to use :
 Please add the experiment file or sample file you wish to view for a comprehensive comparison of metrics and in-depth experiment analysis. Ensure the experiment name has not been previously defined to avoid overwriting existing files with the same experiment name.

There are a total of 6 tabs available:

Tab 1: Scatter Plot
Tab 2: Residual Plot
Tab 3: Metrics Comparison
Tab 4: NaN Data Visualization
Tab 5: Experiment Comparison (Compare with all previous experiments conducted)
Tab 6: Data Visualization (View incorrect predictions)
 
