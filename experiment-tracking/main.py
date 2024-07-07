import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import statsmodels.api as sm

## CONFIG
json_file_path = "experiments_metrics.json"

if 'tab' not in st.session_state:
    st.session_state['tab'] = 'Experiment Comparison'

# Function to load and process the uploaded CSV file
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data

# Function to filter the data based on user selections

def filter_data(data, source, mycotoxin, location, type,target_column='y_true'):
    filtered_data = data.copy()  # Make a copy of the original data to avoid modifying it directly
    filtered_data['hsi_sample_id'] = filtered_data['hsi_sample_id'].astype('str')
    data_with_nan = filtered_data[filtered_data[target_column].isna()]
    filtered_data = filtered_data.dropna(subset=[target_column])

    if source != 'ALL':
        filtered_data = filtered_data[filtered_data['source'] == source]
    
    if mycotoxin is not None:
        filtered_data = filtered_data[filtered_data['mycotoxin'] == mycotoxin]
    
    if location != 'ALL':
        filtered_data = filtered_data[filtered_data['location'] == location]
    
    if type != 'ALL':
        filtered_data = filtered_data[filtered_data['type'] == type]

    return filtered_data, data_with_nan

rsdmax_mapping = {
    5: 25,
    20: 20,
    100: 16,
    300: 16
}

def get_nearest_rsdmax(value):
    nearest_rsdmax = min(rsdmax_mapping.keys(), key=lambda x: abs(x - value))
    return rsdmax_mapping[nearest_rsdmax]
def load_experiment_data(folder_path):
    experiment_data = {}
    files = os.listdir(folder_path)
    for file in files:
        if file.endswith(".csv"):
            experiment_name = os.path.splitext(file)[0]
            experiment_data[experiment_name] = pd.read_csv(os.path.join(folder_path, file))
    return experiment_data
def filter_experiment_data(data, mycotoxin=None, type=None, location=None, source=None):
    filtered_data = data.copy()

    if mycotoxin:
        filtered_data = filtered_data[filtered_data['mycotoxin'] == mycotoxin]

    if type:
        filtered_data = filtered_data[filtered_data['type'] == type]

    if location:
        filtered_data = filtered_data[filtered_data['location'] == location]

    if source:
        filtered_data = filtered_data[filtered_data['source'] == source]

    return filtered_data
# Function to calculate acceptable range
def calculate_acceptable_range(y_true):
    rsdmax = get_nearest_rsdmax(y_true)
    lower_bound = y_true - 0.02 * rsdmax * y_true
    upper_bound = y_true + 0.02 * rsdmax * y_true
    return lower_bound, upper_bound


def is_y_pred_in_range(y_true, y_pred):
    if y_true <= 10 and y_pred <= 10:
        return 'Yes'
    elif 10 < y_true <= 35 and 10 < y_pred <= 35:
        return 'Yes'
    elif 50 < y_true <= 100 and 50 < y_pred <= 100:
        return 'Yes'
    elif 200 < y_true <= 300 and 200 < y_pred <= 300:
        return 'Yes'
    elif y_true > 300 and y_pred > 250:
        return 'Yes'
    else:
        return 'No'    
def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae

def is_experiment_name_unique(experiment_name, json_file_path):
    if not os.path.exists(json_file_path):
        return True  # File doesn't exist, so the name is unique
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        for experiment in data.keys():
            if experiment == experiment_name:
                return False
    return True

def save_experiment_data(complete_data, experiment_name):
    save_dir = "experiment_results"
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Generate file paths for saving CSV files
    complete_data_file = os.path.join(save_dir, f"{experiment_name}_complete_data.csv")

    # Save complete data to CSV file
    complete_data.to_csv(complete_data_file, index=False)
    st.success(f"Complete data saved successfully to {complete_data_file}")

   
    
def save_metrics_to_json(metrics, file_name, filters, experiment_name, sample_count):
    json_file = "experiments_metrics.json"
    
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            data = json.load(f)

    else:
        data = {}

    if experiment_name not in data:
        data[experiment_name] = {}

    data[experiment_name] = {
        "filters": filters,
        "metrics": metrics,
        "sample_count": sample_count
    }
    
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)
    
    return json_file


def load_json_data(json_file):
    with open(json_file, "r") as f:
        return json.load(f)
    
def process_data(data):
    processed_data = {}
    # max_data = {}
    for exp_name, exp_data in data.items():
        commodity = exp_data['filters']['commodity']
        mycotoxin = exp_data['filters']['mycotoxin']
        if commodity not in processed_data:
            processed_data[commodity] = {}

        if mycotoxin not in processed_data[commodity]:
            processed_data[commodity][mycotoxin] = []
        processed_data[commodity][mycotoxin].append({
            'name': exp_name,
            'metrics': exp_data['metrics'],
            'filters': exp_data['filters'],
            'sample_count': exp_data['sample_count']
        })
    return processed_data



# Main application
st.title("Mycotoxin Experiment Tracking")
st.sidebar.header("Upload your CSV files")
### Add not (file columns should be like )

st.sidebar.markdown("""
### Note: File columns should be as follows:

The CSV file should have the following columns:
- `hsi_sample_id`: Unique identifier for each sample
- `location`: Location of the sample
- `y_true`: True value (ground truth)
- `y_pred`: Predicted value
- `source`: Source of the prediction(TRM, TCS)
- `type`: Type of sample (AS, GR)
- `mycotoxin`: Type of mycotoxin (AFLA, DON, FUM, ZEA)

Example row:
hsi_sample_id,location,y_true,y_pred,source,type,mycotoxin
182015,GCS,0.47,0,TRM,GR,aflatoxin
                    """)


# File uploader
uploaded_files = st.sidebar.file_uploader("Select your CSV files", type=["csv"], accept_multiple_files=True)
experiment_name = st.sidebar.text_input("Enter experiment name (e.g., resnet-v1-satyam)", placeholder = "resnet-v1-satyam")

if experiment_name.strip():
    if not is_experiment_name_unique(experiment_name, json_file_path):
        st.sidebar.error(f"The experiment name '{experiment_name}' already exists. Please choose a different name.")
    else:
        st.sidebar.success(f"The experiment name '{experiment_name}' is available.")
elif experiment_name:
    st.sidebar.error("Experiment name cannot be whitespace only. Please enter a valid name.")


if uploaded_files:
    for file in uploaded_files:
        # Load and process the uploaded CSV file
        data = load_data(file)
        file_name = file.name.split('.')[0]  # Use the base name of the file for saving

        # Create dropdown menus for user selections
        commodity_options = ['Corn', 'Barley', 'Wheat', 'CGM', 'DDGS']
        source_options = ['ALL'] + sorted(data['source'].unique().tolist())
        # Assuming you want unique options for mycotoxin, location, and type
        mycotoxin_options = sorted(data['mycotoxin'].unique().tolist())
        location_options = ['ALL'] + sorted(data['location'].unique().tolist())
        type_options = ['ALL'] + sorted(data['type'].unique().tolist())

        commodity = st.sidebar.selectbox("Select a commodity", commodity_options, key=f"commodity_{file_name}")
        source = st.sidebar.selectbox("Select a source", source_options, key=f"source_{file_name}")
        mycotoxin = st.sidebar.selectbox("Select a mycotoxin", mycotoxin_options, key=f"mycotoxin_{file_name}")
        location = st.sidebar.selectbox("Select a Location", location_options, key=f"location_{file_name}")
        type = st.sidebar.selectbox("Select a Type", type_options, key=f"type_{file_name}")


        # Filter the data based on user selections
        filtered_data, data_with_nan = filter_data(data, source, mycotoxin, location, type)
        

        if filtered_data.empty:

            st.error("No valid data points for the selected filters. However, there are NaN values in the data.")
        else:
            y = filtered_data['y_true']
            y_pred = filtered_data['y_pred']

            residual = y- y_pred
            filtered_data['residuals'] = residual
            r2, rmse, mae = calculate_metrics(y, y_pred)

            # Display metrics
            sample_count = len(filtered_data)
            col1, col2, col3 , col4= st.columns(4)
            col1.metric("R2 Score", f"{r2:.4f}")
            col2.metric("RMSE", f"{rmse:.4f}")
            col3.metric("MAE", f"{mae:.4f}")
            col4.metric("Sample Count", sample_count)


            ###saving metrics
            metrics = {
                "R2": r2,
                "RMSE": rmse,
                "MAE": mae
            }
            filters = {
                "commodity": commodity,
                "source": source,
                "mycotoxin": mycotoxin,
                "location": location,
                "type": type
            }
            if experiment_name != "":
                json_file = save_metrics_to_json(metrics, file_name, filters, experiment_name, sample_count)
                save_experiment_data(data, experiment_name)

            tab1, tab2, tab3, tab4, tab5 ,tab6= st.tabs(["Scatter Plot", "Residual Plot", "Metric Comparison", "NaN Data Visualization", "Experiment Comparison","Data Visualization"])

            with tab1:
                st.header("Scatter Plot: True vs Predicted")
                fig_scatter = px.scatter(filtered_data, x='y_true', y='y_pred', color='source', trendline="ols")
                fig_scatter.add_trace(go.Scatter(x=[filtered_data['y_true'].min(), filtered_data['y_true'].max()], 
                                                y=[filtered_data['y_true'].min(), filtered_data['y_true'].max()],
                                                mode='lines', name='Ideal', line=dict(dash='dash')))
                fig_scatter.update_layout(
                xaxis_title="True Values (PPB)",
                yaxis_title="Predicted Values (PPB)"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            with tab2:
                st.header("Residual Plot")
                fig_residual = px.scatter(filtered_data, x='y_pred', y='residuals', color='source')
                fig_residual.add_hline(y=0, line_dash="dash", line_color="red")
                fig_residual.update_layout(
                xaxis_title="Predicted Values (PPB)",
                yaxis_title="Residual Values (PPB)"
                )
                st.plotly_chart(fig_residual, use_container_width=True)

            with tab3:
                st.header("Metric Comparison by source")
                metrics_df = filtered_data.groupby('source').apply(lambda x: pd.Series({
                    'R2': r2_score(x['y_true'], x['y_pred']),
                    'RMSE': np.sqrt(mean_squared_error(x['y_true'], x['y_pred'])),
                    'MAE': mean_absolute_error(x['y_true'], x['y_pred'])
                })).reset_index()
                
                metric_type = st.selectbox('Select Metric', ['R2', 'RMSE', 'MAE'])
                fig_metric = px.bar(metrics_df, x='source', y=metric_type, color='source')
                st.plotly_chart(fig_metric, use_container_width=True)
        
            with tab4:
                st.header("NaN Data Visualization")
                
                # Filters for NaN data
                nan_location_options = ['ALL'] + sorted(data_with_nan['location'].unique().tolist())
                nan_type_options = ['ALL'] + sorted(data_with_nan['type'].unique().tolist())
                
                col1, col2 = st.columns(2)
                with col1:
                    nan_location = st.selectbox('Select Location', nan_location_options, index=0)
                with col2:
                    nan_type = st.selectbox('Select Type', nan_type_options, index=0)

                # Filter NaN data based on selection
                nan_filtered_data = data_with_nan.copy()

                if nan_location != 'ALL':
                    nan_filtered_data = nan_filtered_data[nan_filtered_data['location'] == nan_location]
                if nan_type != 'ALL':
                    nan_filtered_data = nan_filtered_data[nan_filtered_data['type'] == nan_type]
                
                # Create scatter plot for NaN data
                if not data_with_nan.empty:
                    fig_nan = px.scatter(nan_filtered_data, x='hsi_sample_id', y='y_pred', 
                                        color='source', hover_data=['location', 'type'],
                                        labels={'index': 'Sample', 'y_pred': 'Predicted Value'})
                    
                    fig_nan.update_layout(title='Predictions for Samples with NaN Ground Truth')
                    st.plotly_chart(fig_nan, use_container_width=True)
                else:
                    st.info("No NaN values in the data.")

                # Display the filtered NaN data
                st.subheader("Filtered NaN Data")
                st.dataframe(nan_filtered_data)
            with tab5: 
                st.title('Experiment Data Analysis')
                folder_path = "experiment_results"
                st.write("Only select the experiments with same mycotoxin value")
                if not os.path.exists(folder_path):
                  st.warning(f"No experiment data found in {folder_path}")
                  st.stop()
                experiment_data = load_experiment_data(folder_path)
                experiment_names = list(experiment_data.keys())
                selected_experiments = st.multiselect("Select Experiments for Comparison", experiment_names)
                if not selected_experiments:
                  st.info("Select one or more experiments to compare")
                  st.stop()
                

        # Add a separator between experiments
                st.markdown("---")

                combined_data = pd.concat([experiment_data[exp_name] for exp_name in selected_experiments], ignore_index=True)

                first_key = next(iter(experiment_data))
                print(first_key)
                print(set(combined_data['mycotoxin']))
                if(len(set(combined_data['mycotoxin'])) > 1):
                   st.warning("Warning: You have selected different mycotoxins at a time. Please verify.")
                mycotoxins = sorted(set(combined_data['mycotoxin']))
                locations = sorted(set(combined_data['location']))
                types = sorted(set(combined_data['type']))
                sources = sorted(set(combined_data['source']))

                selected_mycotoxin = st.selectbox("Filter by Mycotoxin", ['All'] + mycotoxins)
                selected_location = st.selectbox("Filter by Location", ['All'] + locations)
                selected_type = st.selectbox("Filter by Type", ['All'] + types)
                selected_source = st.selectbox("Filter by Source", ['All'] + sources)
                filtered_experiment_data = {}
                data_list = []

                for exp_name in selected_experiments:
                  print(exp_name)
                  filtered_data = filter_experiment_data(experiment_data[exp_name], 
                                               mycotoxin=selected_mycotoxin if selected_mycotoxin != 'All' else None,
                                               location=selected_location if selected_location != 'All' else None,
                                               type=selected_type if selected_type != 'All' else None,
                                               source=selected_source if selected_source != 'All' else None)
                  y_true = filtered_data['y_true']
                  y_pred = filtered_data['y_pred']
                  
                  r2 = r2_score(y_true, y_pred)
                  mae = mean_absolute_error(y_true, y_pred)
                  rmse = mean_squared_error(y_true, y_pred, squared=False)
                  no_count = 0
                 
                  filtered_data['Yes / No'] = filtered_data.apply(lambda row: is_y_pred_in_range(row['y_true'], row['y_pred']), axis=1)
                  yes_count = filtered_data[filtered_data['Yes / No'] == 'Yes'].shape[0]
                  no_count = filtered_data[filtered_data['Yes / No'] == 'No'].shape[0]
                  no_count_0 = filtered_data[(filtered_data['Yes / No'] == 'No') & (filtered_data['y_true'] == 0)].shape[0]

    # Append the metrics to the list
                  data_list.append({
        "Experiment Name": exp_name,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "Incorrect Predictions": no_count,
        "Incorrect Predictions at 0":no_count_0
    })

                final_comparision_metrics = pd.DataFrame(data_list)
                st.write(final_comparision_metrics)
                
                st.header("Experiment Performance Comparison")
                if os.path.exists(json_file_path):
                   raw_data = load_json_data(json_file_path)
                if raw_data:
                   data = process_data(raw_data)
           
                            
            with tab6:
               st.header("Data Visualization")

    # Define filter options
               mycotoxin_options = list(filtered_data['mycotoxin'].unique())
               source_options = list(filtered_data['source'].unique())
               location_options = list(filtered_data['location'].unique())
               type_options = list(filtered_data['type'].unique())

    # Allow selection of mycotoxin type, source, location, and type
               selected_mycotoxin = st.selectbox('Select Mycotoxin Type', ['All'] + mycotoxin_options)
               selected_source = st.selectbox('Select Source', ['All'] + source_options)
               selected_location = st.selectbox('Select Location', ['All'] + location_options)
               selected_type = st.selectbox('Select Type', ['All'] + type_options)

    # Filter data based on selected filters
               filtered_data = filtered_data.copy()
               if selected_mycotoxin != 'All':
                filtered_data = filtered_data[filtered_data['mycotoxin'] == selected_mycotoxin]
               if selected_source != 'All':
                filtered_data = filtered_data[filtered_data['source'] == selected_source]
               if selected_location != 'All':
                filtered_data = filtered_data[filtered_data['location'] == selected_location]
               if selected_type != 'All':
                filtered_data = filtered_data[filtered_data['type'] == selected_type]

    # Calculate acceptable ranges and yes/no predictions
               acceptable_ranges = filtered_data['y_true'].apply(calculate_acceptable_range)
               filtered_data['Acceptable Range AFLA (ppb)'] = acceptable_ranges.apply(lambda x: f"[{x[0]}, {x[1]}]")
               filtered_data['Yes / No'] = filtered_data.apply(lambda row: is_y_pred_in_range(row['y_true'], row['y_pred']), axis=1)
               yes_count = filtered_data[filtered_data['Yes / No'] == 'Yes'].shape[0]
               no_count = filtered_data[filtered_data['Yes / No'] == 'No'].shape[0]

    # Display the filtered data
               st.subheader("Data Visualization for y_true, y_pred, and mycotoxin")
               st.dataframe(filtered_data[['mycotoxin', 'y_true', 'y_pred',  'location', 'source', 'type', 'Yes / No',]])
               st.write(f"Number of Correct Predictions: {yes_count}")
               st.write(f"Number of Incorrect Predictions: {no_count}")



    # Example: Custom visualization based on your requirements
    # Add your own custom visualization code here

               if filtered_data.empty:
                st.info("No data available for the selected filters.")


            
else:                
    st.header("Experiment Performance Comparison")

    if os.path.exists(json_file_path):
        raw_data = load_json_data(json_file_path)
        if raw_data:
            data = process_data(raw_data)
            

            # Create a dropdown to select the commodity
            commodities = ['Corn', 'Barley', 'Wheat', 'CGM', 'DDGS']
            selected_commodity = st.selectbox("Select Commodity", commodities)
            
            if selected_commodity in data:
                commodity_data = data[selected_commodity]
                mycotoxins = list(commodity_data.keys())
                
                # all_sources = list(set(exp['filters']['source'] for mycotoxin in mycotoxins for exp in commodity_data[mycotoxin]))
                # Create filters for location, type, and mycotoxin
                all_locations = list(set(exp['filters']['location'] for mycotoxin in mycotoxins for exp in commodity_data[mycotoxin]))
                all_types = list(set(exp['filters']['type'] for mycotoxin in mycotoxins for exp in commodity_data[mycotoxin]))
                all_sources = ['ALL','TRM', 'TCS', 'Others']

                selected_mycotoxin = st.selectbox("Filter by Mycotoxin", mycotoxins)
                selected_source = st.selectbox("Filter by Source", all_sources)
                selected_location = st.selectbox("Filter by Location", all_locations)
                selected_type = st.selectbox("Filter by Type", all_types)

                # Filter data based on selected location, type, and mycotoxin
                filtered_commodity_data = {}
                for mycotoxin, experiments in commodity_data.items():
                    if selected_mycotoxin == "ALL" or mycotoxin == selected_mycotoxin:
                        if selected_source == "ALL":
                            source_filtered_experiments = experiments
                        else:
                            source_filtered_experiments = [exp for exp in experiments if exp['filters']['source'] == selected_source]
                        
                        if not source_filtered_experiments:
                            continue  # Skip to next mycotoxin if no data available after source filtering
    
                        
                        filtered_experiments = [
                            exp for exp in source_filtered_experiments
                            if (selected_location == "ALL" or exp['filters']['location'] == selected_location) and
                            (selected_type == "ALL" or exp['filters']['type'] == selected_type)
                        ]
                        
                        if filtered_experiments:
                            filtered_commodity_data[mycotoxin] = filtered_experiments
                if not filtered_commodity_data:
                    st.info("No performance data available")
                    st.stop()
                # Display max_data in a matrix format
                max_data = {}
                for mycotoxin, experiments in filtered_commodity_data.items():
                    if experiments:  # Ensure there are experiments for this mycotoxin
                        max_data[mycotoxin] = {
                            'R2': max(exp['metrics']['R2'] for exp in experiments),
                            'RMSE': min(exp['metrics']['RMSE'] for exp in experiments),
                            'MAE': min(exp['metrics']['MAE'] for exp in experiments)
                        }
                    else:
                        max_data[mycotoxin] = {
                        'R2': None,
                        'RMSE': None,
                        'MAE': None
                        }

                st.subheader("Max Data Metrics")    
                if selected_mycotoxin == "ALL":
                    for mycotoxin, metrics in max_data.items():
                        st.write(f"**{mycotoxin}**")
                        max_df = pd.DataFrame([metrics])
                        st.dataframe(max_df)
                else:
                    if selected_mycotoxin in max_data:
                        st.write(f"**{selected_mycotoxin}**")
                        max_df = pd.DataFrame([max_data[selected_mycotoxin]])
                        st.dataframe(max_df)                                
                
                st.subheader("Experiments")    
            
                fig = sp.make_subplots(
                    rows=len(filtered_commodity_data), cols=1, 
                    subplot_titles=list(filtered_commodity_data.keys()),
                    vertical_spacing=0.1
                )

                colors = ['#636EFA', '#EF553B', '#00CC96']  # Colors for the bars

                for i, (mycotoxin, experiments) in enumerate(filtered_commodity_data.items(), 1):
                    x = [exp['name'] for exp in experiments]
                    r2 = [exp['metrics']['R2'] for exp in experiments]
                    rmse = [exp['metrics']['RMSE'] for exp in experiments]
                    mae = [exp['metrics']['MAE'] for exp in experiments]
                    
                    hover_text = [
                        f"Experiment: {exp['name']}<br>"
                        f"R2: {exp['metrics']['R2']:.4f}<br>"
                        f"RMSE: {exp['metrics']['RMSE']:.4f}<br>"
                        f"MAE: {exp['metrics']['MAE']:.4f}<br>"
                        f"Source: {exp['filters']['source']}<br>"
                        f"Location: {exp['filters']['location']}<br>"
                        f"Type: {exp['filters']['type']}<br>"
                        f"Sample Count: {exp['sample_count']}"
                        for exp in experiments
                    ]
                    
                    fig.add_trace(go.Bar(x=x, y=r2, name='R2', marker_color=colors[0], text=[f"{val:.4f}" for val in r2], textposition='auto', hovertext=hover_text, hoverinfo='text'), row=i, col=1)
                    fig.add_trace(go.Bar(x=x, y=rmse, name='RMSE', marker_color=colors[1], text=[f"{val:.4f}" for val in rmse], textposition='auto', hovertext=hover_text, hoverinfo='text'), row=i, col=1)
                    fig.add_trace(go.Bar(x=x, y=mae, name='MAE', marker_color=colors[2], text=[f"{val:.4f}" for val in mae], textposition='auto', hovertext=hover_text, hoverinfo='text'), row=i, col=1)
                
                fig.update_layout(
                    height=400 * len(filtered_commodity_data),
                    title_text=f"Performance Metrics for {selected_commodity}",
                    barmode='group',
                    legend_title_text='Metrics'
                )
                
                fig.update_xaxes(title_text="Experiments", tickangle=45)
                fig.update_yaxes(title_text="Metric Value")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display detailed information in a table
                st.subheader(f"Detailed Performance for {selected_commodity}")
                
                table_data = []
                for mycotoxin, experiments in filtered_commodity_data.items():
                    for exp in experiments:
                        table_data.append({
                            "Mycotoxin": mycotoxin,
                            "Experiment": exp['name'],
                            "R2": f"{exp['metrics']['R2']:.4f}",
                            "RMSE": f"{exp['metrics']['RMSE']:.4f}",
                            "MAE": f"{exp['metrics']['MAE']:.4f}",
                            "Source": exp['filters']['source'],
                            "Location": exp['filters']['location'],
                            "Type": exp['filters']['type'],
                            "Sample Count": exp['sample_count']
                        })
                
                df = pd.DataFrame(table_data)
                st.dataframe(df)
                
                # Add download button for CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name=f"{selected_commodity}_performance_data.csv",
                    mime="text/csv",
                )
        else:
            st.info("No performance data available")