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

def is_y_pred_in_range_don(y_true,y_pred):
    lower_bound = y_true - 0.2 * y_true
    upper_bound = y_true + 0.2 * y_true
    if lower_bound <= y_pred <= upper_bound:
        return 'Yes'
    else :
        return 'No'




def is_y_pred_in_range_fum(y_true,y_pred):
    lower_bound = y_true - 0.26 * y_true
    upper_bound = y_true + 0.26 * y_true
    if lower_bound <= y_pred <= upper_bound:
        return 'Yes'
    else :
        return 'No'
def check_acceptable_range_zea(y_true, y_pred):
    # Define the specifications based on the given table
    # Assuming target_concentration is any value and not fixed
    if y_true<=50 :
        if y_pred <=50 :
         return 'Yes'
        return 'No'
    elif 50<y_true <= 100 :
        lower_bound = y_true-50
        upper_bound = y_true+50
    elif 100<y_true <= 250:
        lower_bound = y_true-100
        upper_bound = y_true+100
    elif 250<y_true<400:
        lower_bound = y_true-200
        upper_bound = y_true+200
    elif 400< y_true <= 1000 :
        lower_bound = y_true-400
        upper_bound = y_true+400
    else :
        lower_bound = y_true-400
        upper_bound = y_true+400



    # Check if y_pred falls within the acceptable range
    if lower_bound <= y_pred <= upper_bound:
        return 'Yes'
    else:
        return 'No'
def is_y_pred_in_range_zea(y_true,y_pred):
    lower_bound = y_true - 0.4 * y_true
    upper_bound = y_true + 0.4 * y_true
    if lower_bound <= y_pred <= upper_bound:
        return 'Yes'
    else :
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

def calculate_ranking_score(filtered_data):
    y_true = filtered_data['y_true']
    y_pred = filtered_data['y_pred']
    
    # Calculate R2 scores
    r2 = r2_score(y_true, y_pred)
    y_true_tcs_non_zero = filtered_data[(filtered_data['source'] == 'TCS') & (filtered_data['y_true'] != 0)]['y_true']
    y_pred_tcs_non_zero = filtered_data[(filtered_data['source'] == 'TCS') & (filtered_data['y_true'] != 0)]['y_pred']
    tcs_non_zero_r2 = r2_score(y_true_tcs_non_zero, y_pred_tcs_non_zero)
    y_true_trm_non_zero = filtered_data[(filtered_data['source'] == 'TRM') & (filtered_data['y_true'] != 0)]['y_true']
    y_pred_trm_non_zero = filtered_data[(filtered_data['source'] == 'TRM') & (filtered_data['y_true'] != 0)]['y_pred']
    trm_non_zero_r2 = r2_score(y_true_trm_non_zero, y_pred_trm_non_zero)
    
    # Count occurrences where y_true == 0
    no_count_0 = filtered_data[filtered_data['y_true'] == 0].shape[0]
    
    tcs_no_count_0 = filtered_data[(filtered_data['Yes / No'] == 'No') & (filtered_data['source'] == 'TCS') & (filtered_data['y_true'] == 0)].shape[0]
    trm_no_count_0 = filtered_data[(filtered_data['Yes / No'] == 'No') & (filtered_data['source'] == 'TRM') & (filtered_data['y_true'] == 0)].shape[0]
    
    # Calculate All R2 (assuming it's for all data)
    all_r2 = r2_score(y_true, y_pred)
    
    # Define a composite ranking score (you can adjust weights as needed)
    if tcs_no_count_0 == 0 : 
        tcs_no_count_0 = 1
    else :
        tcs_no_count_0 = (1/tcs_no_count_0)
    if trm_no_count_0 == 0:
        trm_no_count_0 = 1
    else : 
        trm_no_count_0 = (1/trm_no_count_0)
    
    ranking_score = ((tcs_no_count_0) * 0.3) + \
                    ((trm_no_count_0) * 0.25) + \
                    (tcs_non_zero_r2 * 0.2) + \
                    (trm_non_zero_r2 * 0.15) + \
                    (r2 * 0.1)  # Adjust weights based on importance
    
    return ranking_score 
    
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
                rank_data_list = []
                for exp_name in selected_experiments:
               
                  filtered_data = filter_experiment_data(experiment_data[exp_name], 
                                               mycotoxin=selected_mycotoxin if selected_mycotoxin != 'All' else None,
                                               location=selected_location if selected_location != 'All' else None,
                                               type=selected_type if selected_type != 'All' else None,
                                               source=selected_source if selected_source != 'All' else None)
                  df = experiment_data[exp_name]
                  y_true = filtered_data['y_true']
                  y_pred = filtered_data['y_pred']
                  
                  r2 = r2_score(y_true, y_pred)
                  mae = mean_absolute_error(y_true, y_pred)
                  rmse = mean_squared_error(y_true, y_pred, squared=False)
                  no_count = 0
                  
                  if(mycotoxins[0] == 'AFLA'):
                      filtered_data['Yes / No'] = filtered_data.apply(lambda row: is_y_pred_in_range(row['y_true'], row['y_pred']), axis=1)
                      df['Yes / No'] = df.apply(lambda row: is_y_pred_in_range(row['y_true'], row['y_pred']), axis=1)
                  elif(mycotoxins[0]=='DON'):
                      filtered_data['Yes / No'] = filtered_data.apply(lambda row: is_y_pred_in_range_don(row['y_true'], row['y_pred']), axis=1)
                      df['Yes / No'] = df.apply(lambda row: is_y_pred_in_range_don(row['y_true'], row['y_pred']), axis=1)
                  elif(mycotoxins[0]=='FUM'):
                      filtered_data['Yes / No'] = filtered_data.apply(lambda row: is_y_pred_in_range_fum(row['y_true'], row['y_pred']), axis=1)
                      df['Yes / No'] = df.apply(lambda row: is_y_pred_in_range_fum(row['y_true'], row['y_pred']), axis=1)
                  elif(mycotoxins[0]=='ZEA'):
                      filtered_data['Yes / No'] = filtered_data.apply(lambda row: check_acceptable_range_zea(row['y_true'], row['y_pred']), axis=1)
                      df['Yes / No'] = df.apply(lambda row: check_acceptable_range_zea(row['y_true'], row['y_pred']), axis=1)
                  yes_count = filtered_data[filtered_data['Yes / No'] == 'Yes'].shape[0]
                  no_count = filtered_data[filtered_data['Yes / No'] == 'No'].shape[0]
                  no_count_0 = filtered_data[(filtered_data['Yes / No'] == 'No') & (filtered_data['y_true'] == 0)].shape[0]
                  ranking_score = calculate_ranking_score(df)
    # Append the metrics to the list
                  rank_data_list.append({
                      "Experiment Name": exp_name,
                      "ranking_score":ranking_score
                  })
                  data_list.append({
        "Experiment Name": exp_name,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "Incorrect Predictions": no_count,
        "Incorrect Predictions at 0":no_count_0,
        
    })

                final_comparision_metrics = pd.DataFrame(data_list)
                
                rank_data_list = pd.DataFrame(rank_data_list)
                rank_data_list = rank_data_list.sort_values(by='ranking_score', ascending=False)
                st.write(final_comparision_metrics)
                st.write(rank_data_list)
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
               if(mycotoxin_options[0]=='AFLA'):
                filtered_data['Yes / No'] = filtered_data.apply(lambda row: is_y_pred_in_range(row['y_true'], row['y_pred']), axis=1)
               elif(mycotoxin_options[0]=='DON'):
                filtered_data['Yes / No'] = filtered_data.apply(lambda row: is_y_pred_in_range_don(row['y_true'], row['y_pred']), axis=1)
               elif(mycotoxin_options[0]=='FUM'):
                filtered_data['Yes / No'] = filtered_data.apply(lambda row: is_y_pred_in_range_fum(row['y_true'], row['y_pred']), axis=1)
               elif(mycotoxin_options[0]=='ZEA'):
                filtered_data['Yes / No'] = filtered_data.apply(lambda row: check_acceptable_range_zea(row['y_true'], row['y_pred']), axis=1)

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

    st.title('Experiment Data Analysis')
    folder_path = "experiment_results"
    st.write("Only select the experiments with same mycotoxin value")
    if not os.path.exists(folder_path):
        st.warning(f"No experiment data found in {folder_path}")
        st.stop()
    experiment_data = load_experiment_data(folder_path)
    experiment_names = list(experiment_data.keys())
    selected_experiments = st.multiselect("Select Experiments for Comparison", experiment_names)
    st.header("Delete Previously done experiment data")
    delete_experiments = st.checkbox("Select experiments to delete")
    if delete_experiments:
      experiments_to_delete = st.multiselect("Select Experiments to Delete", experiment_names)
    if st.button("Delete Selected Experiments"):
        for exp_name in experiments_to_delete:
            file_path = os.path.join(folder_path, f"{exp_name}.csv")
            if os.path.exists(file_path):
                os.remove(file_path)
        st.success("Selected experiments deleted successfully.")
    if not selected_experiments:
        st.info("Select one or more experiments to compare")
        st.stop()
                

        # Add a separator between experiments
    st.markdown("---")

    combined_data = pd.concat([experiment_data[exp_name] for exp_name in selected_experiments], ignore_index=True)

    first_key = next(iter(experiment_data))
   
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
    rank_data_list = []
    for exp_name in selected_experiments:
        df = experiment_data[exp_name]
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
        if(mycotoxins[0] == 'AFLA'):
            filtered_data['Yes / No'] = filtered_data.apply(lambda row: is_y_pred_in_range(row['y_true'], row['y_pred']), axis=1)
            df['Yes / No'] = df.apply(lambda row: is_y_pred_in_range(row['y_true'], row['y_pred']), axis=1)
        elif(mycotoxins[0]=='DON'):
            filtered_data['Yes / No'] = filtered_data.apply(lambda row: is_y_pred_in_range_don(row['y_true'], row['y_pred']), axis=1)
            df['Yes / No'] = df.apply(lambda row: is_y_pred_in_range_don(row['y_true'], row['y_pred']), axis=1)
        elif(mycotoxins[0]=='FUM'):
            filtered_data['Yes / No'] = filtered_data.apply(lambda row: is_y_pred_in_range_fum(row['y_true'], row['y_pred']), axis=1)
            df['Yes / No'] = df.apply(lambda row: is_y_pred_in_range_fum(row['y_true'], row['y_pred']), axis=1)
        elif(mycotoxins[0]=='ZEA'):
            filtered_data['Yes / No'] = filtered_data.apply(lambda row: check_acceptable_range_zea(row['y_true'], row['y_pred']), axis=1)
            df['Yes / No'] = df.apply(lambda row: check_acceptable_range_zea(row['y_true'], row['y_pred']), axis=1)
        yes_count = filtered_data[filtered_data['Yes / No'] == 'Yes'].shape[0]
        no_count = filtered_data[filtered_data['Yes / No'] == 'No'].shape[0]
        no_count_0 = filtered_data[(filtered_data['Yes / No'] == 'No') & (filtered_data['y_true'] == 0)].shape[0]
        ranking_score = calculate_ranking_score(df)
        
    # Append the metrics to the list
        rank_data_list.append({
            "Experiment Name": exp_name,
            "ranking_score":ranking_score
        })
        data_list.append({
        "Experiment Name": exp_name,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "Incorrect Predictions": no_count,
        "Incorrect Predictions at 0":no_count_0,
        
    })

    final_comparision_metrics = pd.DataFrame(data_list)
    rank_data_list = pd.DataFrame(rank_data_list)
    rank_data_list = rank_data_list.sort_values(by='ranking_score', ascending=False)
    st.write(final_comparision_metrics)
    st.write(rank_data_list)
    csv = final_comparision_metrics.to_csv(index=False)
    st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name=f"experiments_performance_data.csv",
                    mime="text/csv")
    
    

                
        
