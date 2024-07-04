import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio

# Set the default plotly theme to dark
pio.templates.default = "plotly_dark"

# Define a custom color palette
color_palette = [
    '#00CED1',  # Dark Turquoise
    '#FF6347',  # Tomato
    '#32CD32',  # Lime Green
    '#FFD700',  # Gold
    '#9370DB',  # Medium Purple
    '#FF69B4',  # Hot Pink
    '#20B2AA',  # Light Sea Green
    '#F08080',  # Light Coral
    '#7B68EE',  # Medium Slate Blue
    '#00FA9A',  # Medium Spring Green
]

# Function to load and process the uploaded CSV file
@st.cache_data
def load_data(file):
    """
    Function to load and process the uploaded CSV file.
    Args:
        file (UploadedFile): Uploaded CSV file.
    Returns:
        DataFrame: Processed data.
    """
    data = pd.read_csv(file)
    return data

def save_uploaded_file(uploaded_file):
    """
    Function to save the uploaded file locally.
    Args:
        uploaded_file (UploadedFile): Uploaded CSV file.
    """
    with open("ground_corn_mycotoxin_icp.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())

def ppm_to_ppb(value, unit):
    """
    Function to convert ppm to ppb.
    Args:
        value (float): Value to convert.
        unit (str): Unit of the value (ppm or ppb).
    Returns:
        float: Converted value in ppb.
    """
    return value * 1000 if unit.lower() == 'ppm' else value



def display_kde_plot(data, selected_uname, selected_test):
    """
    Function to display KDE plot.
    Args:
    data (DataFrame): Filtered data.
    selected_uname (str): Selected UNAME.
    selected_test (str): Selected test.
    """
    st.subheader("Result Distribution")
    unique_combinations = data.groupby(['uname', 'test'])
    
    # Check if we have enough data for KDE plot
    if len(unique_combinations) == 0:
        st.warning("Not enough data to create a KDE plot.")
        return
    
    valid_groups = []
    valid_labels = []
    single_value_groups = []

    for name, group in unique_combinations:
        if len(group['result_ppb']) > 1:
            valid_groups.append(group['result_ppb'])
            valid_labels.append(f"{name[0]} - {name[1]}")
        else:
            single_value_groups.append((name, group['result_ppb'].iloc[0]))

    if len(valid_groups) > 0:
        fig_kde = ff.create_distplot(
            valid_groups,
            valid_labels,
            show_hist=False,
            show_rug=False,
            colors=color_palette[:len(valid_groups)]
        )

        fig_kde.update_layout(
            title="KDE Plot of Result Distribution",
            xaxis_title="Result (ppb)",
            yaxis_title="Density",
            legend_title="uname - test"
        )

        # Add vertical lines for single-value groups
        for i, (name, value) in enumerate(single_value_groups):
            fig_kde.add_vline(x=value, line_dash="dash", line_color=color_palette[len(valid_groups) + i],
                              annotation_text=f"{name[0]} - {name[1]}: {value:.2f}")

        st.plotly_chart(fig_kde, use_container_width=True)

    else:
        # If no groups have more than one value, create a simple scatter plot
        fig = go.Figure()
        for i, (name, value) in enumerate(single_value_groups):
            fig.add_trace(go.Scatter(
                x=[value], 
                y=[0], 
                mode='markers',
                name=f"{name[0]} - {name[1]}",
                marker=dict(size=10, color=color_palette[i])
            ))

        fig.update_layout(
            title="Single Value Result Distribution",
            xaxis_title="Result (ppb)",
            yaxis_title="",
            showlegend=True,
            legend_title="uname - test"
        )

        # Hide y-axis
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

        st.plotly_chart(fig, use_container_width=True)

    # Display information about single-value groups
    if single_value_groups:
        st.info("The following groups have only one value:")
        for name, value in single_value_groups:
            st.write(f"{name[0]} - {name[1]}: {value:.2f} ppb")

def display_histogram(data, selected_uname, selected_test,filter_type):
    """
    Function to display histogram.
    Args:
        data (DataFrame): Filtered data.
        selected_uname (str): Selected UNAME.
        selected_test (str): Selected test.
        filter_type (str): Type of date filter applied.
    """
    fig_hist = px.histogram(
        data,
        x='result_ppb',
        color='uname',
        facet_col='test',
        marginal="box",
        # hover_data=['date'] + data.columns.tolist(),
        color_discrete_sequence=color_palette
    )

    fig_hist.update_layout(
        title="Histogram of Result Distribution by Test",
        xaxis_title="Result (ppb)",
        yaxis_title="Count",
        legend_title="uname"
    )

    # Update facet titles
    for annotation in fig_hist.layout.annotations:
        annotation.text = annotation.text.split("=")[-1]

    # Add filter information to the title
    if filter_type == "Date Range":
        date_info = f"{data['date'].min().date()} to {data['date'].max().date()}"
    elif filter_type == "Year":
        date_info = f"Year: {data['year'].iloc[0]}"
    elif filter_type == "Month":
        date_info = f"Month: {data['month'].iloc[0]}"
    elif filter_type == "Week":
        date_info = f"Week: {data['week'].iloc[0]}"
    
    fig_hist.update_layout(title=f"Histogram of Result Distribution by Test<br><sup>{filter_type}: {date_info}</sup>")

    st.plotly_chart(fig_hist, use_container_width=True)


def display_summary_statistics(data,filter_type):
    """
    Function to display summary statistics.
    Args:
        data (DataFrame): Filtered data.
        filter_type (str): Type of date filter applied.
    """
    st.subheader("Summary Statistics")
    
    # Add filter information
    if filter_type == "Date Range":
        date_info = f"{data['date'].min().date()} to {data['date'].max().date()}"
    elif filter_type == "Year":
        date_info = f"Year: {data['year'].iloc[0]}"
    elif filter_type == "Month":
        date_info = f"Month: {data['month'].iloc[0]}"
    elif filter_type == "Week":
        date_info = f"Week: {data['week'].iloc[0]}"
    
    st.write(f"Date Filter: {filter_type} - {date_info}")

    summary = data.groupby(['uname', 'test']).agg({
        'result_ppb': ['mean', 'min', 'max', 'count']
    }).reset_index()
    summary.columns = ['uname', 'mycotoxin', 'mean_result (ppb)', 'min_result (ppb)', 'max_result (ppb)', 'count']
    st.dataframe(summary)

    return summary


def display_customer_count(data_custom):
    """
    Function to display count of samples by customer_name.
    Args:
    data (DataFrame): Filtered data.
    """


    data = data_custom[data_custom['customer_name'].notna()]
    if data.empty:
        st.info("No customer data available for the current selection.")
        return

    st.subheader("Sample Count by Customer")


    # Count samples for each customer
    customer_counts = data['customer_name'].value_counts().reset_index()
    customer_counts.columns = ['customer_name', 'Count']

    # Add a row for null/unknown customer if exists
    null_count = data['customer_name'].isnull().sum()
    if null_count > 0:
        customer_counts = customer_counts.append({'customer_name': 'Unknown', 'Count': null_count}, ignore_index=True)

    # Sort by count in descending order
    customer_counts = customer_counts.sort_values('Count', ascending=False)

    # Create the bar plot
    fig = px.bar(customer_counts, x='customer_name', y='Count', 
                 title='Number of Samples by Customer',
                 labels={'customer_name': 'Customer', 'Count': 'Number of Samples'},
                 color='Count',
                 color_continuous_scale=px.colors.sequential.Viridis)

    fig.update_layout(xaxis_tickangle=-45, xaxis_title="", yaxis_title="Number of Samples")

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

    # Display the data as a table
    # st.write("Sample Count Data:")
    # st.dataframe(customer_counts)

def main(data_file):
    """
    Main function to run the Streamlit application.
    """
    # Main application title
    st.title("Mycotoxin Analysis")

    # File uploader in the sidebar
    uploaded_file = st.sidebar.file_uploader("Select your CSV files", type=["csv"])

    if uploaded_file:
        save_uploaded_file(uploaded_file)
        data = load_data(uploaded_file)
    else:
        data = load_data(data_file)

    # Convert result to ppb
    data['result_ppb'] = data.apply(lambda row: ppm_to_ppb(row['result'], row['unit']), axis=1)
    data['date'] = pd.to_datetime(data['date'])



    # Create dropdown menus for user selections
    uname_options = ['All'] + sorted(data['uname'].unique().tolist())
    test_options = sorted(data['test'].unique().tolist())


    col1, col2 = st.columns(2)
    with col1:
        selected_uname = st.selectbox("Select Uname", uname_options)
    with col2:
        selected_test = st.selectbox("Select Mycotoxin", test_options)


    # Filter the data based on user selections
    filtered_data = data.copy()
    if selected_uname != 'All':
        filtered_data = filtered_data[filtered_data['uname'] == selected_uname]
    if selected_test != 'All':
        filtered_data = filtered_data[filtered_data['test'] == selected_test]


    # Add date components to filtered data
    filtered_data['year'] = filtered_data['date'].dt.year
    filtered_data['month'] = filtered_data['date'].dt.to_period('M')
    filtered_data['week'] = filtered_data['date'].dt.to_period('W')

    # Date filtering options
    st.header("Date Filtering Options")
    filter_type = st.selectbox("Select date filter type", 
                               ["Date Range", "Year", "Month", "Week"])

    if filter_type == "Date Range":
        min_date = filtered_data['date'].min().date()
        max_date = filtered_data['date'].max().date()
        date_range = st.date_input(
            "Select Date Range",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, max_date
        filtered_data = filtered_data[(filtered_data['date'].dt.date >= start_date) & 
                                      (filtered_data['date'].dt.date <= end_date)]
    elif filter_type == "Year":
        selected_year = st.selectbox("Select Year", sorted(filtered_data['year'].unique(), reverse=True))
        filtered_data = filtered_data[filtered_data['year'] == selected_year]
    elif filter_type == "Month":
        selected_month = st.selectbox("Select Month", sorted(filtered_data['month'].unique(), reverse=True))
        filtered_data = filtered_data[filtered_data['month'] == selected_month]
    elif filter_type == "Week":
        selected_week = st.selectbox("Select Week", sorted(filtered_data['week'].unique(), reverse=True))
        filtered_data = filtered_data[filtered_data['week'] == selected_week]


    if filtered_data.empty:
        st.error("No data points for the selected filters.")
    else:
        display_kde_plot(filtered_data, selected_uname, selected_test)
        # display_histogram(filtered_data, selected_uname, selected_test,filter_type)
        summary = display_summary_statistics(filtered_data, filter_type)
        # display_summary_visualizations(summary, filtered_data, selected_uname, selected_test)

        display_customer_count(filtered_data)        # Display raw data
        st.subheader("Raw Data")
        st.dataframe(filtered_data)

        # Add download button for CSV
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name="filtered_mycotoxin_data.csv",
            mime="text/csv",
        )

# Run the main function
if __name__ == "__main__":
    data_file = "ground_corn_mycotoxin_icp.csv"
    main(data_file)