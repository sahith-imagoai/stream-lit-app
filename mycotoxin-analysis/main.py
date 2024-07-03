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

    fig_kde = ff.create_distplot(
        [group['result_ppb'] for name, group in unique_combinations],
        [f"{name[0]} - {name[1]}" for name, group in unique_combinations],
        show_hist=False,
        show_rug=False,
        colors=color_palette[:len(unique_combinations)]
    )

    fig_kde.update_layout(
        title="KDE Plot of Result Distribution",
        xaxis_title="Result (ppb)",
        yaxis_title="Density",
        legend_title="uname - test"
    )

    if selected_test == 'All' or selected_uname == 'All':
        fig_kde.update_xaxes(type="log", title="Result (log ppb)")
        fig_kde.update_yaxes(type="log", title="Result (log density)")
        
    st.plotly_chart(fig_kde, use_container_width=True)

def display_histogram(data, selected_uname, selected_test):
    """
    Function to display histogram.
    Args:
        data (DataFrame): Filtered data.
        selected_uname (str): Selected UNAME.
        selected_test (str): Selected test.
    """
    if selected_test != 'All' and selected_uname != 'All':
        fig_hist = px.histogram(
            data,
            x='result_ppb',
            color='uname',
            facet_col='test',
            marginal="box",
            hover_data=data.columns,
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

        if selected_test == 'All' or selected_uname == 'All':
            fig_hist.update_xaxes(type="log", title="Result (log ppb)")
            fig_hist.update_yaxes(type="log", title="Result (log density)")

        st.plotly_chart(fig_hist, use_container_width=True)

def display_summary_statistics(data):
    """
    Function to display summary statistics.
    Args:
        data (DataFrame): Filtered data.
    """
    st.subheader("Summary Statistics")
    summary = data.groupby(['uname', 'test']).agg({
        'result_ppb': ['mean', 'median', 'min', 'max', 'count']
    }).reset_index()
    summary.columns = ['uname', 'test', 'mean_result', 'median_result', 'min_result', 'max_result', 'count']
    st.dataframe(summary)

    return summary

def display_summary_visualizations(summary, data, selected_uname, selected_test):
    """
    Function to display summary visualizations.
    Args:
        summary (DataFrame): Summary statistics data.
        data (DataFrame): Filtered data.
        selected_uname (str): Selected UNAME.
        selected_test (str): Selected test.
    """
    st.subheader("Summary Statistics Visualizations")

    # Bar chart for mean and median results
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=summary['uname'] + ' - ' + summary['test'], y=summary['mean_result'], name='Mean'))
    fig_bar.add_trace(go.Bar(x=summary['uname'] + ' - ' + summary['test'], y=summary['median_result'], name='Median'))
    fig_bar.update_layout(
        title='Mean and Median Results by uname and mycotoxin',
        xaxis_title='uname - test',
        yaxis_title='Result (ppb)',
        barmode='group'
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    if selected_test == 'All' or selected_uname == 'All':
        # Box plot for result distribution
        fig_box = px.box(data, x='uname', y='result_ppb', color='test',
                         title='Result Distribution by uname and mycotoxin')
        fig_box.update_layout(yaxis_title='Result (ppb)')
        st.plotly_chart(fig_box, use_container_width=True)

    # Heatmap for count
    fig_heatmap = px.imshow(summary.pivot(index='uname', columns='test', values='count'),
                            labels=dict(x='test', y='uname', color='Count'),
                            title='Sample Count Heatmap')
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Scatter plot for min vs max
    fig_scatter = px.scatter(
        summary,
        x='min_result',
        y='max_result',
        color='uname',
        symbol='test',  # Use different symbols for different tests
        hover_data=['test', 'count', 'mean_result', 'median_result'],
        size='count',
        labels={
            'min_result': 'Minimum Result (ppb)',
            'max_result': 'Maximum Result (ppb)',
            'uname': 'uname',
            'test': 'Test',
            'count': 'Sample Count',
            'mean_result': 'Mean Result (ppb)',
            'median_result': 'Median Result (ppb)'
        },
        title='Minimum vs Maximum Results by uname and mycotoxin'
    )

    fig_scatter.update_layout(
        legend_title_text='uname',
        xaxis_title="Minimum Result (ppb)",
        yaxis_title="Maximum Result (ppb)",
    )

    if selected_test == 'All' and selected_uname == 'All':
        fig_scatter.update_xaxes(type="log")
        fig_scatter.update_yaxes(type="log")

    st.plotly_chart(fig_scatter, use_container_width=True)

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

    # Create dropdown menus for user selections
    uname_options = ['All'] + sorted(data['uname'].unique().tolist())
    test_options = ['All'] + sorted(data['test'].unique().tolist())

    col1, col2 = st.columns(2)
    with col1:
        selected_uname = st.selectbox("Select UNAME", uname_options)
    with col2:
        selected_test = st.selectbox("Select Mycotoxin", test_options)

    # Filter the data based on user selections
    filtered_data = data.copy()
    if selected_uname != 'All':
        filtered_data = filtered_data[filtered_data['uname'] == selected_uname]
    if selected_test != 'All':
        filtered_data = filtered_data[filtered_data['test'] == selected_test]

    if filtered_data.empty:
        st.error("No data points for the selected filters.")
    else:
        display_kde_plot(filtered_data, selected_uname, selected_test)
        display_histogram(filtered_data, selected_uname, selected_test)
        summary = display_summary_statistics(filtered_data)
        display_summary_visualizations(summary, filtered_data, selected_uname, selected_test)

        # Display raw data
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