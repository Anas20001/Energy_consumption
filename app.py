from final_refactored_app import DataLoader, Utils, Analyzer, Visualizer  
import streamlit as st
import base64

def run_logic(df, unit, consumption_col, uploaded_file):
    st.write("## Data Sample (Top 5 rows)")
    st.table(df.head())
    
    loaded_df = df.reset_index().to_csv(index=False)
    st.download_button(
        label="Download cleaned data",
        data=loaded_df,
        file_name=f"{uploaded_file.name.split('.')[0]}_cleaned.csv",
        
    )

    st.write("## General Consumption Information")
    st.table(df['consumption'].describe())

    st.write("## Statistical Analysis")
    analyzer = Analyzer(df, unit,consumption_col, uploaded_file.name)
    combined_df = analyzer.generate_combined_info_table()

    cols_to_convert = ['Value', 'Mean', 'Median', 'Standard Deviation']
    for col in cols_to_convert:
        combined_df[col] = combined_df[col].astype(str)
    st.table(combined_df.replace('[]', ''))

    # Generate download link for the table
    csv = combined_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{uploaded_file.name}_combined_info_in_{unit}.csv" style="font-size:20px; color:red; text-decoration:underline;">Download combined info</a>'
    st.markdown(href, unsafe_allow_html=True)

def wrangling_uploaded_file(uploaded_file, unit, preview_lines=10, column_names=None, skip_rows=None, 
                            data_loader=None):

    # Load Data
    columns = st.text_input("Enter the column indices you want to import, separated by commas (e.g. 0,2,4):")
    usecols = [int(x.strip()) for x in columns.split(',')] if columns else None
    df = None
    consumption_col = None
    if usecols and skip_rows is not None and unit:
        df, consumption_col = data_loader.load_data(column_names, skip_rows, usecols)

    return df, consumption_col
    
def display_preview_data(uploaded_file, unit, data_loader, lines_to_preview=10):
    df_preview = data_loader.preview_data(lines_to_preview)
    st.write(f"## Preview of uploaded data (Top {lines_to_preview} lines)")
    st.table(df_preview.head(lines_to_preview))

def show_plot_options():
    order = st.sidebar.radio("Order data by:", ["Ascending", "Descending"])
    plot_type = st.sidebar.radio("Choose the analysis type:", ["None", "Daily", "Weekly", "Monthly", "Average Daily Load Profile"])
    plausibility_check = st.sidebar.radio("Choose a plausibility Check:", ['None', 'Outliers Detection', 'Weekday vs Weekend', 'Histogram of data points'])
    heatmap = st.sidebar.radio("Choose a heatmap:", ['None', 'Heatmap'])
    return order, plot_type, plausibility_check, heatmap

def main():
    st.title("Energy Consumption Analysis")
    st.sidebar.title("Options")

    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    st.write("Welcome to the Energy Consumption Analysis App!")
    st.write("Please upload a CSV file or use the preloaded LP and select an analysis type to get started.")
    st.image("Landing_page.png")
    df = None
    consumption_col = None
    
    if uploaded_file:
        
        preview_lines = st.sidebar.slider("Select number of lines to preview:", 5, 25, 10)
        column_names = st.sidebar.text_input("Column names (comma-separated)")
        skip_rows = st.sidebar.number_input("Number of rows to skip", step=1)
        unit = st.sidebar.radio("Results unit", ["kWh", "kW"])
        
        data_loader = DataLoader(uploaded_file, unit)
        
        display_preview_data(uploaded_file, unit, data_loader, preview_lines)
        
        if column_names and skip_rows and unit:
            df, consumption_col = wrangling_uploaded_file(uploaded_file, unit, 
                                                      preview_lines, column_names,
                                                      skip_rows,data_loader)
        
    if df is not None:
        run_logic(df, unit, consumption_col, uploaded_file)
        
        order, plot_type, plausibility_check, heatmap = show_plot_options()
        
        visualizer = Visualizer(df, unit, order)
        utils = Utils(df, unit, order)
        

        utils.handle_plotting_and_analysis(visualizer, plot_type, plausibility_check, heatmap)
        
        if heatmap != 'None':
            
            utils.handle_heatmap()

# Execute main function
if __name__ == "__main__":
    main()

                
                            