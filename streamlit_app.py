import streamlit as st # type: ignore
import pandas as pd # type: ignore
from run import predict_values, add_product_features, predict_continuous_values
from src.PossibleNumber import get_predicted_number
from src.PositionAnalyzer import PatternPositionAnalyzer
from src.DataProcessor import ProductDataProcessor
from src.calculator import PatternProbabilityCalculator
from src.Predict_features import predict_features_from_number
from src.Strategy2 import find_next_numbers_and_features, standardize_values, calculate_feature_percentages , get_last_n_numbers_and_frequent_features
from utils.utils import file_path
import numpy as np # type: ignore
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure the Streamlit page
st.set_page_config(
    page_title="Pattern Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    h1 {
        color: #1E88E5;
    }
    h2 {
        color: #424242;
    }
    .stAlert {
        border-radius: 5px;
    }
    footer {
        visibility: hidden;
    }
    [data-testid=stSidebar] {
        background-color: #1E1E1E;
        color: white;
    }
    [data-testid=stSidebar] [data-testid=stMarkdown] {
        color: white;
    }
    [data-testid=stSidebar] .stRadio label {
        color: white;
        font-size: 16px;
        padding: 10px 0;
    }
    [data-testid=stSidebar] hr {
        margin: 20px 0;
        border-color: #333;
    }
    [data-testid=stSidebar] [data-testid="stRadio"] > label {
        display: none;
    }
    .sidebar-link {
        padding: 10px;
        color: white;
        text-decoration: none;
        display: block;
        transition: background-color 0.3s;
        border-radius: 5px;
    }
    .sidebar-link:hover {
        background-color: #333;
    }
    .sidebar-link.active {
        background-color: #4CAF50;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-size: 16px;
        font-weight: 500;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .stButton>button:active {
        background-color: #3d8b40;
    }
</style>
""", unsafe_allow_html=True)

data_processor = ProductDataProcessor()

# Define your page functions
def page_1():
    # Initialize session state for all features if they don't exist
    if 'number_input' not in st.session_state:
        st.session_state.number_input = ""
    for feature in ['Dozen', 'color', 'series', 'Column', 'parity', 'Group']:
        if feature not in st.session_state:
            st.session_state[feature] = ""

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.title("🎯 Pattern Prediction Dashboard")
    
    st.markdown("---")

    input_col, output_col = st.columns([1, 1])
    
    with input_col:
        st.markdown("### 📝 Data Entry")
        with st.expander("Add New Product Features", expanded=True):
            # Number input with automatic update
            number = st.text_input("Number", value=st.session_state.number_input, 
                                  key="number_input_field").upper()
            
            # Update features immediately when number changes
            if number != st.session_state.number_input:
                st.session_state.number_input = number
                if number.isdigit() and 0 <= int(number) <= 36:
                    features = predict_features_from_number(int(number))
                    for feature, value in features.items():
                        st.session_state[feature] = value
                else:
                    # Clear features if number is invalid
                    for feature in ['Dozen', 'color', 'series', 'Column', 'parity', 'Group']:
                        st.session_state[feature] = ""

            col1, col2 = st.columns(2)
            with col1:
                dozen = st.selectbox("Dozen", 
                                    options=["", "D1", "D2", "D3"],
                                    key="dozen_selector",
                                    index=["", "D1", "D2", "D3"].index(st.session_state.Dozen) if st.session_state.Dozen else 0)
                color = st.selectbox("Color", 
                                    options=["", "RED", "BLACK", "GREEN"],
                                    key="color_selector",
                                    index=["", "RED", "BLACK", "GREEN"].index(st.session_state.color) if st.session_state.color else 0)
                series = st.selectbox("Series", 
                                     options=["", "A", "B", "C"],
                                     key="series_selector",
                                     index=["", "A", "B", "C"].index(st.session_state.series) if st.session_state.series else 0)
            
            with col2:
                column = st.selectbox("Column", 
                                     options=["", "C1", "C2", "C3"],
                                     key="column_selector",
                                     index=["", "C1", "C2", "C3"].index(st.session_state.Column) if st.session_state.Column else 0)
                parity = st.selectbox("Parity", 
                                     options=["", "ODD", "EVEN"],
                                     key="parity_selector",
                                     index=["", "ODD", "EVEN"].index(st.session_state.parity) if st.session_state.parity else 0)
                group = st.selectbox("Group", 
                                    options=["", "G1", "G2"],
                                    key="group_selector",
                                    index=["", "G1", "G2"].index(st.session_state.Group) if st.session_state.Group else 0)

        # Update session state based on manual selections
        st.session_state.dozen = dozen
        st.session_state.color = color
        st.session_state.series = series
        st.session_state.column = column
        st.session_state.parity = parity
        st.session_state.group = group

        optional_features = {
            'Number': number or None,
            'Dozen': dozen or None,
            'Column': column or None,
            'parity': parity or None,
            'color': color or None,
            'series': series or None,
            'Group': group or None
        }

        if st.button("✨ Add Features", key="add_features_button_page1"):
            add_product_features(optional_features)
            st.success("✅ Features added successfully!")

    with output_col:
        st.markdown("### 🔮 Prediction Settings")
        range_col1, range_col2 = st.columns(2)
        with range_col1:
            start_number = st.number_input("Start Number", min_value=0, value=0)
        with range_col2:
            end_number = st.number_input("End Number", min_value=100, value=100, step=100)

        pattern_col1, pattern_col2 = st.columns(2)
        with pattern_col1:
            initial_pattern_length = st.number_input("Initial Pattern Length", min_value=5, value=5)
        with pattern_col2:
            end_pattern_length = st.number_input("End Pattern Length", min_value=0, value=0)

    st.markdown("---")

    # Check if the "Predict Next Values" button has been clicked
    if st.button("🔍 Predict Next Values", key="predict_next_values_button_page1"):
        with st.spinner("Calculating predictions..."):
            predicted_values = predict_values(start_number,initial_pattern_length,end_pattern_length or None, end_number or None)
            st.session_state['predicted_values'] = predicted_values
            st.session_state['predict_next_clicked'] = True  # Set flag for button click

    # Only run this block if the button was clicked
    if st.session_state.get('predict_next_clicked', False):
        st.markdown("### 📊 Prediction Results")
        pred_cols = st.columns(3)
        
        if 'predicted_values' in st.session_state:
                pred_values_list = st.session_state['predicted_values'].split()
                st.success(pred_values_list)                
                if len(pred_values_list) >= 7:
                    predictions = {
                        "Series": pred_values_list[0],
                        "Column": pred_values_list[1] + (pred_values_list[2] if len(pred_values_list) > 7 else ""),
                        "Parity": pred_values_list[3 if len(pred_values_list) > 7 else 2],
                        "Color": pred_values_list[4 if len(pred_values_list) > 7 else 3],
                        "Dozen": pred_values_list[5 if len(pred_values_list) > 7 else 4] + 
                                pred_values_list[6 if len(pred_values_list) > 7 else 5],
                        "Group": pred_values_list[7 if len(pred_values_list) > 7 else 6]
                    }

                    for i, (key, value) in enumerate(predictions.items()):
                        with pred_cols[i % 3]:
                            st.metric(label=key, value=value)

                    if st.button("🎲 Predict the Number", key="number_button_page1"):
                        
                        predicted_number = get_predicted_number(
                            predictions["Series"], 
                            predictions["Column"],
                            predictions["Parity"], 
                            predictions["Color"],
                            predictions["Dozen"],
                        )
                        if predicted_number == "Number not found":
                            st.warning("⚠️ No numbers found matching all the criteria.")
                        elif isinstance(predicted_number, list):
                            if len(predicted_number) == 1:
                                st.success(f"🎯 Predicted Number: {predicted_number[0]}")
                            else:
                                st.success("🎯 Possible Numbers:")
                                cols = st.columns(min(5, len(predicted_number)))
                                for idx, num in enumerate(predicted_number):
                                    with cols[idx % 5]:
                                        st.metric(f"Option {idx+1}", str(num))
                        else:
                            st.success(f"🎯 Predicted Number: {predicted_number}")          
                                        
      

def page_2():
    st.title("📈 Back-testing Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_number = st.number_input("Start Number", min_value=0, value=0)
        initial_pattern_length = st.number_input("Initial Pattern Length", min_value=5, value=5)

    with col2:
        end_number = st.number_input("End Number", min_value=100, value=100, step=100)
        end_pattern_length = st.number_input("End Pattern Length", min_value=0, value=0)

    with col3:
        num_predictions = st.number_input("Number of Predictions", min_value=1, value=1)

    if st.button("🔍 Analyze Accuracy", key="analyze_accuracy_button_page2"):
        with st.spinner("Running back-testing analysis..."):
            _, counts, _, accuracy, overall_accuracy = predict_continuous_values(
                start_number, end_number, num_predictions, initial_pattern_length, end_pattern_length)
            
        st.markdown("### 📊 Accuracy Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Feature Counts")
            features = ["Series", "Column", "Odd/Even", "Color", "Dozen", "Group"]
            for feature, count in zip(features, counts):
                st.metric(label=feature, value=count)
        
        with col2:
            st.markdown("#### Accuracy Percentages")
            for feature, acc in zip(features, accuracy):
                st.metric(label=feature, value=f"{acc:.2f}%")
        
        st.metric("Overall Accuracy", value=f"{overall_accuracy:.2f}%")
        
        
def page_3():
    st.title("📊 Pattern Position Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_number = st.number_input("Start Number", min_value=0, value=0)
        initial_pattern_length = st.number_input("Initial Pattern Length", min_value=5, value=5)

    with col2:
        end_number = st.number_input("End Number", min_value=100, value=100, step=100)
        end_pattern_length = st.number_input("End Pattern Length", min_value=0, value=0)

    with col3:
        max_range = st.number_input("Max Range", min_value=500, value=500, step=100)

    if st.button("📈 Analyze Patterns", key="analyze_patterns_button_page3"):
        with st.spinner("Analyzing pattern positions..."):
            series, _, _, _, _, _ = data_processor.get_product_data(start=start_number, end=end_number)
            
            calculator = PatternProbabilityCalculator(series , initial_patern_length=initial_pattern_length, last_pattern_length=end_pattern_length)
            _, _, _ = calculator.calculate_next_probabilities()
            pattern_positions = calculator.find_pattern_positions()
            
            analyzer = PatternPositionAnalyzer(pattern_positions)
            position_differences = analyzer.calculate_position_differences()
            
            figs = analyzer.plot_difference_barcharts(position_differences, bin_size=10, max_range=max_range)

        if not figs:
            st.warning("⚠️ No patterns with multiple occurrences found. Try increasing the range.")
        else:
            st.success(f"📊 Generated {len(figs)} pattern analysis charts")
            for fig in figs:
                st.plotly_chart(fig, use_container_width=True)
                
                

def page_4():
    st.title("🔍 Predicting Probability")
    
    # Add custom CSS to make the dataframe more professional
    st.markdown("""
    <style>
    .dataframe {
        font-family: Arial, sans-serif;
        font-size: 12px;
        border-collapse: collapse;
        width: 100%;
    }
    .dataframe td {
        border: 1px solid #ddd;
        padding: 8px;
    }
    .dataframe tr:nth-child(even) {
        background-color: #f2f2f2;
    }
    .dataframe tr:hover {
        background-color: #ddd;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        add_number = st.number_input("Add Number", min_value=0, max_value=36, value=0)
    with col2:
        lookback = st.number_input("Lookback", min_value=1, value=100)

    if st.button("📊 Analyze Features", key="analyze_features_button"):
        with st.spinner("Analyzing features..."):
            # Predict features from the number
            predicted_features = predict_features_from_number(add_number)
            
            # Add features to the DataFrame
            data_processor.add_features({'Number': add_number, **predicted_features})
            
            df = pd.read_excel(file_path)
            
            # Find next numbers and features
            last_number, next_numbers_and_features = find_next_numbers_and_features(df, lookback)
            
            st.write(f"## Last Number : {last_number}")
            
            # Standardize values
            next_numbers_df = pd.DataFrame(next_numbers_and_features)
            next_numbers_df = standardize_values(next_numbers_df)
            
             # Calculate probabilities
            feature_percentages = calculate_feature_percentages(next_numbers_df)
            
            # Get the last 5 numbers and their features, and the most frequent feature values
            last_5_numbers, most_frequent_features = get_last_n_numbers_and_frequent_features(df, n=5)
            
            # Create a summary row
            summary_row = {}
            for feature, data in feature_percentages.items():
                events = ", ".join(data['Most Frequent Events'])
                percentage = data['Percentage']
                summary_row[feature] = f"{events} ({percentage:.2f}%)"
            
            # Add the summary row to the dataframe
            summary_df = pd.DataFrame([summary_row])
            next_numbers_df = pd.concat([next_numbers_df, summary_df], ignore_index=True)
            
            # Reorder columns for better readability
            columns_order = ['Next Number', 'Dozen', 'Column', 'parity', 'color', 'series', 'Group']
            next_numbers_df = next_numbers_df.reindex(columns=columns_order)
            
            # Format the 'Next Number' column to remove decimal places
            next_numbers_df['Next Number'] = next_numbers_df['Next Number'].apply(lambda x: f"{x:.0f}" if pd.notnull(x) else x)
            
            # Display next numbers and their features in a styled table
            st.markdown("### 🔮 Next Numbers and Features")
            
            # Style the dataframe
            def style_dataframe(val):
                if pd.isna(val):
                    return ''
                elif isinstance(val, (int, float)):
                    return 'color: black; font-weight: bold;'
                else:
                    return ''

            styled_df = next_numbers_df.style.applymap(style_dataframe)
            
            # Highlight the summary row
            styled_df = styled_df.apply(lambda x: ['background-color: #010a12' for _ in x], axis=1, subset=pd.IndexSlice[-1:, :])
            
            # Format numeric columns
            numeric_columns = next_numbers_df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                styled_df = styled_df.format({col: '{:.0f}'})
            
            # Display the styled dataframe
            st.dataframe(styled_df, height=400, use_container_width=True)

            # Display feature percentages
            with st.expander("📈 Feature Probabilities"):
                st.markdown("### 📈 Feature Probabilities")
                for feature, data in feature_percentages.items():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.markdown(f"**{feature.capitalize()}**")
                    with col2:
                        events = ", ".join(data['Most Frequent Events'])
                        st.markdown(f"Most Frequent: **{events}**")
                    with col3:
                        st.markdown(f"Probability: **{data['Percentage']:.2f}%**")
            
            # Display Last 5 Numbers and Most Frequent Features
            with st.expander("📜 Last 5 Numbers and Most Frequent Features"):
                st.markdown("### Last 5 Numbers")
                for index, row in last_5_numbers.iterrows():
                    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
                    with col1:
                        st.markdown(f"**{row['Number']}**")
                    with col2:
                        st.write(row['Dozen'])
                    with col3:
                        st.write(row['Column'])
                    with col4:
                        st.write(row['parity'])
                    with col5:
                        st.write(row['color'])
                    with col6:
                        st.write(row['series'])
                    with col7:
                        st.write(f"Group: {row['Group']}")
                
                st.markdown("---")
                st.markdown("### Most Frequent Features")
                for feature, data in most_frequent_features.items():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.markdown(f"**{feature.capitalize()}**")
                    with col2:
                        events = ", ".join(data['Most Frequent Events'])
                        st.markdown(f"Most Frequent: **{events}**")
                    with col3:
                        st.markdown(f"Percentage: **{data['Percentage']:.2f}%**")


def page_5():
    st.title("📊 Combined Strategy Dashboard")
    
    # Add custom CSS for better styling
    st.markdown("""
    <style>
    .dataframe {
        font-family: Arial, sans-serif;
        font-size: 12px;
        border-collapse: collapse;
        width: 100%;
    }
    .dataframe td {
        border: 1px solid #ddd;
        padding: 8px;
    }
    .dataframe tr:nth-child(even) {
        background-color: #f2f2f2;
    }
    .dataframe tr:hover {
        background-color: #ddd;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1E88E5;
    }
    </style>
    """, unsafe_allow_html=True)

    # Input parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        add_number = st.number_input("Add Number", min_value=0, max_value=36, value=0)
        start_number = st.number_input("Start Number", min_value=0, value=0)
    with col2:
        lookback = st.number_input("Lookback", min_value=1, value=100)
        initial_pattern_length = st.number_input("Initial Pattern Length", min_value=5, value=5)
    with col3:
        end_number = st.number_input("End Number", min_value=100, value=100, step=100)
        end_pattern_length = st.number_input("End Pattern Length", min_value=0, value=0)

    if st.button("🔄 Analyze All Strategies", key="analyze_all_button"):
        with st.spinner("Analyzing all strategies..."):
            # Strategy 1: Pattern Prediction
            predicted_values_st_1 = predict_values(start_number, initial_pattern_length, end_pattern_length or None, end_number or None)
            
            # Strategy 2: Next Numbers Analysis
            df = pd.read_excel(file_path)
            predicted_features = predict_features_from_number(add_number)
            data_processor.add_features({'Number': add_number, **predicted_features})
            last_number, predicted_values_st_2 = find_next_numbers_and_features(df, lookback)
            
            # Strategy 3: Feature Analysis
            last_5_numbers, most_frequent_features = get_last_n_numbers_and_frequent_features(df, n=5)
            
            # Display Results in Tabs
            tab1, tab2, tab3 = st.tabs(["📈 Pattern Prediction", "🎯 Next Numbers", "📊 Feature Analysis"])
            
            with tab1:
                st.subheader("Pattern Prediction Results")
                if predicted_values_st_1:
                    pred_values_list = predicted_values_st_1.split()
                    if len(pred_values_list) >= 7:
                        predictions = {
                            "Series": pred_values_list[0],
                            "Column": pred_values_list[1] + (pred_values_list[2] if len(pred_values_list) > 7 else ""),
                            "Parity": pred_values_list[3 if len(pred_values_list) > 7 else 2],
                            "Color": pred_values_list[4 if len(pred_values_list) > 7 else 3],
                            "Dozen": pred_values_list[5 if len(pred_values_list) > 7 else 4] + pred_values_list[6 if len(pred_values_list) > 7 else 5],
                            "Group": pred_values_list[7 if len(pred_values_list) > 7 else 6]
                        }
                        
                        cols = st.columns(3)
                        for i, (key, value) in enumerate(predictions.items()):
                            with cols[i % 3]:
                                st.metric(label=key, value=value)
            
            with tab2:
                st.subheader("Next Numbers Analysis")
                if isinstance(predicted_values_st_2, pd.DataFrame):
                    # Standardize values
                    next_numbers_df = standardize_values(predicted_values_st_2)
                    
                    # Calculate probabilities
                    feature_percentages = calculate_feature_percentages(next_numbers_df)
                    
                    # Display the dataframe
                    st.dataframe(next_numbers_df.style.apply(lambda x: ['background-color: #010a12' for _ in x], 
                                                           axis=1, 
                                                           subset=pd.IndexSlice[-1:, :]), 
                               height=400)
                    
                    # Feature probabilities in expander
                    with st.expander("📈 Feature Probabilities", expanded=True):
                        for feature, data in feature_percentages.items():
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.markdown(f"**{feature.capitalize()}**")
                            with col2:
                                events = ", ".join(data['Most Frequent Events'])
                                st.markdown(f"Most Frequent: **{events}**")
                            with col3:
                                st.markdown(f"Probability: **{data['Percentage']:.2f}%**")
            
            with tab3:
                st.subheader("Feature Analysis")
                
                # Last 5 Numbers
                st.markdown("### 📜 Last 5 Numbers")
                for index, row in last_5_numbers.iterrows():
                    cols = st.columns(7)
                    with cols[0]:
                        st.markdown(f"**{row['Number']}**")
                    with cols[1]:
                        st.write(row['Dozen'])
                    with cols[2]:
                        st.write(row['Column'])
                    with cols[3]:
                        st.write(row['parity'])
                    with cols[4]:
                        st.write(row['color'])
                    with cols[5]:
                        st.write(row['series'])
                    with cols[6]:
                        st.write(f"Group: {row['Group']}")
                
                # Most Frequent Features
                st.markdown("### 📊 Most Frequent Features")
                for feature, data in most_frequent_features.items():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.markdown(f"**{feature.capitalize()}**")
                    with col2:
                        events = ", ".join(data['Most Frequent Events'])
                        st.markdown(f"Most Frequent: **{events}**")
                    with col3:
                        st.markdown(f"Percentage: **{data['Percentage']:.2f}%**")
                        
            # Summary Metrics at the bottom
            st.markdown("---")
            st.subheader("📈 Combined Insights")
            summary_cols = st.columns(3)
            
            # Pattern Match Rate
            with summary_cols[0]:
                if predicted_values_st_1:
                    st.metric("Pattern Match Rate", 
                             f"{len(pred_values_list)/7:.0%}",
                             "Based on Pattern Prediction")
            
            # Feature Accuracy
            with summary_cols[1]:
                if feature_percentages:
                    avg_accuracy = np.mean([data['Percentage'] for data in feature_percentages.values()])
                    st.metric("Average Feature Accuracy", 
                             f"{avg_accuracy:.1f}%",
                             "Based on Next Numbers")
            
            # Most Consistent Feature
            with summary_cols[2]:
                if most_frequent_features:
                    max_feature = max(most_frequent_features.items(), 
                                    key=lambda x: x[1]['Percentage'])
                    st.metric("Most Consistent Feature",
                             f"{max_feature[0]}",
                             f"{max_feature[1]['Percentage']:.1f}% consistency")

        
# Define pages with icons and descriptions
pages = {
    "pattern_prediction": {
        "icon": "🎯",
        "name": "Pattern Prediction",
        "function": page_1,
        "description": "Predict patterns and analyze data"
    },
    "backtesting": {
        "icon": "📈",
        "name": "Back-testing Analysis",
        "function": page_2,
        "description": "Verify prediction accuracy"
    },
    "position_analysis": {
        "icon": "📊",
        "name": "Position Analysis",
        "function": page_3,
        "description": "Analyze pattern positions"
    },
    "probability_prediction": {
        "icon": "🔍",
        "name": "Predicting Probability",
        "function": page_4,
        "description": "Analyze features and predict probabilities"
    },
    "Dashboard": {
        "icon": "🔮",
        "name": "Dashboard for all strategies",
        "function": page_5,
        "description": "Dashboard for all strategies"
    }
}

# Sidebar navigation
st.sidebar.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <h1 style="color: white;">🔍 Navigation</h1>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Create radio buttons for navigation
selected_page = st.sidebar.radio(
    "",
    list(pages.keys()),
    format_func=lambda x: f"{pages[x]['icon']} {pages[x]['name']}"
)

# Display description of selected page
st.sidebar.markdown(f"""
<div style="padding: 10px; background-color: #333; border-radius: 5px; margin-top: 20px;">
    <p style="color: #CCC; margin: 0;">{pages[selected_page]['description']}</p>
</div>
""", unsafe_allow_html=True)

# Add some space
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Add a footer to the sidebar
st.sidebar.markdown("""
<div style="position: fixed; bottom: 0; left: 0; padding: 20px; width: 100%; background-color: #1E1E1E;">
    <p style="color: #666; text-align: center; font-size: 12px;">© 2024 Pattern Analysis Tool</p>
</div>
""", unsafe_allow_html=True)

# Render the selected page
pages[selected_page]['function']()
