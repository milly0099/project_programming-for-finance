import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import numpy as np

# Import theme pages
from themes.zombie import zombie_theme
from themes.futuristic import futuristic_theme
from themes.got import got_theme
from themes.gaming import gaming_theme
from utils.data_loader import load_example_data
from utils.audio import init_audio_state, play_theme_sound
from assets.theme_descriptions import theme_descriptions, app_description

# Function to preprocess dataframe for compatibility
def preprocess_dataframe(df):
    """Preprocess dataframe to ensure compatibility with all themes"""
    if df is None:
        return None
        
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Check for MultiIndex columns and handle them
    if isinstance(processed_df.columns, pd.MultiIndex):
        # For each column in the MultiIndex, create a new column with a string name
        for col in processed_df.columns:
            if isinstance(col, tuple):
                # Create a new flat column name
                new_col_name = "_".join([str(c) for c in col if c != ""])
                # Only apply if the new column doesn't already exist
                if new_col_name not in processed_df.columns:
                    processed_df[new_col_name] = processed_df[col]
        
    # Convert datetime columns to string to avoid Arrow conversion errors
    for col in processed_df.columns:
        if pd.api.types.is_datetime64_any_dtype(processed_df[col]):
            processed_df[col] = processed_df[col].astype(str)
            
    return processed_df

# Page configuration
st.set_page_config(
    page_title="Financial ML Multi-Theme App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state variables if they don't exist
if 'dataframe' not in st.session_state:
    st.session_state.dataframe = None
if 'last_theme' not in st.session_state:
    st.session_state.last_theme = None

def main():
    # Initialize audio state
    init_audio_state()
    
    # Add sound toggle control in sidebar
    st.sidebar.title("Settings")
    st.session_state.sound_enabled = st.sidebar.checkbox("Enable Sound Effects", value=st.session_state.sound_enabled)
    
    # Sidebar for navigation and data loading
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Welcome", "Zombie Theme - Linear Regression", "Futuristic Theme - Logistic Regression", 
         "Game of Thrones Theme - K-Means Clustering", "Gaming Theme - Random Forest"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.title("Data Options")
    
    # Data loading options
    data_option = st.sidebar.radio(
        "Choose data source:",
        ["Upload CSV file", "Use Yahoo Finance API", "Use Example Data"]
    )
    
    if data_option == "Upload CSV file":
        uploaded_file = st.sidebar.file_uploader("Upload your financial dataset (CSV)", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                # Process the dataframe immediately after loading
                st.session_state.dataframe = preprocess_dataframe(df)
                st.sidebar.success("Data loaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error loading file: {e}")
    
    elif data_option == "Use Yahoo Finance API":
        st.sidebar.subheader("Yahoo Finance Data")
        ticker = st.sidebar.text_input("Enter stock ticker (e.g., AAPL, MSFT, GOOGL)", "AAPL")
        period = st.sidebar.selectbox("Select period", 
                                    ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
                                    index=3)
        
        if st.sidebar.button("Fetch Data"):
            with st.sidebar:
                with st.spinner("Fetching data from Yahoo Finance..."):
                    try:
                        stock_data = yf.download(ticker, period=period)
                        if isinstance(stock_data, pd.DataFrame) and len(stock_data) > 0:
                            stock_data.reset_index(inplace=True)
                            # Process the dataframe immediately after loading
                            st.session_state.dataframe = preprocess_dataframe(stock_data)
                            st.success(f"Successfully loaded data for {ticker}")
                        else:
                            st.error(f"No data found for ticker: {ticker}")
                    except Exception as e:
                        st.error(f"Error fetching data: {e}")
    
    elif data_option == "Use Example Data":
        example_option = st.sidebar.selectbox(
            "Choose an example dataset",
            ["Stock Price Data", "Financial Metrics", "Trading Signals"]
        )
        
        if st.sidebar.button("Load Example Data"):
            example_df = load_example_data(example_option)
            # Process the dataframe immediately after loading
            st.session_state.dataframe = preprocess_dataframe(example_df)
            st.sidebar.success(f"Loaded example {example_option} dataset")
    
    # Display appropriate page based on selection
    if page == "Welcome":
        play_theme_sound("welcome")
        welcome_page()
    elif page == "Zombie Theme - Linear Regression":
        play_theme_sound("zombie")
        if st.session_state.dataframe is not None:
            # Preprocess dataframe before passing to theme
            processed_df = preprocess_dataframe(st.session_state.dataframe)
            zombie_theme(processed_df)
        else:
            st.error("Please load data first using the sidebar options.")
    elif page == "Futuristic Theme - Logistic Regression":
        play_theme_sound("futuristic")
        if st.session_state.dataframe is not None:
            # Preprocess dataframe before passing to theme
            processed_df = preprocess_dataframe(st.session_state.dataframe)
            futuristic_theme(processed_df)
        else:
            st.error("Please load data first using the sidebar options.")
    elif page == "Game of Thrones Theme - K-Means Clustering":
        play_theme_sound("got")
        if st.session_state.dataframe is not None:
            # Preprocess dataframe before passing to theme
            processed_df = preprocess_dataframe(st.session_state.dataframe)
            got_theme(processed_df)
        else:
            st.error("Please load data first using the sidebar options.")
    elif page == "Gaming Theme - Random Forest":
        play_theme_sound("gaming")
        if st.session_state.dataframe is not None:
            # Preprocess dataframe before passing to theme
            processed_df = preprocess_dataframe(st.session_state.dataframe)
            gaming_theme(processed_df)
        else:
            st.error("Please load data first using the sidebar options.")
    
    # Save the last theme visited
    st.session_state.last_theme = page

def welcome_page():
    st.title("🌟 Multi-Themed Financial Machine Learning Application 🌟")
    
    # Main app description
    st.markdown(app_description)
    
    # Display financial data visualization
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://pixabay.com/get/g27de72676ab664597476d6ec7e1da0a9b7167a8dd8ecd26e3b197370a26bd8b4e51f02ebca790e5b4398e51bdb9a5f4452049c3862fa7aa47e33102add2653cc_1280.jpg", 
                caption="Financial Data Visualization")
    with col2:
        st.image("https://pixabay.com/get/g74f47724bae7fd341287c5cbe858bbf3202e07bc3a6dd7a41f31d3dad1c6ad9615aadf26dd8eb76f06bd373a04188177512ca6553648250f5298fc443870dd76_1280.jpg", 
                caption="Stock Market Analysis")
    
    # Instructions section
    st.header("📝 How to Use This Application")
    st.markdown("""
    1. **Load Data**: Use the sidebar to upload your own CSV file, fetch data from Yahoo Finance, or use our example datasets.
    2. **Explore Themes**: Navigate between four uniquely themed sections, each implementing a different machine learning algorithm.
    3. **Analyze & Learn**: Adjust parameters, visualize results, and gain insights from the financial data analysis.
    """)
    
    # Theme previews section
    st.header("🎭 Theme Previews")
    
    # Create columns for the theme cards
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    
    # Theme cards with descriptions
    with col1:
        st.subheader("🧟 Zombie Theme")
        st.markdown(theme_descriptions["zombie"])
        st.markdown("**Model**: Linear Regression")
        if st.button("Explore Zombie Theme"):
            if st.session_state.dataframe is not None:
                st.session_state.last_theme = "Zombie Theme - Linear Regression"
                st.rerun()
            else:
                st.error("Please load data first using the sidebar options.")
    
    with col2:
        st.subheader("🚀 Futuristic Theme")
        st.markdown(theme_descriptions["futuristic"])
        st.markdown("**Model**: Logistic Regression")
        if st.button("Explore Futuristic Theme"):
            if st.session_state.dataframe is not None:
                st.session_state.last_theme = "Futuristic Theme - Logistic Regression"
                st.rerun()
            else:
                st.error("Please load data first using the sidebar options.")
    
    with col3:
        st.subheader("⚔️ Game of Thrones Theme")
        st.markdown(theme_descriptions["got"])
        st.markdown("**Model**: K-Means Clustering")
        if st.button("Explore Game of Thrones Theme"):
            if st.session_state.dataframe is not None:
                st.session_state.last_theme = "Game of Thrones Theme - K-Means Clustering"
                st.rerun()
            else:
                st.error("Please load data first using the sidebar options.")
    
    with col4:
        st.subheader("🎮 Gaming Theme")
        st.markdown(theme_descriptions["gaming"])
        st.markdown("**Model**: Random Forest")
        if st.button("Explore Gaming Theme"):
            if st.session_state.dataframe is not None:
                st.session_state.last_theme = "Gaming Theme - Random Forest"
                st.rerun()
            else:
                st.error("Please load data first using the sidebar options.")
    
    # Data preview section
    st.header("📊 Data Preview")
    if st.session_state.dataframe is not None:
        st.dataframe(st.session_state.dataframe.head())
    else:
        st.info("No data loaded yet. Please use the sidebar to load data.")

if __name__ == "__main__":
    main()
