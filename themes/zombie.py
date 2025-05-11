import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from utils.ml_models import train_linear_regression
from utils.visualization import create_zombie_theme_chart
from utils.audio import autoplay_audio
import time
import os

def zombie_theme(data):
    # Set the zombie theme styling
    st.markdown("""
    <style>
    .zombie-header {
        color: #9a0000;
        text-shadow: 2px 2px 4px #000000;
        font-family: "Times New Roman", Times, serif;
    }
    .zombie-text {
        color: #600000;
        font-family: "Courier New", Courier, monospace;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with zombie theme
    st.markdown('<h1 class="zombie-header">🧟‍♂️ ZOMBIE FINANCIAL ANALYSIS: LINEAR REGRESSION 🧟‍♀️</h1>', unsafe_allow_html=True)
    st.markdown('<p class="zombie-text">Welcome to the apocalyptic world of financial predictions... where only the most accurate models survive.</p>', unsafe_allow_html=True)
    
    # Display themed image
    st.image("https://pixabay.com/get/g619619e020c795b122807106a5609fe6a69163a5ed69c1f9b72835a761d2306d3e304f6d045ccc1e84408256688541faff1e0ae693566e863a1ed0b0a85bbf9c_1280.jpg", 
            caption="Surviving the Financial Apocalypse")
    
    # Data overview section
    st.markdown('<h2 class="zombie-header">💀 The Financial Wasteland: Your Data 💀</h2>', unsafe_allow_html=True)
    if st.checkbox("Show data overview", value=True):
        st.dataframe(data.head())
        
        # Show basic statistics
        st.markdown('<h3 class="zombie-header">Basic Statistics: The Vital Signs</h3>', unsafe_allow_html=True)
        st.dataframe(data.describe())
    
    # Feature selection for linear regression
    st.markdown('<h2 class="zombie-header">🧠 Select Variables for Brain Analysis 🧠</h2>', unsafe_allow_html=True)
    
    # Get numerical columns only
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if len(numerical_columns) < 2:
        st.error("Not enough numerical columns for regression analysis. Please load a different dataset.")
        return
    
    # Feature selection
    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox(
            "Select target variable (what you want to predict)",
            numerical_columns,
            index=0
        )
    
    with col2:
        available_features = [col for col in numerical_columns if col != target_col]
        selected_features = st.multiselect(
            "Select feature variables (predictors)",
            available_features,
            default=available_features[:1]  # Select first feature by default
        )
    
    if not selected_features:
        st.error("Please select at least one feature for prediction.")
        return
    
    # Model parameters
    st.markdown('<h2 class="zombie-header">🔪 Survival Settings: Model Parameters 🔪</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test set size (%)", 10, 50, 20, 5) / 100
    with col2:
        random_state = st.slider("Random seed (for reproducibility)", 0, 100, 42)
    
    # Handle missing values
    if data[selected_features + [target_col]].isna().sum().sum() > 0:
        st.warning("Your data contains missing values. They need to be dealt with for survival!")
        missing_strategy = st.radio(
            "How to handle missing values?",
            ["Drop rows with missing values", "Fill with mean", "Fill with median", "Fill with zero"],
            index=0
        )
        
        # Apply missing value strategy
        if missing_strategy == "Drop rows with missing values":
            clean_data = data.dropna(subset=selected_features + [target_col])
            if len(clean_data) == 0:
                st.error("No data left after dropping missing values. Try another strategy.")
                return
            st.info(f"Rows remaining after dropping missing values: {len(clean_data)} (from {len(data)})")
        else:
            clean_data = data.copy()
            for col in selected_features + [target_col]:
                if clean_data[col].isna().sum() > 0:
                    if missing_strategy == "Fill with mean":
                        clean_data[col] = clean_data[col].fillna(clean_data[col].mean())
                    elif missing_strategy == "Fill with median":
                        clean_data[col] = clean_data[col].fillna(clean_data[col].median())
                    elif missing_strategy == "Fill with zero":
                        clean_data[col] = clean_data[col].fillna(0)
    else:
        clean_data = data.copy()
    
    # Train-test split explanation
    st.markdown('<p class="zombie-text">In the wasteland, we need to test our survival strategy on one group before applying it to everyone. This is why we split our data into training and testing sets.</p>', unsafe_allow_html=True)
    
    # Run linear regression model when button is clicked
    if st.button("🧟‍♂️ UNLEASH THE MODEL 🧟‍♀️", help="Run linear regression analysis"):
        # Dramatic effect for zombie theme
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(101):
            progress_bar.progress(i)
            if i < 30:
                status_text.markdown('<p class="zombie-text">Raising the dead data from the graves...</p>', unsafe_allow_html=True)
            elif i < 60:
                status_text.markdown('<p class="zombie-text">Feeding the model with brains...</p>', unsafe_allow_html=True)
            elif i < 90:
                status_text.markdown('<p class="zombie-text">Zombies calculating regression coefficients...</p>', unsafe_allow_html=True)
            else:
                status_text.markdown('<p class="zombie-text">Finalizing the apocalyptic predictions...</p>', unsafe_allow_html=True)
            time.sleep(0.02)
        
        status_text.empty()
        
        # Use the helper function to train the model
        X = clean_data[selected_features]
        y = clean_data[target_col]
        
        try:
            model_results = train_linear_regression(X, y, test_size=test_size, random_state=random_state)
            
            # Play success sound if enabled
            if st.session_state.sound_enabled and os.path.exists("assets/audio/model_success.mp3"):
                autoplay_audio("assets/audio/model_success.mp3")
            
            # Display results
            st.markdown('<h2 class="zombie-header">🪦 Prophecy Results: The Forbidden Knowledge 🪦</h2>', unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score (Training)", f"{model_results['r2_train']:.4f}")
            with col2:
                st.metric("R² Score (Testing)", f"{model_results['r2_test']:.4f}")
            with col3:
                st.metric("RMSE (Testing)", f"{model_results['rmse_test']:.4f}")
            
            # Coefficients
            st.markdown('<h3 class="zombie-header">Model Coefficients: The Dark Formulas</h3>', unsafe_allow_html=True)
            coeff_df = pd.DataFrame({
                'Feature': selected_features,
                'Coefficient': model_results['coefficients']
            })
            coeff_df['Absolute Value'] = np.abs(coeff_df['Coefficient'])
            coeff_df = coeff_df.sort_values('Absolute Value', ascending=False)
            
            # Coefficients visualization
            fig = px.bar(
                coeff_df,
                x='Feature',
                y='Coefficient',
                title='Feature Importance (Coefficients)',
                color='Coefficient',
                color_continuous_scale=['darkred', 'black', 'darkred']
            )
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0.8)',
                plot_bgcolor='rgba(0,0,0,0.8)',
                font=dict(color='#bb0000')
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Actual vs Predicted values plot
            st.markdown('<h3 class="zombie-header">Reality vs. Prophecy: The Comparison</h3>', unsafe_allow_html=True)
            
            # Create the scatter plot
            zombie_chart = create_zombie_theme_chart(
                model_results['y_test'],
                model_results['y_pred'],
                target_col
            )
            st.plotly_chart(zombie_chart, use_container_width=True)
            
            # Model equation
            st.markdown('<h3 class="zombie-header">The Apocalyptic Formula</h3>', unsafe_allow_html=True)
            
            equation = f"{target_col} = {model_results['intercept']:.4f}"
            for feat, coef in zip(selected_features, model_results['coefficients']):
                sign = "+" if coef >= 0 else ""
                equation += f" {sign} {coef:.4f} × {feat}"
            
            st.markdown(f'<p class="zombie-text">{equation}</p>', unsafe_allow_html=True)
            
            # Prediction tool
            st.markdown('<h3 class="zombie-header">Predict Your Fate: The Oracle</h3>', unsafe_allow_html=True)
            st.markdown('<p class="zombie-text">Enter values to predict the future...</p>', unsafe_allow_html=True)
            
            # Create input fields for each feature
            col_count = min(3, len(selected_features))
            cols = st.columns(col_count)
            
            input_values = {}
            for i, feature in enumerate(selected_features):
                col_idx = i % col_count
                with cols[col_idx]:
                    # Use the mean as default value
                    default_val = float(clean_data[feature].mean())
                    input_values[feature] = st.number_input(
                        f"{feature}",
                        value=default_val,
                        format="%.4f"
                    )
            
            # Make prediction
            if st.button("🔮 Predict My Dark Future 🔮"):
                input_df = pd.DataFrame([input_values])
                prediction = model_results['model'].predict(input_df)[0]
                
                # Display prediction with eerie styling
                st.markdown(f"""
                <div style="background-color: rgba(0,0,0,0.7); padding: 20px; border: 1px solid #bb0000; border-radius: 5px;">
                    <h3 style="color: #bb0000; text-align: center;">The spirits have spoken...</h3>
                    <p style="color: #ffffff; text-align: center; font-size: 24px;">
                        Predicted {target_col}: <span style="color: #ff0000; font-weight: bold;">{prediction:.4f}</span>
                    </p>
                    <p style="color: #bb0000; text-align: center; font-style: italic;">
                        Use this forbidden knowledge wisely...
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error during model training and evaluation: {e}")
