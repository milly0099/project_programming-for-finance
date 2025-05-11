import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from utils.ml_models import train_logistic_regression
from utils.visualization import create_futuristic_theme_chart
from utils.audio import autoplay_audio
import time
import os

def futuristic_theme(data):
    # Set the futuristic theme styling
    st.markdown("""
    <style>
    .futuristic-header {
        color: #00ccff;
        text-shadow: 0 0 10px #00ccff, 0 0 20px #00ccff;
        font-family: 'Arial', sans-serif;
    }
    .futuristic-text {
        color: #0099cc;
        font-family: 'Arial', sans-serif;
    }
    .neon-box {
        border: 1px solid #00ccff;
        border-radius: 5px;
        padding: 10px;
        background-color: rgba(0, 12, 24, 0.7);
        box-shadow: 0 0 10px #00ccff;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with futuristic theme
    st.markdown('<h1 class="futuristic-header">🚀 QUANTUM FINANCIAL PREDICTOR: LOGISTIC REGRESSION 🛰️</h1>', unsafe_allow_html=True)
    st.markdown('<p class="futuristic-text">Welcome to the neural network of tomorrow. Analyze binary outcomes with advanced algorithmic precision.</p>', unsafe_allow_html=True)
    
    # Display futuristic themed image
    st.image("https://pixabay.com/get/gcd541a68eae29fd1cdd7a2dabac6cdac7b9554309bcb3f5327fa76584e71a2d9fe44a5d22e47ee291b1face7afbdad756e1606bbb0115296ae6f9fdb2975b1ff_1280.jpg", 
             caption="Neural Network: Advanced Financial Analysis")
    
    # Data overview section
    st.markdown('<h2 class="futuristic-header">📊 DATASTREAM ANALYSIS 📊</h2>', unsafe_allow_html=True)
    if st.checkbox("Display Data Matrix", value=True):
        st.dataframe(data.head())
        
        # Show basic statistics
        st.markdown('<h3 class="futuristic-header">Quantum Statistics</h3>', unsafe_allow_html=True)
        st.dataframe(data.describe())
    
    # Prepare for logistic regression - need to create a binary target
    st.markdown('<h2 class="futuristic-header">🧬 QUANTUM BINARY CLASSIFICATION 🧬</h2>', unsafe_allow_html=True)
    
    # Get numerical columns only
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if len(numerical_columns) < 2:
        st.error("Insufficient numerical features detected. Please upload a dataset with more numerical features.")
        return
    
    # Choose column for binary target creation
    target_base_col = st.selectbox(
        "Select column to transform into binary target",
        numerical_columns,
        index=0
    )
    
    # Create binary target based on threshold
    threshold_method = st.radio(
        "Select threshold method for binary classification",
        ["Above/Below Mean", "Above/Below Median", "Custom Threshold", "Use Existing Binary Column"],
        index=0
    )
    
    binary_col_name = None
    
    if threshold_method == "Use Existing Binary Column":
        # Find columns with only 0s and 1s
        binary_columns = []
        for col in data.columns:
            unique_vals = data[col].dropna().unique()
            if set(unique_vals).issubset({0, 1}) and len(unique_vals) == 2:
                binary_columns.append(col)
        
        if not binary_columns:
            st.warning("No binary columns (with only 0s and 1s) found in the dataset.")
            st.info("Creating a binary target from a numerical column instead.")
            threshold_method = "Above/Below Mean"
        else:
            binary_col_name = st.selectbox("Select existing binary column", binary_columns)
    
    # Display the threshold value and create binary target
    if threshold_method == "Above/Below Mean":
        threshold = data[target_base_col].mean()
        threshold_display = f"Mean: {threshold:.4f}"
    elif threshold_method == "Above/Below Median":
        threshold = data[target_base_col].median()
        threshold_display = f"Median: {threshold:.4f}"
    elif threshold_method == "Custom Threshold":
        min_val = float(data[target_base_col].min())
        max_val = float(data[target_base_col].max())
        threshold = st.slider(
            f"Set threshold for {target_base_col}",
            min_val, max_val, 
            (min_val + max_val) / 2,  # Default to midpoint
            format="%.4f"
        )
        threshold_display = f"Custom: {threshold:.4f}"
    
    # Create and display binary target
    if binary_col_name is None:  # If not using an existing binary column
        binary_col_name = f"{target_base_col}_binary"
        data[binary_col_name] = (data[target_base_col] > threshold).astype(int)
        
        # Show threshold and class distribution
        col1, col2 = st.columns(2)
        with col1:
            if threshold_method != "Use Existing Binary Column":
                st.markdown(f"""
                <div class="neon-box">
                    <h4 class="futuristic-header">Threshold Value</h4>
                    <p class="futuristic-text">{threshold_display}</p>
                    <p class="futuristic-text">Class 1: {target_base_col} > {threshold:.4f}</p>
                    <p class="futuristic-text">Class 0: {target_base_col} ≤ {threshold:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            class_counts = data[binary_col_name].value_counts()
            fig = px.pie(
                values=class_counts.values, 
                names=class_counts.index.map({0: "Class 0", 1: "Class 1"}),
                title="Binary Target Distribution",
                color_discrete_sequence=["#00ccff", "#ff00cc"]
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0.8)",
                plot_bgcolor="rgba(0,0,0,0.8)",
                font=dict(color="#00ccff")
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Feature selection for logistic regression
    st.markdown('<h2 class="futuristic-header">🔬 SELECT PREDICTIVE VARIABLES 🔬</h2>', unsafe_allow_html=True)
    
    # Remove the binary target and its source from potential features if we created it
    available_features = [col for col in numerical_columns if col != binary_col_name]
    if threshold_method != "Use Existing Binary Column":
        available_features = [col for col in available_features if col != target_base_col]
    
    if not available_features:
        st.error("No features available for prediction after removing target column.")
        return
    
    selected_features = st.multiselect(
        "Select feature variables for the logistic regression model",
        available_features,
        default=available_features[:min(3, len(available_features))]
    )
    
    if not selected_features:
        st.error("Please select at least one feature for the model.")
        return
    
    # Model parameters
    st.markdown('<h2 class="futuristic-header">⚙️ ALGORITHM PARAMETERS ⚙️</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test data ratio", 0.1, 0.5, 0.2, 0.05)
    with col2:
        random_state = st.slider("Random seed", 0, 100, 42)
    with col3:
        standardize = st.checkbox("Standardize features", value=True)
    
    # Additional model parameters
    col1, col2 = st.columns(2)
    with col1:
        c_value = st.slider(
            "Regularization strength (C)",
            0.01, 10.0, 1.0, 0.01,
            help="Lower values indicate stronger regularization"
        )
    with col2:
        max_iter = st.slider(
            "Maximum iterations",
            100, 1000, 200, 50,
            help="Maximum number of iterations for solver convergence"
        )
    
    # Handle missing values
    clean_data = data.copy()
    
    try:
        # Try to access the columns - this handles both regular and MultiIndex columns
        check_cols = selected_features.copy()
        if binary_col_name in data.columns:
            check_cols.append(binary_col_name)
        
        missing_values = False
        try:
            missing_values = data[check_cols].isna().sum().sum() > 0
        except:
            # If we can't check for missing values, assume none
            pass
            
        if missing_values:
            st.warning("Your data contains missing values which must be processed for optimal model performance.")
            missing_strategy = st.radio(
                "Missing value strategy",
                ["Drop rows with missing values", "Fill with mean", "Fill with median", "Fill with zero"],
                index=0
            )
            
            # Apply missing value strategy
            if missing_strategy == "Drop rows with missing values":
                clean_data = data.dropna(subset=check_cols)
                if len(clean_data) == 0:
                    st.error("No data remains after dropping missing values. Try another strategy.")
                    return
                st.info(f"Remaining observations: {len(clean_data)} (removed {len(data) - len(clean_data)} rows with missing values)")
            else:
                # Apply other missing value strategies
                for col in selected_features:
                    try:
                        if clean_data[col].isna().sum() > 0:
                            if missing_strategy == "Fill with mean":
                                clean_data[col] = clean_data[col].fillna(clean_data[col].mean())
                            elif missing_strategy == "Fill with median":
                                clean_data[col] = clean_data[col].fillna(clean_data[col].median())
                            elif missing_strategy == "Fill with zero":
                                clean_data[col] = clean_data[col].fillna(0)
                    except:
                        st.warning(f"Could not process missing values for column {col}")
                        
    except KeyError as e:
        st.error(f"Column access error: {e}. This may be due to MultiIndex columns not being properly handled.")
        st.info("Please select a different dataset or ensure your data has the expected format.")
        return
    
    # Train model when button is clicked
    if st.button("🔮 INITIALIZE QUANTUM PREDICTION 🔮", help="Train logistic regression model"):
        # Futuristic loading effect
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(101):
            progress_bar.progress(i)
            if i < 30:
                status_text.markdown('<p class="futuristic-text">Initializing quantum calculation matrices...</p>', unsafe_allow_html=True)
            elif i < 60:
                status_text.markdown('<p class="futuristic-text">Optimizing hyperplane coefficients...</p>', unsafe_allow_html=True)
            elif i < 90:
                status_text.markdown('<p class="futuristic-text">Calibrating prediction algorithms...</p>', unsafe_allow_html=True)
            else:
                status_text.markdown('<p class="futuristic-text">Finalizing neural synchronization...</p>', unsafe_allow_html=True)
            time.sleep(0.02)
        
        status_text.empty()
        
        # Train the model using helper function
        X = clean_data[selected_features]
        y = clean_data[binary_col_name]
        
        try:
            model_results = train_logistic_regression(
                X, y, 
                test_size=test_size, 
                random_state=random_state,
                standardize=standardize,
                C=c_value,
                max_iter=max_iter
            )
            
            # Play success sound if enabled
            if st.session_state.sound_enabled and os.path.exists("assets/audio/model_success.mp3"):
                autoplay_audio("assets/audio/model_success.mp3")
            
            # Display results
            st.markdown('<h2 class="futuristic-header">🌌 NEURAL NETWORK RESULTS 🌌</h2>', unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{model_results['accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{model_results['precision']:.4f}")
            with col3:
                st.metric("Recall", f"{model_results['recall']:.4f}")
            with col4:
                st.metric("F1 Score", f"{model_results['f1']:.4f}")
            
            # Confusion Matrix
            st.markdown('<h3 class="futuristic-header">Quantum State Matrix</h3>', unsafe_allow_html=True)
            
            cm = model_results['confusion_matrix']
            cm_fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted 0', 'Predicted 1'],
                y=['Actual 0', 'Actual 1'],
                colorscale=[[0, "#000033"], [1, "#00ccff"]],
                showscale=False,
                text=cm,
                texttemplate="%{text}",
                textfont={"size":20}
            ))
            cm_fig.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted Class",
                yaxis_title="Actual Class",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0.8)",
                plot_bgcolor="rgba(0,0,0,0.8)",
                font=dict(color="#00ccff")
            )
            st.plotly_chart(cm_fig, use_container_width=True)
            
            # ROC Curve
            st.markdown('<h3 class="futuristic-header">Probability Wave Function (ROC Curve)</h3>', unsafe_allow_html=True)
            
            roc_fig = create_futuristic_theme_chart(
                model_results['fpr'],
                model_results['tpr'],
                model_results['auc']
            )
            st.plotly_chart(roc_fig, use_container_width=True)
            
            # Feature Importance
            st.markdown('<h3 class="futuristic-header">Quantum Feature Significance</h3>', unsafe_allow_html=True)
            
            coeffs = model_results['coefficients'][0]
            coeff_df = pd.DataFrame({
                'Feature': selected_features,
                'Coefficient': coeffs,
                'Absolute Value': np.abs(coeffs)
            })
            coeff_df = coeff_df.sort_values('Absolute Value', ascending=False)
            
            coeff_fig = px.bar(
                coeff_df,
                x='Feature',
                y='Coefficient',
                title='Feature Coefficients',
                color='Coefficient',
                color_continuous_scale=['#ff00cc', '#ffffff', '#00ccff']
            )
            coeff_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0.8)",
                plot_bgcolor="rgba(0,0,0,0.8)",
                font=dict(color="#00ccff")
            )
            st.plotly_chart(coeff_fig, use_container_width=True)
            
            # Prediction tool
            st.markdown('<h3 class="futuristic-header">🌠 FUTURE OUTCOME PREDICTOR 🌠</h3>', unsafe_allow_html=True)
            st.markdown('<p class="futuristic-text">Enter values to predict future binary outcomes...</p>', unsafe_allow_html=True)
            
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
            if st.button("🔮 Calculate Quantum Probability 🔮"):
                input_df = pd.DataFrame([input_values])
                
                # Apply the same transformations as the training data if needed
                if standardize:
                    input_df = model_results['scaler'].transform(input_df)
                
                prediction_prob = model_results['model'].predict_proba(input_df)[0, 1]
                prediction_class = 1 if prediction_prob > 0.5 else 0
                
                # Display prediction with futuristic styling
                st.markdown(f"""
                <div style="background-color: rgba(0, 12, 24, 0.7); padding: 20px; border: 1px solid #00ccff; border-radius: 5px; box-shadow: 0 0 20px #00ccff;">
                    <h3 style="color: #00ccff; text-align: center; text-shadow: 0 0 10px #00ccff;">Quantum Algorithm Prediction</h3>
                    <div style="display: flex; justify-content: space-around; margin-bottom: 20px;">
                        <div>
                            <p style="color: #ffffff; text-align: center; font-size: 18px;">
                                Probability of Class 1: <span style="color: #00ccff; font-weight: bold;">{prediction_prob:.4f}</span>
                            </p>
                        </div>
                        <div>
                            <p style="color: #ffffff; text-align: center; font-size: 18px;">
                                Predicted Class: <span style="color: #00ccff; font-weight: bold;">{prediction_class}</span>
                            </p>
                        </div>
                    </div>
                    <div style="background-color: rgba(0, 204, 255, 0.1); height: 30px; border-radius: 15px; overflow: hidden;">
                        <div style="background-color: #00ccff; width: {prediction_prob * 100}%; height: 100%; border-radius: 15px;"></div>
                    </div>
                    <p style="color: #0099cc; text-align: center; font-style: italic; margin-top: 10px;">
                        Quantum prediction complete with {prediction_prob * 100:.2f}% confidence.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Quantum calculation error: {e}")
