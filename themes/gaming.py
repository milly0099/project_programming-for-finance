import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from utils.ml_models import train_random_forest
from utils.visualization import create_gaming_theme_chart
from utils.audio import autoplay_audio
import time
import os

def gaming_theme(data):
    # Set the gaming theme styling
    st.markdown("""
    <style>
    .gaming-header {
        color: #ff4500;
        text-shadow: 2px 2px 4px #000000;
        font-family: 'Arial', sans-serif;
    }
    .gaming-text {
        color: #00ff00;
        font-family: 'Courier New', Courier, monospace;
    }
    .power-up {
        color: #ffff00;
        font-weight: bold;
    }
    .bonus-points {
        color: #00ffff;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with gaming theme
    st.markdown('<h1 class="gaming-header">🎮 FINANCIAL QUEST: RANDOM FOREST ADVENTURE 🎲</h1>', unsafe_allow_html=True)
    st.markdown('<p class="gaming-text">Welcome Player 1! Ready to level up your financial analysis skills?</p>', unsafe_allow_html=True)
    
    # Display themed image
    st.image("https://pixabay.com/get/g157dba804c52ef83b4d8f47e521f93a76787e4fbbd8c5f9ce991821a1b81158204f1cd839bb01fd9055d967011fd9868d315b6f0ff9e5fc103b5265bbae1f004_1280.jpg", 
             caption="Market Game Stats")
    
    # Data overview section
    st.markdown('<h2 class="gaming-header">📊 GAME STATS 📊</h2>', unsafe_allow_html=True)
    if st.checkbox("Show data leaderboard", value=True):
        st.dataframe(data.head())
        
        # Show basic statistics
        st.markdown('<h3 class="gaming-header">High Scores & Stats</h3>', unsafe_allow_html=True)
        st.dataframe(data.describe())
    
    # Model type selection
    st.markdown('<h2 class="gaming-header">🎯 SELECT YOUR GAME MODE 🎯</h2>', unsafe_allow_html=True)
    
    model_type = st.radio(
        "Choose your adventure type",
        ["Regression Quest (predict continuous values)", "Classification Battle (predict categories)"],
        index=0
    )
    
    # Get numerical columns only
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if len(numerical_columns) < 2:
        st.error("Not enough power-ups! You need more numerical features for this adventure.")
        return
    
    # Feature selection
    st.markdown('<h2 class="gaming-header">⚔️ SELECT YOUR WEAPONS ⚔️</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox(
            "Choose your quest target (what to predict)",
            numerical_columns,
            index=0
        )
    
    with col2:
        available_features = [col for col in numerical_columns if col != target_col]
        selected_features = st.multiselect(
            "Select your power-ups (feature variables)",
            available_features,
            default=available_features[:min(3, len(available_features))]
        )
    
    if not selected_features:
        st.error("You need to select at least one power-up (feature) to begin your quest!")
        return
    
    # For classification, we need to convert the target to categorical
    if model_type == "Classification Battle (predict categories)":
        st.markdown('<h3 class="gaming-header">Level Design for Classification</h3>', unsafe_allow_html=True)
        
        # Check if the target already has few unique values (likely categorical)
        unique_values = data[target_col].nunique()
        
        if unique_values > 10:  # If many unique values, likely continuous
            st.warning(f"Your target has {unique_values} unique values. For classification, we'll convert it to categories.")
            
            # Let user choose how to categorize
            categorize_method = st.radio(
                "Choose your categorization spell",
                ["Quantiles (equal sized groups)", "Equal ranges", "Custom threshold"],
                index=0
            )
            
            num_classes = st.slider("Number of classes (game difficulty levels)", 2, 10, 3)
            
            # Create categorical target based on method
            if categorize_method == "Quantiles (equal sized groups)":
                data['target_category'] = pd.qcut(
                    data[target_col], 
                    q=num_classes, 
                    labels=[f"Level {i+1}" for i in range(num_classes)]
                )
                st.info(f"Created {num_classes} levels with approximately equal number of data points in each.")
                
            elif categorize_method == "Equal ranges":
                data['target_category'] = pd.cut(
                    data[target_col], 
                    bins=num_classes, 
                    labels=[f"Level {i+1}" for i in range(num_classes)]
                )
                st.info(f"Created {num_classes} levels with equal value ranges.")
                
            elif categorize_method == "Custom threshold":
                # For simplicity, we'll just do binary classification with custom threshold
                threshold = st.slider(
                    f"Threshold for {target_col}",
                    float(data[target_col].min()),
                    float(data[target_col].max()),
                    float(data[target_col].median())
                )
                data['target_category'] = np.where(data[target_col] > threshold, "High Level", "Low Level")
                st.info(f"Created binary classes: values > {threshold:.4f} are 'High Level', others are 'Low Level'.")
            
            # Encode the target for modeling
            le = LabelEncoder()
            data['target_encoded'] = le.fit_transform(data['target_category'])
            target_for_model = 'target_encoded'
            
            # Show distribution of classes
            st.markdown('<h4 class="gaming-header">Class Distribution</h4>', unsafe_allow_html=True)
            fig = px.histogram(
                data, 
                x='target_category',
                color='target_category',
                title="Player Levels Distribution",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0.8)",
                plot_bgcolor="rgba(0,0,0,0.8)",
                font=dict(color="#00ff00")
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            # If already few categories, treat as categorical
            st.info(f"Your target has {unique_values} unique values, perfect for classification!")
            # Encode the target
            le = LabelEncoder()
            data['target_encoded'] = le.fit_transform(data[target_col])
            data['target_category'] = data[target_col]
            target_for_model = 'target_encoded'
    else:
        # For regression, use the original target
        target_for_model = target_col
    
    # Model parameters
    st.markdown('<h2 class="gaming-header">🎛️ GAME SETTINGS 🎛️</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        n_estimators = st.slider("Number of trees (game levels)", 10, 500, 100, 10)
        max_depth = st.slider("Max depth (dungeon depth)", 1, 30, 10)
    with col2:
        test_size = st.slider("Test set size (boss battle difficulty %)", 10, 50, 20) / 100
        random_state = st.slider("Random seed (game world seed)", 0, 100, 42)
    
    # Additional parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        min_samples_split = st.slider("Min samples to split (party size)", 2, 20, 2)
    with col2:
        criterion = st.selectbox(
            "Split criterion (battle strategy)",
            ["gini", "entropy"] if model_type == "Classification Battle (predict categories)" else ["squared_error", "absolute_error"]
        )
    with col3:
        bootstrap = st.checkbox("Bootstrap (random power-ups)", value=True)
    
    # Handle missing values
    if data[selected_features + [target_for_model]].isna().sum().sum() > 0:
        st.warning("Your quest data contains missing values! These need to be fixed before proceeding.")
        missing_strategy = st.radio(
            "Choose your repair spell",
            ["Drop rows with missing values", "Fill with mean", "Fill with median", "Fill with zero"],
            index=0
        )
        
        # Apply missing value strategy
        if missing_strategy == "Drop rows with missing values":
            clean_data = data.dropna(subset=selected_features + [target_for_model])
            if len(clean_data) == 0:
                st.error("Game over! No data left after dropping missing values.")
                return
            st.info(f"Remaining player data: {len(clean_data)} (removed {len(data) - len(clean_data)} incomplete records)")
        else:
            clean_data = data.copy()
            for col in selected_features + [target_for_model]:
                if clean_data[col].isna().sum() > 0:
                    if missing_strategy == "Fill with mean":
                        clean_data[col] = clean_data[col].fillna(clean_data[col].mean())
                    elif missing_strategy == "Fill with median":
                        clean_data[col] = clean_data[col].fillna(clean_data[col].median())
                    elif missing_strategy == "Fill with zero":
                        clean_data[col] = clean_data[col].fillna(0)
    else:
        clean_data = data.copy()
    
    # Run random forest model when button is clicked
    if st.button("🎮 START GAME 🎮", help="Run Random Forest model"):
        # Gaming loading effect
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        loading_messages = [
            "Loading game assets...",
            "Generating random forest world...",
            "Spawning decision trees...",
            "Calculating feature importance...",
            "Rendering prediction engine...",
            "Preparing final battle...",
            "Game almost ready..."
        ]
        
        for i in range(101):
            progress_bar.progress(i)
            message_idx = min(int(i / 15), len(loading_messages) - 1)
            status_text.markdown(f'<p class="gaming-text">{loading_messages[message_idx]}</p>', unsafe_allow_html=True)
            time.sleep(0.02)
        
        status_text.markdown('<p class="gaming-text power-up">GAME LOADED! Let\'s begin the adventure!</p>', unsafe_allow_html=True)
        time.sleep(0.5)
        status_text.empty()
        
        # Use the helper function to train the model
        X = clean_data[selected_features]
        
        if model_type == "Classification Battle (predict categories)":
            y = clean_data[target_for_model]
            is_classification = True
        else:
            y = clean_data[target_col]
            is_classification = False
        
        try:
            model_results = train_random_forest(
                X, y, 
                test_size=test_size, 
                random_state=random_state,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                criterion=criterion,
                bootstrap=bootstrap,
                is_classification=is_classification
            )
            
            # Play success sound if enabled
            if st.session_state.sound_enabled and os.path.exists("assets/audio/model_success.mp3"):
                autoplay_audio("assets/audio/model_success.mp3")
            
            # Display results
            st.markdown('<h2 class="gaming-header">🏆 GAME RESULTS 🏆</h2>', unsafe_allow_html=True)
            
            # Performance metrics
            if is_classification:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy Score", f"{model_results['accuracy']:.4f}")
                with col2:
                    st.metric("F1 Score", f"{model_results['f1_score']:.4f}")
                
                # Confusion matrix
                st.markdown('<h3 class="gaming-header">Battle Results Matrix</h3>', unsafe_allow_html=True)
                
                # Get class names
                if 'target_category' in clean_data.columns:
                    class_names = clean_data['target_category'].unique()
                    if len(class_names) != len(set(y)):
                        class_names = [f"Class {i}" for i in range(len(set(y)))]
                else:
                    class_names = [f"Class {i}" for i in range(len(set(y)))]
                
                # Create the confusion matrix plot
                gaming_cm_chart = create_gaming_theme_chart(
                    model_results['confusion_matrix'],
                    class_names,
                    "Confusion Matrix: Battle Results"
                )
                st.plotly_chart(gaming_cm_chart, use_container_width=True)
                
            else:  # Regression metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² Score", f"{model_results['r2_score']:.4f}")
                with col2:
                    st.metric("RMSE", f"{model_results['rmse']:.4f}")
                with col3:
                    st.metric("Mean Absolute Error", f"{model_results['mae']:.4f}")
                
                # Actual vs Predicted values plot
                st.markdown('<h3 class="gaming-header">Quest Results: Actual vs Predicted</h3>', unsafe_allow_html=True)
                
                # Create scatter plot
                gaming_scatter_chart = create_gaming_theme_chart(
                    model_results['y_test'],
                    model_results['y_pred'],
                    target_col
                )
                st.plotly_chart(gaming_scatter_chart, use_container_width=True)
            
            # Feature importance
            st.markdown('<h3 class="gaming-header">Power-Up Rankings: Feature Importance</h3>', unsafe_allow_html=True)
            
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'Feature': selected_features,
                'Importance': model_results['feature_importance']
            })
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Create feature importance bar chart
            fig = px.bar(
                importance_df,
                x='Feature',
                y='Importance',
                title="Power-Up Levels (Feature Importance)",
                color='Importance',
                color_continuous_scale=['green', 'yellow', 'orange', 'red']
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0.8)",
                plot_bgcolor="rgba(0,0,0,0.8)",
                font=dict(color="#00ff00")
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction tool
            st.markdown('<h3 class="gaming-header">🎮 PLAYER PREDICTION CONSOLE 🎮</h3>', unsafe_allow_html=True)
            st.markdown('<p class="gaming-text">Enter your stats to predict the outcome...</p>', unsafe_allow_html=True)
            
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
            if st.button("🎲 ROLL FOR PREDICTION 🎲"):
                input_df = pd.DataFrame([input_values])
                
                prediction = model_results['model'].predict(input_df)[0]
                
                # Display prediction with gaming styling
                if is_classification:
                    # Convert numeric prediction back to category label if we have them
                    if 'target_category' in clean_data.columns:
                        unique_categories = clean_data['target_category'].unique()
                        unique_encoded = clean_data['target_encoded'].unique()
                        category_map = dict(zip(unique_encoded, unique_categories))
                        prediction_label = category_map.get(prediction, f"Class {prediction}")
                    else:
                        prediction_label = f"Class {prediction}"
                    
                    # Also get prediction probabilities
                    prediction_proba = model_results['model'].predict_proba(input_df)[0]
                    
                    # Display classification result
                    st.markdown(f"""
                    <div style="background-color: rgba(0,0,0,0.8); padding: 20px; border: 2px solid #ff4500; border-radius: 10px;">
                        <h3 style="color: #ff4500; text-align: center; text-shadow: 2px 2px 4px #000000;">🎮 GAME RESULT 🎮</h3>
                        <p style="color: #ffff00; text-align: center; font-size: 28px; margin: 20px 0;">
                            Predicted Class: <span style="color: #00ffff; font-weight: bold;">{prediction_label}</span>
                        </p>
                        <div style="margin-top: 20px;">
                            <h4 style="color: #00ff00; text-align: center;">Class Probabilities</h4>
                            <div style="display: flex; justify-content: center;">
                    """, unsafe_allow_html=True)
                    
                    # Create probability bars
                    for i, prob in enumerate(prediction_proba):
                        if 'target_category' in clean_data.columns:
                            class_name = category_map.get(i, f"Class {i}")
                        else:
                            class_name = f"Class {i}"
                        
                        color = ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#00ffff", "#ff00ff"][i % 6]
                        
                        st.markdown(f"""
                        <div style="margin: 0 10px; text-align: center;">
                            <div style="background-color: #333333; height: 100px; width: 40px; border-radius: 5px; position: relative; display: inline-block;">
                                <div style="background-color: {color}; position: absolute; bottom: 0; width: 100%; height: {prob * 100}%; border-radius: 0 0 5px 5px;"></div>
                            </div>
                            <p style="color: #ffffff; margin-top: 5px;">{class_name}</p>
                            <p style="color: {color};">{prob:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("""
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:  # Regression result
                    # Get prediction interval if available
                    lower_bound = upper_bound = None
                    interval_width = 0
                    
                    if hasattr(model_results['model'], 'estimators_'):
                        # Calculate prediction intervals using the forest's variance
                        predictions = [tree.predict(input_df)[0] for tree in model_results['model'].estimators_]
                        lower_bound = np.percentile(predictions, 2.5)
                        upper_bound = np.percentile(predictions, 97.5)
                        interval_width = upper_bound - lower_bound
                    
                    # Map prediction to a game achievement level
                    y_min = float(clean_data[target_col].min())
                    y_max = float(clean_data[target_col].max())
                    y_range = y_max - y_min
                    
                    achievement_level = int(((prediction - y_min) / y_range) * 100)
                    achievement_level = max(0, min(achievement_level, 100))  # Ensure between 0-100
                    
                    # Get achievement text based on level
                    if achievement_level < 20:
                        achievement_text = "Novice Explorer"
                        color = "#00ff00"  # Green
                    elif achievement_level < 40:
                        achievement_text = "Skilled Adventurer"
                        color = "#00ffff"  # Cyan
                    elif achievement_level < 60:
                        achievement_text = "Expert Tactician"
                        color = "#0000ff"  # Blue
                    elif achievement_level < 80:
                        achievement_text = "Master Strategist"
                        color = "#ff00ff"  # Magenta
                    else:
                        achievement_text = "Legendary Champion"
                        color = "#ff0000"  # Red
                    
                    # Display regression result
                    st.markdown(f"""
                    <div style="background-color: rgba(0,0,0,0.8); padding: 20px; border: 2px solid #ff4500; border-radius: 10px;">
                        <h3 style="color: #ff4500; text-align: center; text-shadow: 2px 2px 4px #000000;">🎮 QUEST COMPLETED 🎮</h3>
                        <p style="color: #ffff00; text-align: center; font-size: 28px; margin: 20px 0;">
                            Predicted {target_col}: <span style="color: {color}; font-weight: bold;">{prediction:.4f}</span>
                        </p>
                        <div style="width: 100%; background-color: #333333; height: 30px; border-radius: 15px; margin: 20px 0; position: relative;">
                            <div style="position: absolute; left: 0; top: 0; height: 100%; width: {achievement_level}%; background-color: {color}; border-radius: 15px;"></div>
                        </div>
                        <p style="color: {color}; text-align: center; font-size: 20px; margin: 10px 0;">
                            Achievement: {achievement_text} (Level {achievement_level}/100)
                        </p>
                    """, unsafe_allow_html=True)
                    
                    # Add prediction interval if available
                    if lower_bound is not None and upper_bound is not None:
                        st.markdown(f"""
                        <p style="color: #00ff00; text-align: center; margin-top: 15px;">
                            95% Confidence Range: [{lower_bound:.4f} to {upper_bound:.4f}]
                        </p>
                        <p style="color: #bbbbbb; text-align: center; font-style: italic; font-size: 14px;">
                            (This represents the range of possible outcomes in the game)
                        </p>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Game crashed! Error: {e}")
