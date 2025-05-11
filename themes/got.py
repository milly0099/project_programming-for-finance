import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from utils.ml_models import train_kmeans
from utils.visualization import create_got_theme_chart
from utils.audio import autoplay_audio
import time
import os
from sklearn.metrics import silhouette_score

def got_theme(data):
    # Set the Game of Thrones theme styling
    st.markdown("""
    <style>
    .got-header {
        color: #d3a625;
        text-shadow: 2px 2px 4px #000000;
        font-family: 'Times New Roman', Times, serif;
    }
    .got-text {
        color: #c0c0c0;
        font-family: 'Times New Roman', Times, serif;
    }
    .house-stark {
        color: #414a4c;
        font-family: 'Times New Roman', Times, serif;
    }
    .house-lannister {
        color: #d3a625;
        font-family: 'Times New Roman', Times, serif;
    }
    .house-targaryen {
        color: #9e1c20;
        font-family: 'Times New Roman', Times, serif;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with Game of Thrones theme
    st.markdown('<h1 class="got-header">⚔️ THE SEVEN KINGDOMS OF CLUSTERING: K-MEANS ⚔️</h1>', unsafe_allow_html=True)
    st.markdown('<p class="got-text">"When you play the game of financial clusters, you win or you die. There is no middle ground."</p>', unsafe_allow_html=True)
    
    # Display themed image
    st.image("https://pixabay.com/get/gcb3372c89278ebd2107814fc6bb1f0c38f010bacb40ed6288cb4c7fbb437a8e6a7762f9a2d7d09351d70e9579ed351955d46692a2a55a3d61947c1a161530bda_1280.jpg", 
             caption="The Iron Bank of Braavos")
    
    # Data overview section
    st.markdown('<h2 class="got-header">📜 THE MAESTERS\' RECORDS 📜</h2>', unsafe_allow_html=True)
    if st.checkbox("Reveal the scrolls of data", value=True):
        st.dataframe(data.head())
        
        # Show basic statistics
        st.markdown('<h3 class="got-header">The Maesters\' Calculations</h3>', unsafe_allow_html=True)
        st.dataframe(data.describe())
    
    # Feature selection for clustering
    st.markdown('<h2 class="got-header">🗡️ CHOOSE YOUR WEAPONS 🗡️</h2>', unsafe_allow_html=True)
    
    # Get numerical columns only
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if len(numerical_columns) < 2:
        st.error("The ravens bring bad news: Not enough numerical columns for clustering. Please provide more data.")
        return
    
    # Feature selection
    selected_features = st.multiselect(
        "Select the dimensions for your clustering map",
        numerical_columns,
        default=numerical_columns[:min(3, len(numerical_columns))]  # Select first 3 features by default
    )
    
    if len(selected_features) < 2:
        st.error("A wise maester needs at least two dimensions for proper clustering.")
        return
    
    # Model parameters
    st.markdown('<h2 class="got-header">🛡️ THE SMALL COUNCIL PARAMETERS 🛡️</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        n_clusters = st.slider("Number of Houses (clusters)", 2, 10, 3)
    with col2:
        random_state = st.slider("The prophecy seed (random state)", 0, 100, 42)
    
    # Additional parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        standardize = st.checkbox("Standardize features", value=True, 
                                 help="Like equalizing the noble houses, standardization makes all features equally important")
    with col2:
        max_iter = st.slider("Maximum iterations", 100, 1000, 300, 50,
                            help="The number of times the maesters will recalculate")
    with col3:
        n_init = st.slider("Number of initializations", 1, 20, 10, 
                          help="How many times to attempt finding the best clustering")
    
    # Handle missing values
    if data[selected_features].isna().sum().sum() > 0:
        st.warning("Winter is coming... and it brings missing values in your data!")
        missing_strategy = st.radio(
            "How shall we handle the missing scrolls?",
            ["Drop rows with missing values", "Fill with mean", "Fill with median", "Fill with zero"],
            index=0
        )
        
        # Apply missing value strategy
        if missing_strategy == "Drop rows with missing values":
            clean_data = data.dropna(subset=selected_features)
            if len(clean_data) == 0:
                st.error("The night is dark and full of terrors: No data left after dropping missing values.")
                return
            st.info(f"Remaining records: {len(clean_data)} (removed {len(data) - len(clean_data)} rows with missing values)")
        else:
            clean_data = data.copy()
            for col in selected_features:
                if clean_data[col].isna().sum() > 0:
                    if missing_strategy == "Fill with mean":
                        clean_data[col] = clean_data[col].fillna(clean_data[col].mean())
                    elif missing_strategy == "Fill with median":
                        clean_data[col] = clean_data[col].fillna(clean_data[col].median())
                    elif missing_strategy == "Fill with zero":
                        clean_data[col] = clean_data[col].fillna(0)
    else:
        clean_data = data.copy()
    
    # Run clustering model when button is clicked
    if st.button("⚔️ SUMMON THE BANNERS ⚔️", help="Run K-Means clustering"):
        # Dramatic effect for Game of Thrones theme
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(101):
            progress_bar.progress(i)
            if i < 30:
                status_text.markdown('<p class="got-text">The ravens are gathering intelligence...</p>', unsafe_allow_html=True)
            elif i < 60:
                status_text.markdown('<p class="got-text">The maesters are calculating the optimal groupings...</p>', unsafe_allow_html=True)
            elif i < 90:
                status_text.markdown('<p class="got-text">The Small Council is reviewing the findings...</p>', unsafe_allow_html=True)
            else:
                status_text.markdown('<p class="got-text">The houses are being aligned...</p>', unsafe_allow_html=True)
            time.sleep(0.02)
        
        status_text.empty()
        
        # Use the helper function to train the model
        X = clean_data[selected_features]
        
        try:
            model_results = train_kmeans(
                X, 
                n_clusters=n_clusters, 
                random_state=random_state,
                standardize=standardize,
                max_iter=max_iter,
                n_init=n_init
            )
            
            # Add cluster labels to the data
            clean_data['cluster'] = model_results['labels']
            
            # Play success sound if enabled
            if st.session_state.sound_enabled and os.path.exists("assets/audio/model_success.mp3"):
                autoplay_audio("assets/audio/model_success.mp3")
            
            # Display results
            st.markdown('<h2 class="got-header">🏰 THE GREAT HOUSES OF WESTEROS 🏰</h2>', unsafe_allow_html=True)
            
            # House names based on number of clusters
            house_names = ["Stark", "Lannister", "Targaryen", "Baratheon", "Greyjoy", 
                           "Tyrell", "Martell", "Tully", "Arryn", "Bolton"][:n_clusters]
            
            # Cluster information
            cluster_info = pd.DataFrame({
                'Cluster': range(n_clusters),
                'House': house_names,
                'Count': [sum(model_results['labels'] == i) for i in range(n_clusters)]
            })
            
            # Custom colors for houses
            house_colors = ["#414a4c", "#d3a625", "#9e1c20", "#101010", "#003333", 
                           "#006600", "#ff8000", "#0000ff", "#6699ff", "#660000"][:n_clusters]
            
            # Display cluster distribution
            st.markdown('<h3 class="got-header">The Great Houses and Their Bannermen</h3>', unsafe_allow_html=True)
            
            fig = px.bar(
                cluster_info,
                x='House',
                y='Count',
                title='Distribution of Data Points Among Houses',
                color='House',
                color_discrete_sequence=house_colors
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(20,20,20,0.8)",
                plot_bgcolor="rgba(20,20,20,0.8)",
                font=dict(color="#d3a625"),
                title_font=dict(color="#d3a625")
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display cluster centers
            st.markdown('<h3 class="got-header">The Seat of Power: Cluster Centers</h3>', unsafe_allow_html=True)
            
            centers_df = pd.DataFrame(
                model_results['cluster_centers'],
                columns=selected_features
            )
            centers_df['House'] = house_names
            
            # Display cluster centers in a table with styling
            st.dataframe(centers_df)
            
            # Visualize clusters
            st.markdown('<h3 class="got-header">The Map of the Seven Kingdoms</h3>', unsafe_allow_html=True)
            
            # Create PCA if we have more than 2 dimensions
            if len(selected_features) > 2:
                st.markdown('<p class="got-text">The map is complicated, using magic (PCA) to reduce to 2 dimensions...</p>', unsafe_allow_html=True)
                
                X_pca = model_results['pca_result']
                pca_df = pd.DataFrame(X_pca, columns=['Component 1', 'Component 2'])
                pca_df['cluster'] = model_results['labels']
                pca_df['House'] = pca_df['cluster'].apply(lambda x: house_names[x])
                
                # Create scatter plot
                got_chart = create_got_theme_chart(
                    pca_df,
                    house_names,
                    house_colors,
                    x_col='Component 1',
                    y_col='Component 2',
                    cluster_centers_pca=model_results['pca_centers']
                )
                st.plotly_chart(got_chart, use_container_width=True)
                
                # Explained variance
                st.markdown(f'<p class="got-text">These two components explain {model_results["pca_variance_ratio_sum"]:.2%} of the total variance in the data.</p>', unsafe_allow_html=True)
            
            else:
                # If we have exactly 2 features, plot directly
                scatter_df = clean_data.copy()
                scatter_df['House'] = scatter_df['cluster'].apply(lambda x: house_names[x])
                
                # Create scatter plot
                got_chart = create_got_theme_chart(
                    scatter_df,
                    house_names,
                    house_colors,
                    x_col=selected_features[0],
                    y_col=selected_features[1],
                    cluster_centers=model_results['cluster_centers']
                )
                st.plotly_chart(got_chart, use_container_width=True)
            
            # Silhouette score
            st.markdown('<h3 class="got-header">The Strength of the Alliance</h3>', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background-color: rgba(20,20,20,0.8); padding: 20px; border: 1px solid #d3a625; border-radius: 5px;">
                <h4 style="color: #d3a625; text-align: center;">Silhouette Score</h4>
                <p style="color: #c0c0c0; text-align: center; font-size: 24px;">
                    {model_results['silhouette']:.4f}
                </p>
                <p style="color: #c0c0c0; text-align: center; font-style: italic;">
                    {get_silhouette_interpretation(model_results['silhouette'])}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature analysis per cluster
            st.markdown('<h3 class="got-header">The Characteristics of Each House</h3>', unsafe_allow_html=True)
            
            # Get feature means per cluster
            cluster_profiles = clean_data.groupby('cluster')[selected_features].mean()
            cluster_profiles.index = house_names
            cluster_profiles.index.name = 'House'
            
            # Create heatmap for feature analysis
            fig = px.imshow(
                cluster_profiles,
                title="House Characteristics (Feature Means per Cluster)",
                color_continuous_scale=["#000000", "#9e1c20", "#d3a625"],
                aspect="auto"
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(20,20,20,0.8)",
                plot_bgcolor="rgba(20,20,20,0.8)",
                font=dict(color="#d3a625"),
                yaxis=dict(title="House"),
                xaxis=dict(title="Feature")
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Elbow method for optimal number of clusters
            st.markdown('<h3 class="got-header">The Prophecy: Optimal Number of Houses</h3>', unsafe_allow_html=True)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=model_results['k_range'], 
                y=model_results['inertia_values'],
                mode='lines+markers',
                name='Inertia',
                line=dict(color='#d3a625', width=2),
                marker=dict(size=8, color='#9e1c20')
            ))
            fig.update_layout(
                title="Elbow Method for Optimal k",
                xaxis_title="Number of Clusters (k)",
                yaxis_title="Inertia (Within-Cluster Sum of Squares)",
                template="plotly_dark",
                paper_bgcolor="rgba(20,20,20,0.8)",
                plot_bgcolor="rgba(20,20,20,0.8)",
                font=dict(color="#d3a625")
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction tool
            st.markdown('<h3 class="got-header">🔮 Which House Do You Belong To? 🔮</h3>', unsafe_allow_html=True)
            st.markdown('<p class="got-text">Enter values to discover your house allegiance...</p>', unsafe_allow_html=True)
            
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
            if st.button("🔮 Reveal My House 🔮"):
                input_df = pd.DataFrame([input_values])
                
                # Apply the same transformations as the training data if needed
                if standardize:
                    input_scaled = model_results['scaler'].transform(input_df)
                    input_array = input_scaled
                else:
                    input_array = input_df.values
                
                # Predict cluster
                prediction = model_results['model'].predict(input_array)[0]
                house = house_names[prediction]
                house_color = house_colors[prediction]
                
                # Get distance to cluster center
                distance = np.linalg.norm(input_array - model_results['model'].cluster_centers_[prediction])
                
                # Display prediction with GoT styling
                loyalty_level = get_loyalty_level(distance)
                
                st.markdown(f"""
                <div style="background-color: rgba(20,20,20,0.9); padding: 20px; border: 1px solid {house_color}; border-radius: 5px;">
                    <h3 style="color: {house_color}; text-align: center; text-shadow: 1px 1px 2px #000000;">The Old Gods have spoken...</h3>
                    <p style="color: #ffffff; text-align: center; font-size: 28px; margin: 20px 0;">
                        You belong to House <span style="color: {house_color}; font-weight: bold;">{house}</span>
                    </p>
                    <p style="color: #c0c0c0; text-align: center; font-style: italic; margin-top: 10px;">
                        Your loyalty to House {house}: {loyalty_level}
                    </p>
                    <div style="text-align: center; margin-top: 20px;">
                        <p style="color: #a0a0a0; font-style: italic;">"{get_house_quote(house)}"</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"The maesters encountered an error: {e}")

def get_silhouette_interpretation(score):
    if score < 0:
        return "The houses are at war! Negative score indicates poor clustering."
    elif score < 0.25:
        return "There is much discord in the realm. The clustering structure is weak."
    elif score < 0.5:
        return "The alliances are forming, but loyalty is questionable."
    elif score < 0.75:
        return "The great houses have clear territories and loyal bannermen."
    else:
        return "Perfect harmony in the Seven Kingdoms! The houses are clearly defined."

def get_loyalty_level(distance):
    if distance < 0.5:
        return "Unwavering (You are a true member of this house)"
    elif distance < 1.0:
        return "Strong (Your loyalty to this house is clear)"
    elif distance < 2.0:
        return "Moderate (You have clear ties to this house)"
    elif distance < 3.0:
        return "Questionable (Your allegiance might be divided)"
    else:
        return "You might be a secret agent for another house..."

def get_house_quote(house):
    quotes = {
        "Stark": "Winter is Coming",
        "Lannister": "A Lannister Always Pays His Debts",
        "Targaryen": "Fire and Blood",
        "Baratheon": "Ours is the Fury",
        "Greyjoy": "We Do Not Sow",
        "Tyrell": "Growing Strong",
        "Martell": "Unbowed, Unbent, Unbroken",
        "Tully": "Family, Duty, Honor",
        "Arryn": "As High As Honor",
        "Bolton": "Our Blades Are Sharp"
    }
    return quotes.get(house, "Valar Morghulis")
