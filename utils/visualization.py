import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def create_zombie_theme_chart(y_true, y_pred, target_col):
    """
    Creates a scatter plot with zombie theme styling for actual vs predicted values.
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    target_col : str
        Name of the target column
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Styled plotly figure
    """
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot for actual vs predicted
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        marker=dict(
            color='#9a0000',
            size=8,
            symbol='circle',
            line=dict(
                color='#000000',
                width=1
            )
        ),
        name='Predictions'
    ))
    
    # Add perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(
            color='#600000',
            width=2,
            dash='dash'
        ),
        name='Perfect Prediction'
    ))
    
    # Update layout with zombie theme
    fig.update_layout(
        title=f"Actual vs Predicted {target_col}: The Prophecy Accuracy",
        xaxis_title=f"Actual {target_col}",
        yaxis_title=f"Predicted {target_col}",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0.8)",
        plot_bgcolor="rgba(0,0,0,0.8)",
        font=dict(
            family="Times New Roman, Times, serif",
            size=12,
            color="#9a0000"
        ),
        title_font=dict(
            family="Times New Roman, Times, serif",
            size=20,
            color="#9a0000"
        ),
        legend=dict(
            font=dict(
                family="Times New Roman, Times, serif",
                size=10,
                color="#9a0000"
            )
        ),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig

def create_futuristic_theme_chart(fpr, tpr, auc_score):
    """
    Creates a ROC curve with futuristic theme styling.
    
    Parameters:
    -----------
    fpr : array-like
        False positive rates
    tpr : array-like
        True positive rates
    auc_score : float
        Area under the ROC curve
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Styled plotly figure
    """
    # Create figure
    fig = go.Figure()
    
    # Add ROC curve
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        line=dict(
            color='#00ccff',
            width=3
        ),
        name=f'ROC Curve (AUC = {auc_score:.4f})'
    ))
    
    # Add diagonal line (random classifier)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(
            color='#555555',
            width=2,
            dash='dash'
        ),
        name='Random Classifier (AUC = 0.5)'
    ))
    
    # Update layout with futuristic theme
    fig.update_layout(
        title="Neural Probability Matrix (ROC Curve)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0.8)",
        plot_bgcolor="rgba(0,12,24,0.8)",
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="#00ccff"
        ),
        title_font=dict(
            family="Arial, sans-serif",
            size=20,
            color="#00ccff"
        ),
        legend=dict(
            font=dict(
                family="Arial, sans-serif",
                size=10,
                color="#00ccff"
            )
        ),
        margin=dict(l=40, r=40, t=40, b=40),
        shapes=[
            # Add a grid-like pattern for futuristic feel
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(
                    color="#00ccff",
                    width=1,
                    dash="dot"
                )
            )
        ]
    )
    
    return fig

def create_got_theme_chart(data, house_names, house_colors, x_col, y_col, cluster_centers=None, cluster_centers_pca=None):
    """
    Creates a scatter plot with Game of Thrones theme styling for cluster visualization.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the data with cluster assignments
    house_names : list
        List of house names corresponding to clusters
    house_colors : list
        List of colors for each house
    x_col : str
        Column name for x-axis
    y_col : str
        Column name for y-axis
    cluster_centers : array-like, optional
        Cluster centers in original feature space
    cluster_centers_pca : array-like, optional
        Cluster centers in PCA space if PCA was applied
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Styled plotly figure
    """
    # Create figure
    fig = go.Figure()
    
    # Add scatter points colored by house (cluster)
    for i, house in enumerate(house_names):
        cluster_data = data[data['House'] == house]
        fig.add_trace(go.Scatter(
            x=cluster_data[x_col],
            y=cluster_data[y_col],
            mode='markers',
            marker=dict(
                color=house_colors[i],
                size=8,
                symbol='circle',
                line=dict(
                    color='#000000',
                    width=1
                )
            ),
            name=f'House {house}'
        ))
    
    # Add cluster centers if provided
    if cluster_centers is not None:
        fig.add_trace(go.Scatter(
            x=cluster_centers[:, 0],
            y=cluster_centers[:, 1],
            mode='markers',
            marker=dict(
                color='#ffffff',
                size=12,
                symbol='star',
                line=dict(
                    color='#d3a625',
                    width=2
                )
            ),
            name='House Seats (Cluster Centers)'
        ))
    
    # Add PCA cluster centers if provided
    if cluster_centers_pca is not None:
        fig.add_trace(go.Scatter(
            x=cluster_centers_pca[:, 0],
            y=cluster_centers_pca[:, 1],
            mode='markers',
            marker=dict(
                color='#ffffff',
                size=12,
                symbol='star',
                line=dict(
                    color='#d3a625',
                    width=2
                )
            ),
            name='House Seats (Cluster Centers)'
        ))
    
    # Update layout with Game of Thrones theme
    fig.update_layout(
        title="The Map of the Seven Kingdoms",
        xaxis_title=x_col,
        yaxis_title=y_col,
        template="plotly_dark",
        paper_bgcolor="rgba(20,20,20,0.8)",
        plot_bgcolor="rgba(20,20,20,0.8)",
        font=dict(
            family="Times New Roman, Times, serif",
            size=12,
            color="#d3a625"
        ),
        title_font=dict(
            family="Times New Roman, Times, serif",
            size=20,
            color="#d3a625"
        ),
        legend=dict(
            font=dict(
                family="Times New Roman, Times, serif",
                size=10,
                color="#d3a625"
            )
        ),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig

def create_gaming_theme_chart(x_data, y_data, title_or_col_name):
    """
    Creates a styled chart with gaming theme.
    This function can create either:
    1. A confusion matrix heatmap (if x_data is a confusion matrix)
    2. A scatter plot of actual vs predicted values (if x_data and y_data are arrays)
    
    Parameters:
    -----------
    x_data : array-like
        Either confusion matrix or actual values
    y_data : array-like or list
        Either predicted values or class names
    title_or_col_name : str
        Title for confusion matrix or column name for scatter plot
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Styled plotly figure
    """
    # Determine if this is a confusion matrix or actual vs predicted plot
    is_confusion_matrix = isinstance(x_data, np.ndarray) and len(x_data.shape) == 2
    
    # Create figure
    fig = go.Figure()
    
    if is_confusion_matrix:
        # This is a confusion matrix plot
        confusion_matrix = x_data
        class_names = y_data
        
        # Create heatmap
        heatmap = go.Heatmap(
            z=confusion_matrix,
            x=["Predicted " + str(name) for name in class_names],
            y=["Actual " + str(name) for name in class_names],
            colorscale=[[0, "#000033"], [0.25, "#0000ff"], [0.5, "#00ff00"], [0.75, "#ffff00"], [1, "#ff0000"]],
            showscale=False,
            text=confusion_matrix,
            texttemplate="%{text}",
            textfont={"size": 16}
        )
        
        fig.add_trace(heatmap)
        
        # Update layout with gaming theme
        fig.update_layout(
            title=title_or_col_name,
            xaxis_title="Predicted Class",
            yaxis_title="Actual Class",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0.8)",
            plot_bgcolor="rgba(0,0,0,0.8)",
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="#00ff00"
            ),
            title_font=dict(
                family="Arial, sans-serif",
                size=20,
                color="#ff4500"
            )
        )
        
    else:
        # This is an actual vs predicted plot
        y_true = x_data
        y_pred = y_data
        target_col = title_or_col_name
        
        # Add scatter plot for actual vs predicted
        fig.add_trace(go.Scatter(
            x=y_true,
            y=y_pred,
            mode='markers',
            marker=dict(
                color='#00ff00',
                size=8,
                symbol='circle',
                line=dict(
                    color='#000000',
                    width=1
                )
            ),
            name='Predictions'
        ))
        
        # Add perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(
                color='#ffff00',
                width=2,
                dash='dash'
            ),
            name='Perfect Prediction'
        ))
        
        # Update layout with gaming theme
        fig.update_layout(
            title=f"Actual vs Predicted {target_col}: Quest Results",
            xaxis_title=f"Actual {target_col}",
            yaxis_title=f"Predicted {target_col}",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0.8)",
            plot_bgcolor="rgba(0,0,0,0.8)",
            font=dict(
                family="Courier New, Courier, monospace",
                size=12,
                color="#00ff00"
            ),
            title_font=dict(
                family="Arial, sans-serif",
                size=20,
                color="#ff4500"
            ),
            legend=dict(
                font=dict(
                    family="Courier New, Courier, monospace",
                    size=10,
                    color="#00ff00"
                )
            ),
            margin=dict(l=40, r=40, t=40, b=40)
        )
    
    return fig
