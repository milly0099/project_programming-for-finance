import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix, roc_curve, auc
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
import math

def train_linear_regression(X, y, test_size=0.2, random_state=42):
    """
    Trains a linear regression model and returns various performance metrics.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature variables
    y : pandas.Series
        Target variable
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    dict
        Dictionary containing model, predictions, and performance metrics
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate performance metrics
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Get coefficients and intercept
    coefficients = model.coef_
    intercept = model.intercept_
    
    # Return results as dictionary
    return {
        'model': model,
        'y_test': y_test,
        'y_pred': y_test_pred,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'coefficients': coefficients,
        'intercept': intercept
    }

def train_logistic_regression(X, y, test_size=0.2, random_state=42, standardize=True, C=1.0, max_iter=100):
    """
    Trains a logistic regression model and returns various performance metrics.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature variables
    y : pandas.Series
        Target variable (binary)
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
    standardize : bool
        Whether to standardize the features
    C : float
        Inverse of regularization strength
    max_iter : int
        Maximum number of iterations for solver
    
    Returns:
    --------
    dict
        Dictionary containing model, predictions, and performance metrics
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Standardize features if requested
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        scaler = None
    
    # Create and train the model
    model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = auc(fpr, tpr)
    
    # Get coefficients
    coefficients = model.coef_
    
    # Return results as dictionary
    return {
        'model': model,
        'scaler': scaler,
        'y_test': y_test,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'auc': auc_score,
        'coefficients': coefficients
    }

def train_kmeans(X, n_clusters=3, random_state=42, standardize=True, max_iter=300, n_init=10):
    """
    Trains a KMeans clustering model and returns cluster assignments and metrics.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature variables
    n_clusters : int
        Number of clusters to form
    random_state : int
        Random seed for reproducibility
    standardize : bool
        Whether to standardize the features
    max_iter : int
        Maximum number of iterations for a single run
    n_init : int
        Number of times the algorithm will be run with different initial centroids
    
    Returns:
    --------
    dict
        Dictionary containing model, cluster assignments, and metrics
    """
    # Standardize features if requested
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values
        scaler = None
    
    # Create and train the model
    model = KMeans(n_clusters=n_clusters, random_state=random_state, max_iter=max_iter, n_init=n_init)
    labels = model.fit_predict(X_scaled)
    
    # Get cluster centers
    cluster_centers = model.cluster_centers_
    
    # Calculate silhouette score if more than one cluster
    if n_clusters > 1:
        from sklearn.metrics import silhouette_score
        silhouette = silhouette_score(X_scaled, labels)
    else:
        silhouette = 0  # Cannot calculate silhouette for a single cluster
    
    # Perform PCA for visualization if more than 2 dimensions
    pca_result = None
    pca_centers = None
    pca_variance_ratio_sum = 0
    
    if X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=random_state)
        pca_result = pca.fit_transform(X_scaled)
        pca_centers = pca.transform(cluster_centers)
        pca_variance_ratio_sum = sum(pca.explained_variance_ratio_)
    
    # Calculate inertia for different k values (for elbow method)
    k_range = range(1, min(11, len(X) // 5))  # Up to 10 or 1/5 of data points
    inertia_values = []
    
    for k in k_range:
        k_model = KMeans(n_clusters=k, random_state=random_state, max_iter=max_iter, n_init=n_init)
        k_model.fit(X_scaled)
        inertia_values.append(k_model.inertia_)
    
    # Return results as dictionary
    return {
        'model': model,
        'scaler': scaler,
        'labels': labels,
        'cluster_centers': cluster_centers,
        'silhouette': silhouette,
        'pca_result': pca_result,
        'pca_centers': pca_centers,
        'pca_variance_ratio_sum': pca_variance_ratio_sum,
        'k_range': list(k_range),
        'inertia_values': inertia_values
    }

def train_random_forest(X, y, test_size=0.2, random_state=42, n_estimators=100, max_depth=None, 
                        min_samples_split=2, criterion=None, bootstrap=True, is_classification=True):
    """
    Trains a Random Forest model (classifier or regressor) and returns performance metrics.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature variables
    y : pandas.Series
        Target variable
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
    n_estimators : int
        Number of trees in the forest
    max_depth : int or None
        Maximum depth of the trees
    min_samples_split : int
        Minimum number of samples required to split an internal node
    criterion : str
        Function to measure the quality of a split
    bootstrap : bool
        Whether to use bootstrap samples
    is_classification : bool
        Whether to use RandomForestClassifier (True) or RandomForestRegressor (False)
    
    Returns:
    --------
    dict
        Dictionary containing model, predictions, and performance metrics
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Set criterion if not specified
    if criterion is None:
        criterion = 'gini' if is_classification else 'squared_error'
    
    # Create and train model
    if is_classification:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterion,
            bootstrap=bootstrap,
            random_state=random_state
        )
    else:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterion,
            bootstrap=bootstrap,
            random_state=random_state
        )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Get feature importance
    feature_importance = model.feature_importances_
    
    # Calculate performance metrics
    if is_classification:
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        
        # Return classifier metrics
        return {
            'model': model,
            'y_test': y_test,
            'y_pred': y_pred,
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': cm,
            'feature_importance': feature_importance
        }
    else:
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = np.mean(np.abs(y_test - y_pred))
        
        # Return regressor metrics
        return {
            'model': model,
            'y_test': y_test,
            'y_pred': y_pred,
            'r2_score': r2,
            'rmse': rmse,
            'mae': mae,
            'feature_importance': feature_importance
        }
