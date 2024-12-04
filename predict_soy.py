import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

def create_features(df, target_col, n_lags=5):
    """Create time series features from the dataset"""
    df_copy = df.copy()
    
    # Create lag features
    for i in range(1, n_lags + 1):
        df_copy[f'{target_col}_lag_{i}'] = df_copy[target_col].shift(i)
        
    # Create price changes
    df_copy[f'{target_col}_change'] = df_copy[target_col].pct_change()
    
    # Add other commodities as features
    other_commodities = [col for col in df_copy.columns if col != target_col]
    for commodity in other_commodities:
        df_copy[f'{commodity}_change'] = df_copy[commodity].pct_change()
    
    # Drop rows with NaN values
    df_copy = df_copy.dropna()
    
    return df_copy

def evaluate_model(y_true, y_pred, model_name):
    """Calculate and print model evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} Results:")
    print(f"MSE: {mse:.4f}")
    print(f"R2: {r2:.4f}")
    
    return {'MSE': mse, 'R2': r2}

def plot_predictions(y_test, y_pred, model_name):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='Actual')
    plt.plot(y_test.index, y_pred, label='Predicted')
    plt.title(f'Actual vs Predicted Soy Prices ({model_name})')
    plt.legend()
    plt.savefig(f'soy_predictions_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

def main():
    # Read the data
    df = pd.read_csv('merged_data.csv')
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.set_index('DATE')
    
    # Prepare features
    target_col = 'soy'
    df_features = create_features(df, target_col)
    
    # Split features and target
    X = df_features.drop(target_col, axis=1)
    y = df_features[target_col]
    
    # Create train/test split using time-based split
    train_size = int(len(df_features) * 0.8)
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    # Train and evaluate models
    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate model
        results[name] = evaluate_model(y_test, y_pred, name)
        results[name]['model'] = model
        
        # Plot predictions
        plot_predictions(y_test, y_pred, name)
        
        # Feature importance for Random Forest
        if name == 'Random Forest':
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
            
            # Save feature importance plot
            plt.figure(figsize=(10, 6))
            plt.bar(feature_importance['feature'][:10], feature_importance['importance'][:10])
            plt.xticks(rotation=45, ha='right')
            plt.title('Top 10 Most Important Features')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            plt.close()
    
    # Save results to CSV
    results_df = pd.DataFrame({name: {'MSE': res['MSE'], 'R2': res['R2']} 
                             for name, res in results.items()}).T
    results_df.to_csv('model_results.csv')

if __name__ == "__main__":
    main() 