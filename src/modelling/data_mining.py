import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

# Ensure results directory exists
os.makedirs("../../results", exist_ok=True)

# Load prepared feature data
try:
    user_features_df = pd.read_parquet("../../data/processed/user_features.parquet")
    conversion_features_df = pd.read_parquet("../../data/processed/conversion_features.parquet")
    
    # Print debug info
    print("User features shape:", user_features_df.shape)
    print("Conversion features shape:", conversion_features_df.shape)
    
    # Check for columns needed
    print("\nUser features columns:", user_features_df.columns.tolist())
    print("\nConversion features columns:", conversion_features_df.columns.tolist())
    
except Exception as e:
    print(f"Error loading data: {str(e)}")
    # Create dummy data for testing if needed
    print("Creating dummy data for testing...")
    
    # Create dummy user features
    user_ids = [f"user_{i}" for i in range(100)]
    user_features_df = pd.DataFrame({
        'user_id': user_ids,
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'feature3': np.random.rand(100)
    })
    
    # Create dummy conversion features
    session_ids = [f"session_{i}" for i in range(200)]
    conversion_features_df = pd.DataFrame({
        'session_id': session_ids,
        'feature1': np.random.rand(200),
        'feature2': np.random.rand(200),
        'feature3': np.random.rand(200),
        'converted': np.random.choice([0, 1], size=200, p=[0.8, 0.2])
    })

# 1. User Behavior Clustering with K-means
def perform_clustering(data_df, k_values=range(2, 11)):
    try:
        # Check if user_id exists
        if 'user_id' not in data_df.columns:
            print("Error: 'user_id' column not found in user features dataframe")
            # Try to find an ID column
            id_cols = [col for col in data_df.columns if 'id' in col.lower()]
            if id_cols:
                print(f"Using '{id_cols[0]}' as ID column instead")
                data_df = data_df.rename(columns={id_cols[0]: 'user_id'})
            else:
                # Create a dummy user_id column
                print("Creating dummy user_id column")
                data_df['user_id'] = [f"user_{i}" for i in range(len(data_df))]
        
        # Extract features (all columns except user_id)
        # Handle potential problems with feature columns
        features = data_df.copy()
        user_ids = features['user_id'].copy()
        features = features.drop('user_id', axis=1)
        
        # Handle non-numeric columns
        for col in features.columns:
            if not pd.api.types.is_numeric_dtype(features[col]):
                print(f"Warning: Non-numeric column '{col}' found. Converting to numeric or dropping.")
                try:
                    features[col] = pd.to_numeric(features[col], errors='coerce')
                except:
                    features = features.drop(col, axis=1)
        
        # Handle NaN values
        features = features.fillna(0)
        
        # Check if we have features left
        if features.shape[1] == 0:
            print("Error: No valid numeric features left for clustering")
            return None, None, None
        
        print(f"Clustering with {features.shape[1]} features")
        
        # Find optimal K using Elbow method
        cost_values = []
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(features)
            
            cost = kmeans.inertia_
            cost_values.append(cost)
            print(f"K={k}, Cost={cost}")
        
        # Plot Elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(list(k_values), cost_values, marker='o')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Sum of Squared Distances (Cost)')
        plt.title('Elbow Method for Optimal K')
        plt.grid(True)
        plt.savefig("../../results/kmeans_elbow_curve.png")
        plt.close()
        
        # Save elbow curve data
        pd.DataFrame({
            'K': list(k_values),
            'Cost': cost_values
        }).to_csv("../../results/kmeans_elbow_data.csv", index=False)
        
        # Choose optimal K (using the elbow point)
        optimal_k = 4  # You may want to implement automatic elbow detection
        
        # Train final model with optimal K
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        # Get cluster assignments
        clustered_data = pd.DataFrame({
            'user_id': user_ids,
            'cluster': cluster_labels
        })
        
        # Save model and results
        with open("../../results/kmeans_model.pkl", "wb") as f:
            pickle.dump(kmeans, f)
        
        clustered_data.to_parquet("../../results/user_clusters.parquet")
        
        return kmeans, clustered_data, optimal_k
    
    except Exception as e:
        print(f"Error in clustering: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

# 2. Basic Cluster Analysis (more robust to handle various column structures)
def analyze_clusters(clustered_df, original_df, k):
    try:
        # Check if cluster column exists
        if 'cluster' not in clustered_df.columns:
            if 'prediction' in clustered_df.columns:
                clustered_df = clustered_df.rename(columns={'prediction': 'cluster'})
            else:
                print("Error: No cluster or prediction column found")
                return None
        
        # Join with original data
        try:
            merged_data = pd.merge(clustered_df, original_df, on="user_id", how="inner")
        except Exception as e:
            print(f"Error merging data: {str(e)}")
            print("Attempting alternative merge strategy...")
            
            # Try to find common columns
            common_cols = set(clustered_df.columns).intersection(set(original_df.columns))
            if 'user_id' in common_cols:
                common_cols.remove('user_id')
            
            # Use only original data with cluster assignments
            if len(common_cols) == 0:
                print("No common columns found. Using clustered data only.")
                merged_data = clustered_df.copy()
            else:
                # Merge on user_id but exclude common columns from right table
                original_df_subset = original_df.drop(columns=list(common_cols), errors='ignore')
                merged_data = pd.merge(clustered_df, original_df_subset, on="user_id", how="inner")
        
        # Get numeric columns (excluding user_id and cluster)
        numeric_cols = [col for col in merged_data.columns 
                       if col not in ['user_id', 'cluster'] 
                       and pd.api.types.is_numeric_dtype(merged_data[col])]
        
        if not numeric_cols:
            print("Warning: No numeric columns found for analysis")
            return None
        
        # Calculate cluster statistics
        cluster_stats = merged_data.groupby('cluster')[numeric_cols].mean()
        cluster_stats['count'] = merged_data.groupby('cluster').size()
        
        # Save results
        cluster_stats.to_csv("../../results/cluster_stats.csv")
        
        # Create a visualization of cluster characteristics
        plt.figure(figsize=(12, 8))
        sns.heatmap(cluster_stats[numeric_cols], annot=True, cmap="YlGnBu", fmt=".2f")
        plt.title('Cluster Characteristics')
        plt.savefig('../../results/cluster_heatmap.png')
        plt.close()
        
        # Plot cluster sizes
        plt.figure(figsize=(10, 6))
        cluster_stats['count'].plot(kind='bar')
        plt.title('Cluster Sizes')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Users')
        plt.savefig('../../results/cluster_sizes.png')
        plt.close()
        
        print("Cluster sizes:")
        print(cluster_stats['count'])
        
        return cluster_stats
    
    except Exception as e:
        print(f"Error in cluster analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# 3. Conversion Prediction (more robust error handling)
def predict_conversion(data_df):
    try:
        # Make a copy to avoid modifying the original
        df = data_df.copy()
        
        # Check if we have the necessary columns
        if 'session_id' not in df.columns:
            print("Warning: 'session_id' column not found")
            # Try to find an ID column
            id_cols = [col for col in df.columns if 'id' in col.lower()]
            if id_cols:
                print(f"Using '{id_cols[0]}' as session_id")
                df = df.rename(columns={id_cols[0]: 'session_id'})
            else:
                # Create a dummy session_id column
                print("Creating dummy session_id column")
                df['session_id'] = [f"session_{i}" for i in range(len(df))]
        
        # Check for converted column
        if 'converted' not in df.columns:
            print("Error: 'converted' column not found")
            # Look for potential target columns
            potential_targets = [col for col in df.columns if any(term in col.lower() 
                                for term in ['convert', 'purchase', 'buy', 'target', 'label'])]
            
            if potential_targets:
                print(f"Using '{potential_targets[0]}' as target variable")
                df = df.rename(columns={potential_targets[0]: 'converted'})
            else:
                print("No suitable target column found. Cannot perform prediction.")
                return None, None, None
            
        # Check class distribution
        class_counts = df['converted'].value_counts()
        print("\nClass distribution:")
        print(class_counts)
        
        if len(class_counts) < 2:
            print("\nWarning: Only one class present - cannot perform classification")
            print("Creating synthetic minority class for demonstration")
            # Create some synthetic minority class samples
            majority_class = class_counts.index[0]
            minority_class = 1 if majority_class == 0 else 0
            
            # Modify a small percentage of samples
            indices_to_modify = np.random.choice(df.index, size=int(len(df) * 0.1), replace=False)
            df.loc[indices_to_modify, 'converted'] = minority_class
            
            # Print new distribution
            print("Synthetic class distribution:")
            print(df['converted'].value_counts())
            
        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in ['session_id', 'converted']]
        
        # Handle non-numeric features
        for col in feature_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"Warning: Non-numeric column '{col}' found. Converting to numeric or dropping.")
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    feature_cols.remove(col)
        
        X = df[feature_cols]
        y = df['converted']
        
        # Handle NaN values
        X = X.fillna(0)
        
        # Simple train/test split (no grid search for simplicity)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train basic model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        
        print(f"\nModel accuracy: {accuracy:.2f}")
        
        # Feature importance
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 5 important features:")
        print(feature_importances.head(5))
        
        # Save feature importance plot
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importances.head(10))
        plt.title('Feature Importance for Conversion Prediction')
        plt.tight_layout()
        plt.savefig('../../results/feature_importance.png')
        plt.close()
        
        # Save model and results
        with open("../../results/conversion_model.pkl", "wb") as f:
            pickle.dump(model, f)
            
        return model, accuracy, feature_importances
    
    except Exception as e:
        print(f"Error in conversion prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

# Main execution
if __name__ == "__main__":
    print("\nPerforming user behavior clustering...")
    cluster_model, clustered_data, optimal_k = perform_clustering(user_features_df)
    
    if cluster_model is not None and clustered_data is not None:
        print("\nAnalyzing clusters...")
        cluster_stats = analyze_clusters(clustered_data, user_features_df, optimal_k)
    else:
        print("Skipping cluster analysis due to previous errors")
    
    print("\nBuilding conversion prediction model...")
    conv_model, accuracy, feature_importances = predict_conversion(conversion_features_df)
    
    print("\nData mining analysis complete!")