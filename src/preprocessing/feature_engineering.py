import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load processed data
df = pd.read_parquet("../../data/processed/clickstream_processed.parquet")

# 1. Select relevant features for different analysis tasks
# For user behavior clustering
user_behavior_features = [
    'user_id', 'session_event_count', 'pageview_count', 'click_count', 
    'add_to_cart_count', 'purchase_count', 'avg_time_spent', 
    'session_duration_minutes', 'is_bounce', 'is_weekend', 'hour',
    'device_type', 'referrer'
]

# For conversion prediction
conversion_features = [
    'session_id', 'user_id', 'pageview_count', 'click_count', 
    'add_to_cart_count', 'session_duration_minutes',
    'avg_time_spent', 'is_weekend', 'hour', 'device_type', 
    'referrer', 'page_category', 'is_returning_user'
]

# Create a label for conversion prediction
df['converted'] = df['purchase_count'].apply(lambda x: 1 if x > 0 else 0)

# 2. Aggregate data at user level for user behavior clustering
user_agg = df.groupby("user_id").agg(
    avg_events_per_session=('session_event_count', 'mean'),
    avg_pageviews_per_session=('pageview_count', 'mean'),
    avg_clicks_per_session=('click_count', 'mean'),
    avg_add_to_cart_per_session=('add_to_cart_count', 'mean'),
    total_purchases=('purchase_count', 'sum'),
    avg_time_spent=('avg_time_spent', 'mean'),
    avg_session_duration=('session_duration_minutes', 'mean'),
    bounce_rate=('is_bounce', 'mean'),
    session_count=('session_id', 'nunique')
).reset_index()

# Get most used device and referrer per user
device_counts = df.groupby(['user_id', 'device_type']).size().reset_index(name='count')
most_used_device = device_counts.sort_values(['user_id', 'count'], ascending=[True, False]).drop_duplicates('user_id')
most_used_device = most_used_device[['user_id', 'device_type']].rename(columns={'device_type': 'most_used_device'})

referrer_counts = df.groupby(['user_id', 'referrer']).size().reset_index(name='count')
most_common_referrer = referrer_counts.sort_values(['user_id', 'count'], ascending=[True, False]).drop_duplicates('user_id')
most_common_referrer = most_common_referrer[['user_id', 'referrer']].rename(columns={'referrer': 'most_common_referrer'})

# Merge the additional features
user_agg = user_agg.merge(most_used_device, on='user_id', how='left')
user_agg = user_agg.merge(most_common_referrer, on='user_id', how='left')

# 3. Prepare data for conversion prediction (session-level)
conversion_df = df[conversion_features + ['converted']].drop_duplicates('session_id')

# 4. Feature engineering for ML models
def prepare_features(df, categorical_cols, numeric_cols, id_col, label_col=None):
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Handle missing values
    df_copy[categorical_cols] = df_copy[categorical_cols].fillna('unknown')
    df_copy[numeric_cols] = df_copy[numeric_cols].fillna(0)
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    # Fit and transform
    features = preprocessor.fit_transform(df_copy[categorical_cols + numeric_cols])
    
    # Convert to dataframe if needed
    if isinstance(features, np.ndarray):
        # For sparse matrices from OneHotEncoder
        feature_df = pd.DataFrame(features)
    else:
        # For sparse matrices from OneHotEncoder
        feature_df = pd.DataFrame(features.toarray())
    
    # Add identifiers back
    feature_df[id_col] = df_copy[id_col].values
    
    # Add label if provided
    if label_col:
        feature_df[label_col] = df_copy[label_col].values
    
    return feature_df, preprocessor

# Define feature columns
categorical_cols_user = ["most_used_device", "most_common_referrer"]
numeric_cols_user = ["avg_events_per_session", "avg_pageviews_per_session", 
                    "avg_clicks_per_session", "avg_add_to_cart_per_session", 
                    "total_purchases", "avg_time_spent", "avg_session_duration", 
                    "bounce_rate", "session_count"]

categorical_cols_conversion = ["device_type", "referrer", "page_category"]
numeric_cols_conversion = ["pageview_count", "click_count", "add_to_cart_count", 
                          "session_duration_minutes", "avg_time_spent", 
                          "is_weekend", "hour", "is_returning_user"]

# Prepare features for clustering (user behavior)
user_features_df, user_pipeline = prepare_features(
    user_agg, 
    categorical_cols_user,
    numeric_cols_user,
    "user_id"
)

# Prepare features for conversion prediction
conversion_features_df, conversion_pipeline = prepare_features(
    conversion_df, 
    categorical_cols_conversion,
    numeric_cols_conversion,
    "session_id",
    "converted"
)

# Save processed features
user_features_df.to_parquet("../../data/processed/user_features.parquet")
conversion_features_df.to_parquet("../../data/processed/conversion_features.parquet")

print("Feature engineering complete.")