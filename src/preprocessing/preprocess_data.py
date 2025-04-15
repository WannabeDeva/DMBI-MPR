import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Load data
df = pd.read_csv("../../data/raw/clickstream_data.csv")

# Display basic information
print("Dataset Shape:", df.shape)
print(df.dtypes)
print(df.head(5))

# Summary statistics
print(df.describe())

# Check for missing values
for col in df.columns:
    missing_count = df[col].isnull().sum()
    if missing_count > 0:
        print(f"Column {col} has {missing_count} missing values")

# Basic visualizations
# Time distribution
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day
df['day_of_week'] = df['timestamp'].dt.dayofweek

plt.figure(figsize=(12, 6))
sns.countplot(x='hour', data=df)
plt.title('Distribution of Events by Hour')
plt.savefig('../../results/events_by_hour.png')

# Event distribution
plt.figure(figsize=(10, 6))
sns.countplot(y='event_type', data=df)
plt.title('Distribution of Event Types')
plt.savefig('../../results/event_distribution.png')

# Page category distribution
plt.figure(figsize=(10, 6))
sns.countplot(y='page_category', data=df)
plt.title('Distribution of Page Categories')
plt.savefig('../../results/page_category_distribution.png')

# Device distribution
plt.figure(figsize=(10, 6))
df['device_type'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribution of Device Types')
plt.savefig('../../results/device_distribution.png')

# Preprocess data
def preprocess_data(df):
    # Make a copy to avoid modifying the original
    processed = df.copy()
    
    # 1. Convert timestamp to datetime features
    processed['timestamp'] = pd.to_datetime(processed['timestamp'])
    processed['hour'] = processed['timestamp'].dt.hour
    processed['day'] = processed['timestamp'].dt.day
    processed['day_of_week'] = processed['timestamp'].dt.dayofweek
    processed['is_weekend'] = processed['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # 2. Cleanup and standardize categorical variables
    for col_name in ['page_category', 'event_type', 'device_type', 'referrer', 'browser']:
        if col_name in processed.columns:
            processed[col_name] = processed[col_name].str.lower()
    
    # 3. Extract domain from page_url
    import re
    processed['page_domain'] = processed['page_url'].str.extract(r'^/([^/]+)')
    
    # 4. Create session-level aggregations
    session_aggs = processed.groupby('session_id').agg(
        session_event_count=('session_id', 'count'),
        pageview_count=('event_type', lambda x: (x == 'page_view').sum()),
        click_count=('event_type', lambda x: (x == 'click').sum()),
        add_to_cart_count=('event_type', lambda x: (x == 'add_to_cart').sum()),
        purchase_count=('event_type', lambda x: (x == 'purchase').sum()),
        avg_time_spent=('time_spent', 'mean'),
        session_start=('timestamp', 'min'),
        session_end=('timestamp', 'max')
    ).reset_index()
    
    # Calculate session duration
    session_aggs['session_duration_minutes'] = (session_aggs['session_end'] - session_aggs['session_start']).dt.total_seconds() / 60
    
    # Join back to original data
    processed = pd.merge(processed, session_aggs, on='session_id', how='left')
    
    # 5. Feature engineering - create flags for user behavior
    processed['is_returning_user'] = processed['session_event_count'].apply(lambda x: 1 if x > 1 else 0)
    
    # Flag bounces (sessions with only one pageview)
    processed['is_bounce'] = processed['pageview_count'].apply(lambda x: 1 if x == 1 else 0)
    
    # 6. Handle any missing values
    # For numeric columns
    numeric_cols = processed.select_dtypes(include=['int64', 'float64']).columns
    processed[numeric_cols] = processed[numeric_cols].fillna(0)
    
    # For non-numeric columns
    non_numeric_cols = processed.select_dtypes(exclude=['int64', 'float64']).columns
    processed[non_numeric_cols] = processed[non_numeric_cols].fillna("unknown")
    
    return processed

# Apply preprocessing
processed_df = preprocess_data(df)
processed_df.to_parquet("../../data/processed/clickstream_processed.parquet")

# Show processed data sample
print(processed_df.head(5))