import pandas as pd
import numpy as np
from datetime import datetime

# Load the dataset
df = pd.read_csv('executable_files/delivery_five_cities_tanzania.csv')

print("=" * 50)
print("COMPREHENSIVE DATASET ANALYSIS")
print("=" * 50)

print('\n=== DATASET OVERVIEW ===')
print(f'Dataset shape: {df.shape}')
print(f'Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB')

print('\n=== MISSING VALUES ANALYSIS ===')
missing_values = df.isnull().sum()
print(missing_values)
print(f'Total missing values: {missing_values.sum()}')

print('\n=== CITIES DISTRIBUTION ===')
cities_count = df['from_city_name'].value_counts()
print(cities_count)
print(f'Total cities: {len(cities_count)}')

# Calculate percentages
cities_percentage = (cities_count / len(df) * 100).round(2)
print('\nCities percentage distribution:')
for city, pct in cities_percentage.items():
    print(f'{city}: {pct}%')

print('\n=== UNIQUE VALUES IN KEY COLUMNS ===')
print(f'Unique orders: {df["order_id"].nunique():,}')
print(f'Unique delivery users: {df["delivery_user_id"].nunique():,}')
print(f'Unique from_dipan_id: {df["from_dipan_id"].nunique():,}')
print(f'Unique type codes: {df["typecode"].nunique():,}')
print(f'Unique AOI IDs: {df["aoi_id"].nunique():,}')

print('\n=== TIME COLUMNS ANALYSIS ===')
print('Receipt time samples:')
print(df['receipt_time'].head(10).tolist())
print('\nSign time samples:')
print(df['sign_time'].head(10).tolist())

print('\n=== GEOGRAPHIC COORDINATES ANALYSIS ===')
print('POI (Point of Interest) coordinates:')
print(f'Longitude range: {df["poi_lng"].min():.6f} to {df["poi_lng"].max():.6f}')
print(f'Latitude range: {df["poi_lat"].min():.6f} to {df["poi_lat"].max():.6f}')

print('\nReceipt coordinates:')
print(f'Longitude range: {df["receipt_lng"].min():.6f} to {df["receipt_lng"].max():.6f}')
print(f'Latitude range: {df["receipt_lat"].min():.6f} to {df["receipt_lat"].max():.6f}')

print('\nSign coordinates:')
print(f'Longitude range: {df["sign_lng"].min():.6f} to {df["sign_lng"].max():.6f}')
print(f'Latitude range: {df["sign_lat"].min():.6f} to {df["sign_lat"].max():.6f}')

print('\n=== TYPE CODES ANALYSIS ===')
type_codes = df['typecode'].value_counts()
print(f'Number of unique type codes: {len(type_codes)}')
print('Top 10 type codes:')
print(type_codes.head(10))

print('\n=== DATA QUALITY INSIGHTS ===')
# Check for duplicates
print(f'Duplicate orders: {df["order_id"].duplicated().sum():,}')
print(f'Total unique orders vs total records: {df["order_id"].nunique():,} vs {len(df):,}')

# Check DS column
print(f'\nDS column unique values: {df["ds"].nunique()}')
print(f'DS column values: {df["ds"].unique()}')

print('\n=== BUSINESS INTELLIGENCE INSIGHTS ===')
print('1. OPERATIONAL SCALE:')
print(f'   - Processing {len(df):,} delivery records')
print(f'   - Across {len(cities_count)} major Tanzanian cities')
print(f'   - Involving {df["delivery_user_id"].nunique():,} delivery personnel')
print(f'   - From {df["from_dipan_id"].nunique():,} dispatch locations')

print('\n2. GEOGRAPHIC COVERAGE:')
for city, count in cities_count.items():
    pct = count / len(df) * 100
    print(f'   - {city}: {count:,} deliveries ({pct:.1f}%)')

print('\n3. DATA COMPLETENESS:')
if missing_values.sum() == 0:
    print('   - Excellent: No missing values in any column')
else:
    print(f'   - Missing values detected: {missing_values.sum():,} total')

print('\n4. DELIVERY NETWORK COMPLEXITY:')
print(f'   - {df["typecode"].nunique():,} different delivery types/categories')
print(f'   - {df["aoi_id"].nunique():,} unique area of interest locations')
print(f'   - Geographic span across Tanzania (lat: {df["poi_lat"].min():.2f} to {df["poi_lat"].max():.2f})')

# Calculate approximate distances for insights
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points using Haversine formula"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 6371  # Earth's radius in km

# Sample calculation for first 1000 records to avoid memory issues
sample_df = df.head(1000).copy()
sample_df['delivery_distance_km'] = haversine_distance(
    sample_df['poi_lat'], sample_df['poi_lng'],
    sample_df['sign_lat'], sample_df['sign_lng']
)

print(f'\n5. DELIVERY DISTANCES (Sample Analysis):')
print(f'   - Average distance: {sample_df["delivery_distance_km"].mean():.2f} km')
print(f'   - Min distance: {sample_df["delivery_distance_km"].min():.2f} km')
print(f'   - Max distance: {sample_df["delivery_distance_km"].max():.2f} km')

print('\n=== COLUMN DESCRIPTIONS ===')
column_descriptions = {
    'order_id': 'Unique identifier for each delivery order',
    'from_dipan_id': 'Dispatch location/warehouse identifier',
    'from_city_name': 'Origin city for the delivery',
    'delivery_user_id': 'Unique identifier for delivery personnel',
    'poi_lng/poi_lat': 'Point of Interest coordinates (pickup location)',
    'aoi_id': 'Area of Interest identifier',
    'typecode': 'Delivery type/category code',
    'receipt_time': 'When the package was received/picked up',
    'receipt_lng/receipt_lat': 'Coordinates where package was received',
    'sign_time': 'When the delivery was signed/completed',
    'sign_lng/sign_lat': 'Coordinates where delivery was completed',
    'ds': 'Day sequence or date identifier (value: 318)'
}

for col, desc in column_descriptions.items():
    print(f'{col}: {desc}')

print("\n" + "=" * 50)
print("ANALYSIS COMPLETE")
print("=" * 50)