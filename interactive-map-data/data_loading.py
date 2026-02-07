import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. read CSV
csv_path = 'data/final_dataset_1070_wards.csv' 
df = pd.read_csv(csv_path)

# --- Calculate PCA Deprivation Index ---
dep_vars = ['RoutineOccup', 'LTU', 'SocialRent', 'UnempRate_EA', 'Deprived']


X = df[dep_vars].fillna(df[dep_vars].mean())


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


pca = PCA(n_components=1)
pc1 = pca.fit_transform(X_scaled)[:, 0]


corr = np.corrcoef(pc1, df['UnempRate_EA'])[0, 1]
pc1 = pc1 * -1


df['DeprivationIndex'] = pc1.round(2)


df = df[df['Leave'] >= 0]
df = df[df['NVotes'] > 0]
df['LeavePct'] = (df['Leave'] / df['NVotes'] * 100).round(1)
df['Age65Plus'] = (df['Age_65to74'] + df['Age_75to84'] + df['Age_85to89'] + df['Age_90andover']).round(1)


geojson_path = 'data/ward-2021.geojson' 
gdf = gpd.read_file(geojson_path)
merged_gdf = gdf.merge(df, left_on='WD21CD', right_on='ward_code', how='inner')


column_mapping = {
    'WD21CD': 'ward_code',
    'ward_name': 'ward_name',
    'LeavePct': 'leave_pct',
    'DeprivationIndex': 'deprivation_index', 
    'L4Quals_plus': 'degree_level',
    'Age65Plus': 'age_65_plus',
    'NVotes': 'total_votes',
    'geometry': 'geometry'
}

final_gdf = merged_gdf[list(column_mapping.keys())].copy()
final_gdf.rename(columns=column_mapping, inplace=True)


if final_gdf.crs is None or final_gdf.crs.to_string() != 'EPSG:4326':
    final_gdf = final_gdf.to_crs(epsg=4326)

output_path = 'data/merge_1070_pca.geojson'
final_gdf.to_file(output_path, driver='GeoJSON')

print(f"new file: {output_path}")
print("columns:", final_gdf.columns.tolist())