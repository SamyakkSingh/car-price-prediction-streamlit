# train_model.py (robust version)
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CURRENT_YEAR = 2025

# Load dataset
df = pd.read_csv('car_data.csv')
df.columns = [c.strip() for c in df.columns]
df['Age'] = CURRENT_YEAR - df['Year']

# Features & target
features = ['Present_Price','Driven_kms','Fuel_Type','Transmission','Owner','Age','Car_Name']
target = 'Selling_Price'

X = df[features].copy()
y = df[target].copy()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = ['Present_Price','Driven_kms','Age']
numeric_transformer = Pipeline([('scaler', StandardScaler())])

categorical_features = ['Fuel_Type','Transmission','Owner','Car_Name']

# Compatibility: older sklearn may use 'sparse' arg, newer use 'sparse_output'
try:
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
except TypeError:
    # fallback for older versions
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
], remainder='drop')

models = {
    'rf': RandomForestRegressor(n_estimators=200, random_state=42),
    'gbr': GradientBoostingRegressor(n_estimators=200, random_state=42)
}

results = {}
pipelines = {}
for name, est in models.items():
    pipe = Pipeline([('preprocessor', preprocessor), ('regressor', est)])
    print(f'Training {name}...')
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)     # always supported
    rmse = float(np.sqrt(mse))                  # robust RMSE
    r2 = r2_score(y_test, preds)

    results[name] = {'mae': mae, 'rmse': rmse, 'r2': r2}
    pipelines[name] = pipe
    print(f'{name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}')

best_name = max(results.keys(), key=lambda k: results[k]['r2'])
best_pipeline = pipelines[best_name]
print(f'Best model: {best_name} -> saving car_price_model_user.joblib')
joblib.dump(best_pipeline, 'car_price_model_user.joblib')
