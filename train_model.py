import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor



data = pd.read_csv('data/merc.csv')
X = data.drop('price', axis=1)
y = data['price']

# Identifies categorical and number columns
cats_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'Float64']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cats_cols),
        ('num', StandardScaler(), num_cols)
    ]
)


models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'SVR': SVR(),
    'KNN': KNeighborsRegressor(),
}

pipelines = {
    name: Pipeline([
        ('preprocessing', preprocessor),
        ('model', model)
    ])
    for name, model in models.items()
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = []
best_score = float('-inf')
best_pipeline = None
best_model_name =''

for name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Cross validation
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
    avg_cv_r2 = np.mean(cv_scores)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        'Model': name,
        'MAE': round(mae, 2),
        'RMSE': round(rmse,2),
        'R2 Score': round(r2, 2),
        'CV R2': round(avg_cv_r2, 2)
    })

    if avg_cv_r2 > best_score:
        best_score = avg_cv_r2
        best_pipeline = pipeline
        best_model_name = name

# Convert the result into a Dataframe
comparison_df = pd.DataFrame(results).sort_values(by='CV R2', ascending=False)



# Saves the best model
print(f"Best model is: {best_model_name} with CV R2: {best_score:.2F}")
joblib.dump(best_pipeline, 'best_pipeline_model.pkl')
print('Pipeline saved successfully')





