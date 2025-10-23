import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder 

# Read the data 
data = pd.read_csv('melb_data.csv') 
feature_columns = ['Suburb','Rooms','Type','Price','Bedroom2','Bathroom','Landsize','BuildingArea','YearBuilt','CouncilArea','Regionname',]
X = data[feature_columns].copy()

# Remove rows with missing target, separate target from predictors (Can be solved using Imputation) 
X = X.dropna(axis=0, subset=['Price'])
y = X.Price 
X.drop(['Price'], axis=1, inplace=True) 

# To keep things simple, we'll drop columns with missing values 
cols_with_missing = [col for col in X.columns if X[col].isnull().any()] 
X.drop(cols_with_missing, axis=1, inplace=True) 

object_cols = [col for col in X.columns if X[col].dtype == "object"]
ordinal_encoder = OrdinalEncoder()
X_label = X.copy()
X_label[object_cols] = ordinal_encoder.fit_transform(X[object_cols])

# Break off validation set from training data 
X_train, X_valid, y_train, y_valid = train_test_split(X_label, y,train_size=0.85, test_size=0.15,random_state=0) 

model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)
preds = model.predict(X_valid)