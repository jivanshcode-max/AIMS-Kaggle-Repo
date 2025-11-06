import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Read the data 
data = pd.read_csv('melb_data.csv') 
feature_columns = ['Suburb','Rooms','Type','Price','Bedroom2','Bathroom','Landsize','BuildingArea','YearBuilt','CouncilArea','Regionname',]
X = data[feature_columns].copy()

# Remove rows with missing target, separate target from predictors  
X = X.dropna(axis=0, subset=['Price'])
y = X.Price 
X.drop(['Price'], axis=1, inplace=True) 

# To keep things simple, we'll drop columns with missing values (Can be solved using Imputation)
cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
X.drop(cols_with_missing, axis=1, inplace=True) 

object_cols = [col for col in X.columns if X[col].dtype == "object" ]
# ordinal_encoder = OrdinalEncoder()
# X_Ord = X.copy()
# X_Ord[object_cols] = ordinal_encoder.fit_transform(X[object_cols])

# We'll use ordinal encoding only for type and RegionName
# Suppose your dataset has an 'Education' column
encode_cols = ['Type', 'Regionname']

type_code = {
    'h': 1,
    'u': 2,
    't': 3
}
X['type_encoded'] = X['Type'].map(type_code)

Region_Code = {
    'Western Metropolitan': 1, 
    'Northern Metropolitan': 2, 
    'Southern Metropolitan': 3, 
    'South-Eastern Metropolitan': 4, 
    'Eastern Metropolitan': 5,
    'Northern Victoria': 6, 
    'Eastern Victoria': 7, 
    'Western Victoria': 8
}
X['region_encoded'] = X['Regionname'].map(Region_Code)

# drop the all categorical columns
X.drop(object_cols, axis=1, inplace=True)

# Break off validation set from training data 
X_train, X_valid, y_train, y_valid = train_test_split(X, y,train_size=0.85, test_size=0.15,random_state=0) 

model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)
preds = model.predict(X_valid)