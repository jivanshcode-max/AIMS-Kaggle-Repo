import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder 

# Read the data 
data = pd.read_csv('melb_data.csv') 
feature_columns = ['Suburb','Rooms','Type','Price','Bedroom2','Bathroom','Landsize','BuildingArea','YearBuilt','CouncilArea','Regionname',]
X = data[feature_columns].copy()

# Remove rows with missing target, separate target from predictors  
X = X.dropna(axis=0, subset=['Price'])
y = X.Price 
X.drop(['Price'], axis=1, inplace=True) 

print(X['Regionname'].unique().tolist())
