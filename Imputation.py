import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# Read the data
X = pd.read_csv('melb_data.csv').copy()

# Remove rows with missing target, separate target from predictors
X = X.dropna(axis=0, subset=['Price'])
y = X.Price
X.drop(['Price'], axis=1, inplace=True)

# To keep things simple, we'll use only numerical predictors
X = X.select_dtypes(exclude=['object'])

my_Imputer = SimpleImputer() # Your code here
X_imputed= pd.DataFrame(my_Imputer.fit_transform(X))
X_imputed.columns = X.columns

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X_imputed, y, train_size=0.85, test_size=0.15,random_state=0)

model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)
preds = model.predict(X_valid)
