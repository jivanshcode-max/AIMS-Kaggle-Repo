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

# my_Imputer = SimpleImputer() # Your code here
# X_imputed= pd.DataFrame(my_Imputer.fit_transform(X))
# X_imputed.columns = X.columns

# Perform imputation using pandas
# Seperate numerical and categorical columns from each other
numeric_cols = [col for col in X.columns if X[col].dtype != "object"]
category_cols = [col for col in X.columns if X[col].dtype == "object"]

# We'll fill missing values with mean for numerical columns
for col in numeric_cols:
        X[col].fillna(X[col].mean(), inplace=True)

# We'll fill missing values with mode for categorical columns
for col in category_cols:
    X[col].fillna(X[col].mode()[0], inplace=True)

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.85, test_size=0.15,random_state=0)

model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)
preds = model.predict(X_valid)
