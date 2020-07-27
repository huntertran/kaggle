# Code you have previously used to load data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

# Set up code checking
import os
if not os.path.exists("../input/train.csv"):
    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  
    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex7 import *

# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = '../input/train.csv'

home_data = pd.read_csv(iowa_file_path)

# Create target object and call it y
y = home_data.SalePrice

home_data.fillna(0)

le = LabelEncoder()

home_data['MSZoning_encoded'] = le.fit_transform(home_data['MSZoning'].astype(str))
# home_data['Street_encoded'] = le.fit_transform(home_data['Street'].astype(str))
# home_data['Alley_encoded'] = LabelEncoder().fit_transform(home_data['Alley'].astype(str))
# home_data['Utilities_encoded'] = le.fit_transform(home_data['Utilities'].astype(str))
home_data['LandSlope_encoded'] = le.fit_transform(home_data['LandSlope'].astype(str))
home_data['Neighborhood_encoded'] = le.fit_transform(home_data['Neighborhood'].astype(str))
home_data['Condition1_encoded'] = le.fit_transform(home_data['Condition1'].astype(str))
# home_data['Condition2_encoded'] = le.fit_transform(home_data['Condition2'].astype(str))
# home_data['BldgType_encoded'] = le.fit_transform(home_data['BldgType'].astype(str))
# home_data['HouseStyle_encoded'] = le.fit_transform(home_data['HouseStyle'].astype(str))
home_data['OverallQual_encoded'] = le.fit_transform(home_data['OverallQual'].astype(str))
home_data['OverallCond_encoded'] = le.fit_transform(home_data['OverallCond'].astype(str))
# home_data['RoofMatl_encoded'] = le.fit_transform(home_data['RoofMatl'].astype(str))
# home_data['ExterQual_encoded'] = le.fit_transform(home_data['ExterQual'].astype(str))
home_data['ExterCond_encoded'] = le.fit_transform(home_data['ExterCond'].astype(str))
# home_data['BsmtCond_encoded'] = le.fit_transform(home_data['BsmtCond'].astype(str))
# home_data['HeatingQC_encoded'] = le.fit_transform(home_data['HeatingQC'].astype(str))


# Create X
features = ['LotArea',
            'YearBuilt',
            '1stFlrSF',
            '2ndFlrSF',
            'FullBath',
            'BedroomAbvGr',
            'TotRmsAbvGrd',
            'MSZoning_encoded',
#             'LotFrontage',
#             'Street_encoded',
#             'Alley_encoded',
#             'Utilities_encoded',
            'LandSlope_encoded',
            'Neighborhood_encoded',
            'Condition1_encoded',
#             'Condition2_encoded',
#             'BldgType_encoded',
#             'HouseStyle_encoded',
            'OverallQual_encoded',
            'OverallCond_encoded',
#             'RoofMatl_encoded',
#             'ExterQual_encoded',
            'ExterCond_encoded',
#             'BsmtCond_encoded',
#             'HeatingQC_encoded',
#             '1stFlrSF',
#             '2ndFlrSF'
           ]
X = home_data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit Model
iowa_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# Using best value for max_leaf_nodes
iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = RandomForestRegressor(random_state=1)

# fit rf_model_on_full_data on all data from the training data
rf_model_on_full_data.fit(X, y)

# path to file you will use for predictions
test_data_path = '../input/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)

test_data['MSZoning_encoded'] = le.fit_transform(test_data['MSZoning'].astype(str))
test_data['LandSlope_encoded'] = le.fit_transform(test_data['LandSlope'].astype(str))
test_data['Neighborhood_encoded'] = le.fit_transform(test_data['Neighborhood'].astype(str))

test_data['Condition1_encoded'] = le.fit_transform(test_data['Condition1'].astype(str))
test_data['OverallQual_encoded'] = le.fit_transform(test_data['OverallQual'].astype(str))
test_data['OverallCond_encoded'] = le.fit_transform(test_data['OverallCond'].astype(str))
test_data['ExterCond_encoded'] = le.fit_transform(test_data['ExterCond'].astype(str))

test_data.fillna(0)

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
test_X = test_data[features]

# make predictions which we will submit. 
test_preds = rf_model_on_full_data.predict(test_X)

# The lines below shows how to save predictions in format used for competition scoring
# Just uncomment them.

output = pd.DataFrame({'Id': test_data.Id,
                      'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)