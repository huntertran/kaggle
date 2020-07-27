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

# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = './home-data-for-ml-course/train.csv'
features_file_path = './home-data-for-ml-course/features.csv'

home_data = pd.read_csv(iowa_file_path)
dataFrame = pd.DataFrame(home_data)

meta_features = pd.read_csv(features_file_path)
meta_features_dataframe = pd.DataFrame(meta_features)

# Create target object and call it y
y = home_data.SalePrice

le = LabelEncoder()

features_dataframe = pd.read_csv(features_file_path)

raw_features = []
features = []

for meta_feature in meta_features_dataframe.iterrows():
    if meta_feature[1]['is_include'] == 1:
        raw_features.append(meta_feature[1]["feature_name"])

# transform category data
encoder = LabelEncoder()

for feature in raw_features:
    if dataFrame[feature].dtypes is not int:
        # category data
        new_feature_name = feature + "_encoded"
        home_data[new_feature_name] = encoder.fit_transform(
            home_data[feature].astype(str))
        features.append(new_feature_name)

# Create X
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
print(
    "Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# Using best value for max_leaf_nodes
iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print(
    "Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

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
test_data_path = './home-data-for-ml-course/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)

for feature in raw_features:
    # print("Column: " + feature + " | " + "Data type: " + str(dataFrame[feature].dtypes))
    if test_data[feature].dtypes is not int:
        # category data
        new_feature_name = feature + "_encoded"
        test_data[new_feature_name] = encoder.fit_transform(
            test_data[feature].astype(str))

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