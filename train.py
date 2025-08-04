import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')
from feature_engineering import convert_truth_labels, data_filtering_for_features, one_hot_encoding

RSEED=42


# loading train data
print("Loading train data")
train_data = pd.read_csv("data/financial-inclusion-in-africa20250311-22142-nbnoiv/Train.csv")


# cleaning data and preparing
print("Preparing train data")
train_data = train_data.query("marital_status != 'Dont know'")   # otherwise we will have problems later with one-hot-encoding 
                                                                 # and using X_test and X_train
y = train_data.bank_account
X = train_data.drop("bank_account", axis=1)



# splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=RSEED)


## in order to exemplify how the predict will work.. we will save the y_train
print("Saving test data in the data folder")
X_test.to_csv("data/X_test.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)



print("Feature engineering on train and test")
y_train = convert_truth_labels(y_train)
y_test = convert_truth_labels(y_test)
X_train = data_filtering_for_features(X_train)
X_test = data_filtering_for_features(X_test)
X_train = one_hot_encoding(X_train)
X_test = one_hot_encoding(X_test)



# model
print("Training max voting classifier")
model_lr = LogisticRegression(random_state = RSEED, class_weight="balanced")
model_xgb = XGBClassifier(random_state = RSEED, class_weight="balanced")
model_rf = RandomForestClassifier(random_state = RSEED, n_estimators = 100, class_weight="balanced", max_depth=10, min_samples_leaf=10)

model = VotingClassifier(estimators = [('lr', model_lr), ('xgb', model_xgb), ('rf', model_rf)], voting = 'soft') 
model.fit(X_train, y_train)

# model evaluation on train
train_predictions = model.predict(X_train)
classification_report_train = classification_report(y_train, train_predictions)


# model evaluation on test
test_predictions = model.predict(X_test)
classification_report_test = classification_report(y_test, test_predictions)

print("Model performance on train data:")
print(classification_report_train)
print("")
print("Model performance on test data:")
print(classification_report_test)


#saving the model
print("Saving model in the model folder")
filename = 'models/max_voting_classifier.sav'
pickle.dump(model, open(filename, 'wb'))




