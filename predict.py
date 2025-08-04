import sys
import pandas as pd
import pickle
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import convert_truth_labels, data_filtering_for_features, one_hot_encoding

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv)) 

#in an ideal world this would validated
model = sys.argv[1]
X_test_path = sys.argv[2]
y_test_path = sys.argv[3]


#model_path = 'models/max_voting_classifier.sav'
#X_test_path = 'data/X_test.csv'
#y_test_path = 'data/y_test.csv'

# load the model from disk
loaded_model = pickle.load(open(model, 'rb'))
X_test = pd.read_csv(X_test_path)
y_test_df = pd.read_csv(y_test_path)
y_test = y_test_df["bank_account"]


#feature eng on test data
print("Feature engineering")
y_test = convert_truth_labels(y_test)
X_test = data_filtering_for_features(X_test)
X_test = one_hot_encoding(X_test)


y_test_pred = loaded_model.predict(X_test)
classification_report_test = classification_report(y_test, y_test_pred)
print(f"Model performance on test data:")
print(classification_report_test)
