from datetime import datetime

import sys
import math

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import _pickle as cPickle
import copy

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection._split import train_test_split

import warnings
from itertools import combinations
import getLimeExplainer
warnings.filterwarnings("ignore")

def prepare_df(df, datasetname, class_name, replacing_mv=True, reorder=True):

    # set all columns of dataframe
    columns = df.columns.tolist()

    columns_for_decision_rules = list(columns)
    columns_for_decision_rules.remove(class_name)

    if reorder:
        columns = columns[-1:] + columns[:-1]

    # 1. Step: set type for each feature/column of dataset
    type_features, features_type = recognize_features_type(df)

    continuous = type_features['integer'] + type_features['double']
    discrete = type_features['string']

    # discrete columns include always class_name column!
    # discrete, continuous = set_discrete_continuous(
    #     columns, type_features, class_name, discrete=None, continuous=None)

    # 2. Step: replace missing values of continuous and discrete features
    if replacing_mv:
        df = replace_by_median(df, continuous)
        df = replace_by_most_used_value(df, discrete)

    tmp_columns = list(columns)
    tmp_columns.remove(class_name)

    possible_outcomes = list(df[class_name].unique())

    # set index for features
    idx_features = {i: col for i, col in enumerate(
        list(tmp_columns))}

    # 3. Step: hot_encoding of all discrete features, except the to predicted column
    # df = pd.get_dummies(
    #    df, columns=[col for col in discrete if col != class_name])

    # 3. Step: label encoding of the discrete features
    df_le, label_encoder = label_encode(df, discrete)

    # 4. Step: extract features and class label
    X = df_le.loc[:, df_le.columns != class_name].values
    y = df_le[class_name].values

    # 5. Step: preprare dataset dict
    dataset = {
        'name': datasetname,
        'df': df,
        'df_encoded': df_le,
        'columns': list(columns),
        'columns_for_decision_rules': list(columns_for_decision_rules),
        'class_name': class_name,
        'possible_outcomes': possible_outcomes,
        'type_features': type_features,
        'features_type': features_type,
        'discrete': discrete,
        'continuous': continuous,
        'idx_features': idx_features,
        'label_encoder': label_encoder,
        'X': X,
        'y': y,
    }

    return dataset

# replace missing values of features with median of column
def replace_by_most_used_value(df, features):
    for col in features:
        most_used_value = df[col].value_counts().index[0]
        df[col].fillna(most_used_value, inplace=True)
    return df

# replace missing value of feature with most used value of column
def replace_by_median(df, features):
    for col in features:
        median = df[col].median()
        df[col].fillna(median, inplace=True)
    return df

# recognice different types (integer, double, string) of columns in dataframe.
def recognize_features_type(df):
    integer_features = list(df.select_dtypes(include=['int64']).columns)
    double_features = list(df.select_dtypes(include=['float64']).columns)
    string_features = list(df.select_dtypes(include=['object']).columns)
    type_features = {
        'integer': integer_features,
        'double': double_features,
        'string': string_features,
    }

    features_type = dict()
    for col in integer_features:
        features_type[col] = 'integer'
    for col in double_features:
        features_type[col] = 'double'
    for col in string_features:
        features_type[col] = 'string'

    return type_features, features_type

def set_discrete_continuous(features, type_features, class_name, discrete=None, continuous=None):

    if discrete is None and continuous is None:
        discrete = type_features['string']
        continuous = type_features['integer'] + type_features['double']

    if discrete is None and continuous is not None:
        discrete = [f for f in features if f not in continuous]
        continuous = list(
            set(continuous + type_features['integer'] + type_features['double']))

    if continuous is None and discrete is not None:
        continuous = [f for f in features if f not in discrete and (
            f in type_features['integer'] or f in type_features['double'])]
        discrete = list(set(discrete + type_features['string']))

    discrete = [f for f in discrete if f != class_name] + [class_name]
    continuous = [f for f in continuous if f != class_name]
    return discrete, continuous

def label_encode(df, columns, label_encoder=None):
    df_le = df.copy(deep=True)
    new_le = label_encoder is None
    label_encoder = dict() if new_le else label_encoder
    for col in columns:
        if new_le:
            le = LabelEncoder()
            df_le[col] = le.fit_transform(df_le[col])
            label_encoder[col] = le
        else:
            le = label_encoder[col]
            df_le[col] = le.transform(df_le[col])
    return df_le, label_encoder

def label_decode(df, columns, label_encoder, not_including_columns=[]):
    df_de = df.copy(deep=True)
    for col in columns:
        if col not in not_including_columns:
            le = label_encoder[col]
            df_de[col] = le.inverse_transform(df_de[col])
    return df_de

def dataframe2explain(X2E, dataset, idx_record2explain, blackbox):
    # Dataset to explit to perform explanation (typically is the train or test set (real instances))
    Z = cPickle.loads(cPickle.dumps(X2E))

    # Select record to predict and explain
    x = Z[idx_record2explain]

    # Remove record to explain (optional) from dataset Z and convert into dataframe
    # Z = np.delete(Z, idx_record2explain, axis=0)
    dfZ = build_df2explain(blackbox, Z, dataset)

    return dfZ, x

def build_df2explain(bb, X, dataset):

    columns = dataset['columns']
    features_type = dataset['features_type']
    discrete = dataset['discrete']
    label_encoder = dataset['label_encoder']

    y = bb.predict(X)
    yX = np.concatenate((y.reshape(-1, 1), X), axis=1)
    data = list()
    for i, col in enumerate(columns):
        data_col = yX[:, i]
        data_col = data_col.astype(int) if col in discrete else data_col
        data_col = data_col.astype(
            int) if features_type[col] == 'integer' else data_col
        data.append(data_col)
    # data = map(list, map(None, *data))
    data = [[d[i] for d in data] for i in range(0, len(data[0]))]
    dfZ = pd.DataFrame(data=data, columns=columns)
    dfZ = label_decode(dfZ, discrete, label_encoder)
    return dfZ

def manipulate_adult_hours_per_week(X):
    man_X = copy.deepcopy(X)
    for x0 in man_X:
        x0[10] = x0[10] + 10

    return man_X

# private
# preprocessing of adult dataset
def _preprocess_adult_dataset(datapath, filename):

    # 1. Step: read adult csv and mark missing values
    missing_values = ['?']
    df = pd.read_csv(datapath + filename, delimiter=',',
                     skipinitialspace=True, na_values=missing_values)

    # and remove useless columns
    del df['fnlwgt']
    del df['education-num']

    # preparing dataframe to dataset dict
    dataset = prepare_df(df, filename.replace('.csv', ''), 'income_class')

    return dataset

def get_blackboxes_and_testdata_manipulated_dataset(dataset):
    X1, y1 = dataset['X'], dataset['y']
    X1_train, X1_test, y1_train, y1_test = train_test_split(
        X1, y1, test_size=0.2, random_state=0)

    blackbox1 = DecisionTreeClassifier(random_state=0, max_depth=6)
    blackbox1.fit(X1_train, y1_train)

    X2 = manipulate_adult_hours_per_week(X1)
    y2 = copy.deepcopy(y1)

    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y2, test_size=0.2, random_state=0)

    blackbox2 = DecisionTreeClassifier(random_state=0, max_depth=6)
    blackbox2.fit(X2_train, y2_train)
    return blackbox1, blackbox2, X1_test

def get_blackboxes_and_testdata_different_maxdepth(dataset):
    X1, y1 = dataset['X'], dataset['y']
    X1_train, X1_test, y1_train, y1_test = train_test_split(
        X1, y1, test_size=0.5, random_state=0)

    blackbox1 = DecisionTreeClassifier(random_state=0, max_depth=6)
    blackbox1.fit(X1_train, y1_train)

    blackbox2 = DecisionTreeClassifier(random_state=0, max_depth=8)
    blackbox2.fit(X1_train, y1_train)
    return blackbox1, blackbox2, X1_test

def get_test_and_train_data(dataset, bb1, bb2):
    X, y = dataset['X'], dataset['y']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0)
    y_train = bb1.predict(X_train) * 2 + bb2.predict(X_train)
    y_test = bb1.predict(X_test) * 2 + bb2.predict(X_test)
    return X_train, y_train, X_test, y_test

dataset_name = 'adult.csv'
path_data = './data/adult/'
dataset = _preprocess_adult_dataset(path_data, dataset_name)
FEATURE_NAMES = [c for c in dataset['df_encoded'].columns if c != dataset['class_name']]

blackbox1, blackbox2, testdata = get_blackboxes_and_testdata_different_maxdepth(dataset)

blackbox1_predictions = blackbox1.predict(testdata)
blackbox2_predictions = blackbox2.predict(testdata)

result = np.array(blackbox1_predictions) * 2 + np.array(blackbox2_predictions)

result_counts = np.unique(result, return_counts=True)
print("Result counts:", result_counts)

#get test and train data for surrogate
traindata_X, traindata_Y, testdata_X, testdata_Y = get_test_and_train_data(dataset, blackbox1, blackbox2)

#get neighborhoods for a bunch of sigmas
global_sigma = 0.5
temp_sigma = global_sigma
neighborsDict = {}

while temp_sigma <= global_sigma * 10:
    temp_ngbrs = NearestNeighbors(radius=temp_sigma)
    temp_ngbrs.fit(traindata_X)
    neighborsDict[temp_sigma] = temp_ngbrs
    temp_sigma = math.ceil((temp_sigma + 0.1)* 10) / 10   #avoid floating point precision issues

#do some stuff
predictions, confidences, models_used = getLimeExplainer.local_predict_moredimensional(testdata_X, sigma=0.5, min_neighbors=20, max_sigma=1.0, 
                              neighbors_dict=neighborsDict, X_local=traindata_X, Y_local=traindata_Y)

#check accuracy of predictions
#number of times each permutation of actual data (0, 1, 2, 3) and prediction (0, 1, 2, 3) appear
accuracyDict = {
    "00": [],
    "11": [],
    "22": [],
    "33": [],
    "01": [],
    "02": [],
    "03": [],
    "10": [],
    "12": [],
    "13": [],
    "20": [],
    "21": [],
    "23": [],
    "30": [],
    "31": [],
    "32": [],
}

for i in range(predictions.size):
    surrPredict = predictions[i]
    actualPredict = testdata_Y[i]
    key = f"{actualPredict}{surrPredict}" 
    accuracyDict[key].append(i)

total_correct = 0
total_wrong = 0

for key in accuracyDict:
    size = len(accuracyDict[key])
    print(f"<{key}>: {size}")
    if (key in ["00","11","22","33"]):
        total_correct += size
    else:
        total_wrong += size

TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H%M%S")
orig_stdout = sys.stdout
f = open(f'docout/sections/localtoglobal/resultsAdult/out_{TIMESTAMP}.txt', 'w+')
sys.stdout = f

print(f"Totals: ")
print(f"Correct: {total_correct}")
print(f"Wrong: {total_wrong}")
print(f"Accuracy: {((total_correct / (total_correct + total_wrong)) * 100):.2f}%")
print(f"Average Confidence: {np.mean(confidences):.2f}")

PLOT_POINTS = [42, 61, 72, 121, 168, 2781, 7898, 16254]
CLASS_NAMES = {0: "0|0", 1: "0|1", 2: "1|0", 3: "1|1"}
OVERVIEW_GRID = (4,4)

for cnt, model_info in models_used.items():
    model = model_info['model']
    x_local = model_info['x_local']
    formulas, feature_ranges = getLimeExplainer.extract_multinomial_ranges_universal_moredimensional(model, x_local)
    print(f"Model for point nr. {cnt}:")
    for key, formula in formulas.items():
        print(f"Formula {CLASS_NAMES[key[0]]} vs {CLASS_NAMES[key[1]]}: {formula}")
    print("Feature Ranges:")
    for feature_idx, (min_val, max_val) in feature_ranges.items():
        print(f"Feature {feature_idx}: Min = {min_val}, Max = {max_val}")

sys.stdout = orig_stdout
f.close()
print(f"Results written to docout/sections/localtoglobal/resultsAdult/out_{TIMESTAMP}.txt")
print("generating plots for selected points...")

for cnt in PLOT_POINTS:
    if cnt not in models_used:
        print(f"Point {cnt} not found in models_used, skipping plot.")
        continue
    model_info = models_used[cnt]
    info = getLimeExplainer.get_local_pca_boundaries(model_info['model'], model_info['x_local'], model_info['x_query'])
    fig, ax = plt.subplots(figsize=(9, 6))
    getLimeExplainer.plot_local_pca_boundaries(ax, 
                                               info, 
                                               model_info['y_local'], 
                                               CLASS_NAMES, 
                                               f"Local PCA Boundaries for Point {cnt}",
                                               feature_names=FEATURE_NAMES)
    fig.savefig(f'docout/sections/localtoglobal/resultsAdult/pca_plot_point_{cnt}_at_{TIMESTAMP}.png', dpi=150, bbox_inches="tight")
    plt.close(fig)
print(f"Plots saved to docout/sections/localtoglobal/resultsAdult/pca_plot_point_<cnt>_at_{TIMESTAMP}.png")