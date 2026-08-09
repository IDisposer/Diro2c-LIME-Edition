import sys
import logging
import getLimeExplainer
from diro2c.data_generation.helper import *
import pandas as pd
import numpy as np
from diro2c.data_generation.neighborhood_generation import modified_gpdatagenerator
from diro2c.data_generation.distance_functions import simple_match_distance, normalized_euclidean_distance, mixed_distance
import _pickle as cPickle
from diro2c.enums.diff_classifier_method_type import diff_classifier_method_type
from diro2c.data_generation.neighborhood_generation.gpdatagenerator import calculate_feature_values
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from data.getdata import loaddata, prepare_df
from data.split3fold import split3fold
from scipy.cluster.hierarchy import linkage, fcluster
import pickle
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from datetime import date
import os
import json
from lime.lime_tabular import LimeTabularExplainer
from FromLocalToGlobal import getglobal, getexplainer, clusterbasedinstances, stratifiedrandominstances, randominstances

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print("Usage: python fitFromLocalToGlobal.py <data> <method>")
        print("Example: python fitFromLocalToGlobal.py compas randominstances")
        print("Available methods: randominstances, classstratifiedinstances, clusterbasedinstances")
        raise SystemExit(1)

    data = sys.argv[1]
    methodin = sys.argv[2]

    if methodin == 'randominstances':
        method = randominstances
        methodname = '"Approach 1: Random sampling"'
        npointsin = [8,16,32,64]
        ts = [1]

    if data == 'compas':
        ts = [{'2|0': 2.5, '0|1': 7.5,'2|1': 5, '1|2': 2,'0|2': 4, '1|0': 5},
              {'2|0': 4, '0|1': 15,'2|1': 10, '1|2': 7,'0|2': 12, '1|0': 15},
              {'2|0': 4, '0|1': 15,'2|1': 10, '1|2': 4.5,'0|2': 6, '1|0': 10}]
        npointsin = [1]
    
    logging.basicConfig(filename='FromLocalToGlobalfit'+str(date.today())+'_'+data+'_'+methodin+'.log',
                        level=logging.DEBUG, format = '%(asctime)s %(message)s')
    performancefile = 'LIMEFromLocalToGlobalperformance'+str(date.today())+'_'+data+'_'+methodin+'.txt'

    with open(performancefile, 'a') as myfile:
        myfile.write('model data iteration npoints maxdepthexplainer depthexplainer leavesexplainer accuracy precisionmacro recallmacro\n')

    print(bcolors.WARNING + "Before data generation" + bcolors.ENDC)

    dataA, dataB, cols, discrete, continuous, le = loaddata(data)
    blackboxtrainA, trainA, testA = split3fold(dataA, 0.4, 0.2, random_state=1)
    blackboxtrainB, trainB, testB = split3fold(dataB, 0.4, 0.2, random_state=1)
    modelA = pickle.load(open('blackboxes/'+data+'A.sav', 'rb'))
    modelB = pickle.load(open('blackboxes/'+data+'B.sav', 'rb'))

    train = pd.concat([trainA, trainB])
    train['predA'] = modelA.predict(train[cols].values)
    train['predB'] = modelB.predict(train[cols].values)
    train['difference'] = train.apply(lambda row: str(int(row['predA'])) + '|' + str(int(row['predB'])), axis = 1)
    train.drop(columns=['predA', 'predB', 'y'], inplace=True, errors='ignore')
    train = train.reset_index(drop=True)

    print(bcolors.WARNING + "after train generation" + bcolors.ENDC)

    test = pd.concat([testA, testB])
    test['predA'] = modelA.predict(test[cols].values)
    test['predB'] = modelB.predict(test[cols].values)
    test['difference'] = test.apply(lambda row: str(int(row['predA'])) + '|' + str(int(row['predB'])), axis = 1)
    test.drop(columns=['predA', 'predB', 'y'], inplace=True)
    test = test.reset_index(drop=True)

    print(bcolors.WARNING + "after test generation" + bcolors.ENDC)

    discrete.append('difference')
    populationsize = 1000

    logging.info('Starting experiments for data: ' + data + ' with method: ' + methodin)

    print(bcolors.WARNING + "Before loop" + bcolors.ENDC)

    for npoints in npointsin:
            globaldata = getglobal(train, classname='difference', npoints=npoints, populationsize=populationsize,
                                    discrete = discrete, continuous=continuous, modelA=modelA, modelB=modelB,
                                    method=method, random_state=1)
            print(bcolors.WARNING + "After getglobal" + bcolors.ENDC)
            filename = 'LIMEneighborhoods/' + methodin + '_round' + str(1) + '_' + str(npoints) + '_' + data +'.npy'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as f:
                np.save(f, globaldata)
            for j in [3,5,7,None]:
                print(bcolors.WARNING + "In second loop" + bcolors.ENDC)
                #Marker: should be here in getexplainer
                print(bcolors.WARNING + "Before get Standard explainer" + bcolors.ENDC)
                globalexplainer = getexplainer(globaldata.values, modelA, modelB, max_depth = j)
                lime_train = globaldata.values
                used_cols = list(globaldata.columns)
                instance = globaldata[used_cols].iloc[0].values
                instance_id = str(globaldata.index[0])
                print(bcolors.WARNING + "Before get Lime explainer" + bcolors.ENDC)
                explainer = getLimeExplainer.getLimeExplainer(lime_train, feature_names=used_cols, class_names=[str(c) for c in globalexplainer.classes_])
                print(bcolors.WARNING + "Before standard explain" + bcolors.ENDC)
                pred_label = globalexplainer.predict(instance.reshape(1, -1))[0]
                label_index = list(globalexplainer.classes_).index(pred_label)
                print(bcolors.WARNING + "Before lime explain" + bcolors.ENDC)
                ret_exp, scaled_data, inverse, compr = explainer.explain_instance(
                    instance,
                    globalexplainer.predict_proba,
                    labels=(label_index,),
                    num_features=min(10, len(used_cols))
                )
                explanation_list = ret_exp.as_list(label=label_index)

                expl_filename = f'LIMEneighborhoods/explanation_round{1}_npoints{npoints}_depth{j}.json'
                os.makedirs(os.path.dirname(expl_filename), exist_ok=True)
                with open(expl_filename, 'w') as ef:
                    print(bcolors.WARNING + "Before json dump" + bcolors.ENDC)
                    json.dump({'instance_index': instance_id, 'predicted_label': str(pred_label), 'explanation': explanation_list}, ef)
                    print(bcolors.WARNING + "After json dump" + bcolors.ENDC)
                pred = compr.predict(test[cols].values)
                print(bcolors.WARNING + "prediction: " + str(pred) + bcolors.ENDC)
                pred = list(map(str, pred))
                with open(performancefile, 'a') as f:
                    line = ' '.join([methodname,
                                        data,
                                        str(1),
                                        str(npoints),
                                        str(j),
                                        str(globalexplainer.get_depth()),
                                        str(globalexplainer.get_n_leaves()),
                                        str(metrics.accuracy_score(test.difference, pred)),
                                        str(metrics.precision_score(test.difference, pred, average='macro')),
                                        str(metrics.recall_score(test.difference, pred, average='macro'))
                                        ])
                    f.write(line + '\n')