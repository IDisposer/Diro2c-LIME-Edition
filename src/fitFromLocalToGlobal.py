import sys
import logging
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
from FromLocalToGlobal import getglobal, getexplainer, clusterbasedinstances, stratifiedrandominstances, randominstances

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
    elif methodin == 'classstratifiedinstances':
        method = stratifiedrandominstances
        methodname = '"Approach 2: Class-stratified sampling"'
        npointsin = [8,16,32,64]
        ts = [1]
    elif methodin == 'clusterbasedinstances':
        method = clusterbasedinstances
        methodname = '"Approach 3: Cluster-stratified sampling"'

    if data == 'compas':
        ts = [{'2|0': 2.5, '0|1': 7.5,'2|1': 5, '1|2': 2,'0|2': 4, '1|0': 5},
              {'2|0': 4, '0|1': 15,'2|1': 10, '1|2': 7,'0|2': 12, '1|0': 15},
              {'2|0': 4, '0|1': 15,'2|1': 10, '1|2': 4.5,'0|2': 6, '1|0': 10}]
        npointsin = [1]
    elif data == 'bankmarketing':
        ts = [{'0|1': 25, '1|0': 10},
              {'0|1': 30, '1|0': 15},
              {'0|1': 20, '1|0': 7.5}]
        npointsin = [1]
    
    logging.basicConfig(filename='FromLocalToGlobalfit'+str(date.today())+'_'+data+'_'+methodin+'.log',
                        level=logging.DEBUG, format = '%(asctime)s %(message)s')
    performancefile = 'FromLocalToGlobalperformance'+str(date.today())+'_'+data+'_'+methodin+'.txt'

    with open(performancefile, 'a') as myfile:
        myfile.write('model data iteration npoints maxdepthexplainer depthexplainer leavesexplainer accuracy precisionmacro recallmacro\n')

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

    test = pd.concat([testA, testB])
    test['predA'] = modelA.predict(test[cols].values)
    test['predB'] = modelB.predict(test[cols].values)
    test['difference'] = test.apply(lambda row: str(int(row['predA'])) + '|' + str(int(row['predB'])), axis = 1)
    test.drop(columns=['predA', 'predB', 'y'], inplace=True)
    test = test.reset_index(drop=True)

    discrete.append('difference')
    populationsize = 1000

    if methodin == 'advancedstratifiedrandominstances':
        for i in range(5):
            logging.info('start processing round: ' + str(i))
            for t in ts:
                globaldata = getglobal(train, classname='difference', npoints=1, populationsize=populationsize,
                                       discrete=discrete, continuous=continuous, modelA=modelA, modelB=modelB,
                                       method=method, random_state=i, t = t)
                nchoseninstances = len(np.unique(globaldata.index.get_level_values(0)))
                filename = 'neighborhoods/' + methodin + '_round' + str(i) + '_' + str(nchoseninstances) + '_' + data + '.npy'
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, 'wb') as f:
                    np.save(f, globaldata)
                for j in [3,5,7,None]:
                    globalexplainer = getexplainer(globaldata.values, modelA, modelB, max_depth = j)
                    pred = globalexplainer.predict(test[cols].values)
                    with open(performancefile, 'a') as f:
                        line = ' '.join([methodname,
                                         data,
                                         str(i),
                                         str(nchoseninstances),
                                         str(j),
                                         str(globalexplainer.get_depth()),
                                         str(globalexplainer.get_n_leaves()),
                                         str(metrics.accuracy_score(test.difference, pred)),
                                         str(metrics.precision_score(test.difference, pred, average='macro')),
                                         str(metrics.recall_score(test.difference, pred, average='macro'))
                                         ])
                        f.write(line + '\n')
                logging.info('    finished nchoseninstances: ' + str(nchoseninstances))
    else:
        for i in range(5):
            logging.info('start processing round: ' + str(i))
            for npoints in npointsin:
                globaldata = getglobal(train, classname='difference', npoints=npoints, populationsize=populationsize,
                                       discrete = discrete, continuous=continuous, modelA=modelA, modelB=modelB,
                                       method=method, random_state=i)
                filename = 'neighborhoods/' + methodin + '_round' + str(i) + '_' + str(npoints) + '_' + data +'.npy'
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, 'wb') as f:
                    np.save(f, globaldata)
                for j in [3,5,7,None]:
                    #Marker: should be here in getexplainer
                    globalexplainer = getexplainer(globaldata.values, modelA, modelB, max_depth = j)
                    pred = globalexplainer.predict(test[cols].values)
                    with open(performancefile, 'a') as f:
                        line = ' '.join([methodname,
                                         data,
                                         str(i),
                                         str(npoints),
                                         str(j),
                                         str(globalexplainer.get_depth()),
                                         str(globalexplainer.get_n_leaves()),
                                         str(metrics.accuracy_score(test.difference, pred)),
                                         str(metrics.precision_score(test.difference, pred, average='macro')),
                                         str(metrics.recall_score(test.difference, pred, average='macro'))
                                         ])
                        f.write(line + '\n')
                logging.info('    finished npoints: ' + str(npoints))