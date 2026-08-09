import sys
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from diro2c.data_generation.helper import *
from sklearn import metrics
import minisom
import pickle
from diro2c.data_generation.neighborhood_generation import modified_gpdatagenerator
from diro2c.data_generation.distance_functions import simple_match_distance, normalized_euclidean_distance, mixed_distance
from diro2c.data_generation.helper import *
import _pickle as cPickle
from diro2c.enums.diff_classifier_method_type import diff_classifier_method_type
from diro2c.data_generation.neighborhood_generation.gpdatagenerator import calculate_feature_values
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from collections import defaultdict
import logging
from sklearn.cluster import AgglomerativeClustering
from datetime import date

from data.getdata import loaddata, prepare_df
from data.split3fold import split3fold

def getclusterid(x, som, clusterarr):
    #x ... normalized instance
    bmu = getwinnerid(x, som)
    c = clusterarr.loc[clusterarr.node == bmu, 'cluster']
    return c

def getwinnerid(x, som):
    #x ... normalized instance
    bmu = som.winner(x)[1]
    return bmu

def distance_function(x0, x1, discrete, continuous, class_name):
    return mixed_distance(x0, x1, discrete, continuous, class_name,
                          ddist=simple_match_distance,
                          cdist=normalized_euclidean_distance)

def predict(x, explainers):
    winner = x[-1]
    x = x[:-1]
    mod = explainers[winner]
    return mod.predict(x.reshape(1, -1))

if __name__ == '__main__':

    data = sys.argv[1]

    logging.basicConfig(filename='SOMfit'+str(date.today())+'_'+data+'.log',
                        level=logging.DEBUG, format = '%(asctime)s %(message)s')
    performancefile = 'SOMperformance'+str(date.today())+'_'+data+'.txt'

    with open(performancefile, 'a') as myfile:
        myfile.write('data cutoffdistance round nclusters maxdepthexplainer meandepth meanleaves accuracy precisionmacro recallmacro\n')

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

    discrete_no_class = discrete.copy()
    discrete.append('difference')

    d = defaultdict(lambda: OneHotEncoder(drop = 'first'))
    trainbinary = train.copy()
    testbinary = test.copy()
    colsbinary = cols.copy()
    for feature in discrete_no_class:
        uniquevals = np.concatenate((trainbinary[feature].values.reshape(-1,1), testbinary[feature].values.reshape(-1,1)))
        d[feature].fit(uniquevals)
        tmp = d[feature].transform(trainbinary[feature].values.reshape(-1,1)).toarray()
        colnames = [feature + str(i) for i in range(tmp.shape[1])]
        trainbinary[colnames] = tmp
        testbinary[colnames] = d[feature].transform(testbinary[feature].values.reshape(-1,1)).toarray()
        colsbinary = colsbinary + colnames
        colsbinary.remove(feature)
        trainbinary.drop(columns = feature, inplace = True)
        testbinary.drop(columns = feature, inplace = True)

    trainsom = trainbinary[colsbinary].values
    testsom = testbinary[colsbinary].values

    trainsomnorm = trainbinary[colsbinary].copy()
    testsomnorm = testbinary[colsbinary].copy()
    scaler = StandardScaler()
    scaler = scaler.fit(trainsomnorm[continuous].values)
    trainsomnorm[continuous] = scaler.transform(trainsomnorm[continuous].values)
    testsomnorm[continuous] = scaler.transform(testsomnorm[continuous].values)

    trainsomnorm = trainsomnorm.values
    testsomnorm = testsomnorm.values

    trainsomnorm = trainsomnorm[~train.difference.isin(['0|0', '1|1', '2|2'])]
    trainsom = trainsom[~train.difference.isin(['0|0', '1|1', '2|2'])]

    hyperparameter = {
        'bankmarketing': {'sigma': 0.9, 'learning_rate': 1.0, 'distance_threshold': [4.5]},#, 4, 3.5, 3
        'compas': {'sigma': 0.9, 'learning_rate': 1.0, 'distance_threshold': [3]},#, 2, 1.5, 1.25
    }


    train['difference'] = train['difference'].astype(str)
    dataset = prepare_df(train, 'train', 'difference', discrete, continuous)
    features = dataset['columns'].copy()
    features.remove('difference')
    X = np.array(train[features])
    Z_to_rec_diff = cPickle.loads(cPickle.dumps(X))
    feature_values = calculate_feature_values(X, dataset['columns'], 'difference', dataset['discrete'], dataset['continuous'], len(train))
    traindifferences = train.loc[~train.difference.isin(['0|0', '1|1', '2|2'])]

    for distance_threshold in hyperparameter[data]['distance_threshold']:
        logging.info('processing threshold: ' + str(distance_threshold))
        n_nodes = int(np.floor(5*np.sqrt(len(trainsom))))
        som = minisom.MiniSom(1, n_nodes, trainsom.shape[1], sigma=hyperparameter[data]['sigma'],
                          learning_rate=hyperparameter[data]['learning_rate'], random_seed = 0)
        som.train(trainsomnorm, 100000, verbose = False)

        connectivity_matrix = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes - 1):
            connectivity_matrix[i, i + 1] = 1.0
        weights = som.get_weights()[0]
        ward = AgglomerativeClustering(connectivity=connectivity_matrix, linkage="single",
                                   distance_threshold=distance_threshold,
                                   n_clusters=None).fit(weights)
        label = ward.labels_
        clusterarr = pd.DataFrame({'node': range(n_nodes), 'cluster': label})
        clusterarr['cluster'] = pd.factorize(clusterarr.cluster)[0]
        ncluster = len(np.unique(label))

        clusterwinnerspos = np.apply_along_axis(getclusterid, 1, weights, som, clusterarr)
        clusterwinners = np.apply_along_axis(getclusterid, 1, trainsomnorm, som, clusterarr)
        nodeswinners = np.apply_along_axis(getwinnerid, 1, trainsomnorm, som)

        clusterswithoutdata = [x for x in range(ncluster) if x not in list(np.unique(clusterwinners))]
        nodesofclusterwithoutdata = clusterarr.loc[clusterarr.cluster.isin(clusterswithoutdata), 'node'].tolist()
        #for each cluster, for each node determine nearest node in cluster with data:
        for node in nodesofclusterwithoutdata:
            weightnode = weights[node]
            nextnode = node
            i = 1
            while nextnode in nodesofclusterwithoutdata:
                map = som._activation_distance(weightnode, som._weights)[0, [node - i, node + i]].argsort()
                nextnode = node + i if map[0] > 0 else node - i
                i = i + 1
            if nextnode >= n_nodes:
                nextnode = node -i
            elif nextnode < 0:
                nextnode = node +i
            oldcluster = clusterarr.loc[clusterarr.node == node, 'cluster'].item()
            newcluster = clusterarr.loc[clusterarr.node == nextnode, 'cluster'].item()
            clusterarr.loc[clusterarr.node == node, 'cluster'] = newcluster

        nodeswithoutdata = [x for x in range(n_nodes) if x not in list(np.unique(nodeswinners))]
        subclusterarr = clusterarr.loc[~clusterarr.node.isin(nodeswithoutdata)]

        for i in range(5):
            logging.info('  start processing round: ' + str(i))
            explainers = dict.fromkeys(['3','5','7','None'])
            for k in ['3','5','7','None']:
                explainers[k] = dict.fromkeys(list(range(ncluster)))

            for clusterid in np.unique(clusterwinners):
                logging.info('      processing cluster ' + str(clusterid))
                if len(subclusterarr.loc[subclusterarr.cluster==clusterid])>4:
                    start = subclusterarr.loc[subclusterarr.cluster == clusterid,'node'].min()
                    end = subclusterarr.loc[subclusterarr.cluster == clusterid,'node'].max()
                    middle = int(subclusterarr.loc[subclusterarr.cluster == clusterid,'node'].median())
                    nodes = [start, end, middle]
                elif len(subclusterarr.loc[subclusterarr.cluster==clusterid])>2:
                    start = subclusterarr.loc[subclusterarr.cluster == clusterid,'node'].min()
                    end = subclusterarr.loc[subclusterarr.cluster == clusterid,'node'].max()
                    nodes = [start, end]
                else:
                    nodes = list(subclusterarr.loc[subclusterarr.cluster == clusterid,'node'].sample(n=1, random_state = i))

                Z3 = np.empty((0, train[cols].shape[1]))

                for x in nodes:
                    indx = (nodeswinners == x)
                    if indx.sum() >0:
                        instance = traindifferences.loc[indx].sample(n=1, random_state=i)
                        instanceindex = instance.index[0]
                        instance = instance.values.reshape(-1, )[:-1]
                        Z = modified_gpdatagenerator.generate_modified_data(instance, feature_values, modelA, modelB,
                                                                            diff_classifier_method_type.multiclass_diff_classifier,
                                                                            discrete_no_class, dataset['continuous'], 'difference',
                                                                            dataset['idx_features'],
                                                                            distance_function, neigtype={'ss': 0.5, 'sd': 0.5},
                                                                            population_size=1000, halloffame_ratio=0.05,
                                                                            alpha1=0.5, alpha2=0.5, eta1=1, eta2=0.0,
                                                                            tournsize=3, cxpb=0.2, mutpb=0.3, ngen=100,
                                                                            return_logbook=False, max_steps=20, is_unique=True)
                        Z3 = np.concatenate([Z3, Z])

                #restrict neighborhood to current cluster
                Z3df = pd.DataFrame(Z3, columns = cols)
                for feature in discrete_no_class:
                    tmp = d[feature].transform(Z3df[feature].values.reshape(-1,1)).toarray()
                    colnames = [feature + str(i) for i in range(tmp.shape[1])]
                    Z3df[colnames] = tmp
                    Z3df.drop(columns = feature, inplace = True)
                Z3df[continuous] = scaler.transform(Z3df[continuous].values)
                Z3df = Z3df.values
                neighborhoodwinners = np.apply_along_axis(getclusterid, 1, Z3df, som, clusterarr)
                ind = (neighborhoodwinners == clusterid).flatten()
                Z3 = Z3[ind]

                filename = 'neighborhoods/SOM_round' + str(i) + '_cluster' + str(clusterid) + '_' + str(distance_threshold) + '_' + data + '.npy'
                with open(filename, 'wb') as f:
                    np.save(f, Z3)

                predA = modelA.predict(Z3).astype(str)
                predB = modelB.predict(Z3).astype(str)
                difference = pd.Series(np.char.add(np.char.add(predA, '|'), predB))
                for j in [3,5,7,None]:
                    clf = DecisionTreeClassifier(random_state=0, max_depth=j)
                    clf.fit(Z3, difference)
                    explainers[str(j)][clusterid] = clf
                logging.info('      finished processing cluster ' + str(clusterid) + '/' + str(ncluster))
            winners = np.apply_along_axis(getclusterid, 1, testsomnorm, som, clusterarr)
            winners = winners.reshape((len(winners), 1))
            res = np.append(test[cols].values, winners, axis=1)

            for j in [3,5,7,None]:
                pred = np.apply_along_axis(predict, 1, res, explainers[str(j)])

                #description of explainer:
                depths = [x.get_depth() for x in explainers[str(j)].values() if x is not None]
                leaves = [x.get_n_leaves() for x in explainers[str(j)].values() if x is not None]

                with open(performancefile, 'a') as myfile:
                    line = ' '.join([data,
                                     str(distance_threshold),
                                     str(i),
                                     str(ncluster),
                                     str(j),
                                     str(np.mean(depths)),
                                     str(np.mean(leaves)),
                                     str(metrics.accuracy_score(test.difference, pred)),
                                     str(metrics.precision_score(test.difference, pred, average='macro')),
                                     str(metrics.recall_score(test.difference, pred, average='macro'))
                                     ])
                    myfile.write(line + '\n')