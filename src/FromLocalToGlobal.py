from collections import defaultdict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.cluster.hierarchy import linkage, fcluster
from data.getdata import prepare_df
from diro2c.data_generation.neighborhood_generation import modified_gpdatagenerator
from diro2c.data_generation.distance_functions import simple_match_distance, normalized_euclidean_distance, mixed_distance
from diro2c.enums.diff_classifier_method_type import diff_classifier_method_type
from diro2c.data_generation.neighborhood_generation.gpdatagenerator import calculate_feature_values
from sklearn.tree import DecisionTreeClassifier
from diro2c.data_generation.helper import *
import pandas as pd
import numpy as np

def randominstances(data, **kwargs):
    r"""
    Samples random instances from data

    :param data:
        Data to sample from
    :type data: ``pandas.DataFrame``
    :param \**kwargs:
        See below
    :Keyword Arguments:
        * *npoints* (``int``) --
          Number of instances to sample
        * *random_state* (``int``) --
          Seed for random number generator

    :return: DataFrame containing *n* items sampled from data.
    """

    return data.sample(n=kwargs['npoints'], replace = False, random_state = kwargs['random_state'])

def stratifiedrandominstances(data, **kwargs):
    r"""
    Samples stratified random instances from data

    :param data:
        Data to sample from
    :type data: ``pandas.DataFrame``
    :param \**kwargs:
        See below
    :Keyword Arguments:
        * *npoints* (``int``) --
          Number of instances to sample
        * *classname* (``str``) --
          Name of stratification variable found in data
        * *random_state* (``int``) --
          Seed for random number generator

    :return: DataFrame containing *n* items sampled from data.
    """

    npoints = np.max([1, int(np.floor(kwargs['npoints']/len(data[kwargs['classname']].unique())))])
    return data.groupby(kwargs['classname'], group_keys=False).sample(n=npoints, random_state=kwargs['random_state'], replace = True).drop_duplicates()

def clusterbasedinstances(data, **kwargs):
    r"""
    Samples cluster-based instances from data

    :param data:
        Data to sample from
    :type data: ``pandas.DataFrame``
    :param \**kwargs:
        See below
    :Keyword Arguments:
        * *cols* (``list``) --
          Feature names of data
        * *discrete* (``list``) --
          Discrete feature names of data for One-Hot encoding
        * *classname* (``str``) --
          Name of stratification variable found in data
        * *continuous* (``list``) --
          Continuous feature names of data for scaling
        * *random_state* (``int``) --
          Seed for random number generator
        * *t* (``dict``) --
            key: '<prediction model A>|<prediction model B>', value: t in scipy.cluster.hierarchy.fcluster for criteria 'distance'
        * *linkagetype* (``str``) --
            one of 'ward', 'single'; used for hierarchical clustering

    :return: DataFrame containing *n* items sampled from data.
    """

    d = defaultdict(lambda: OneHotEncoder(drop = 'first'))
    trainbinary = data.copy()
    colsbinary = kwargs['cols'].copy()
    for feature in kwargs['discrete']:
        tmp = d[feature].fit_transform(trainbinary[feature].values.reshape(-1,1)).toarray()
        colnames = [feature + str(i) for i in range(tmp.shape[1])]
        trainbinary[colnames] = tmp
        colsbinary = colsbinary + colnames
        colsbinary.remove(feature)
        trainbinary.drop(columns = feature, inplace = True)

    differenceclasses = data[kwargs['classname']].unique()
    differenceclasses = differenceclasses[~np.isin(differenceclasses, ['0|0', '1|1', '2|2'])]

    regions = data.loc[data.difference.isin(differenceclasses)].copy()
    regions['region'] = np.nan

    trainbinarynorm = trainbinary[colsbinary].copy()
    if len(kwargs['continuous'])>0:
        d = StandardScaler()
        scaler = d.fit(trainbinarynorm[kwargs['continuous']].values)
        trainbinarynorm[kwargs['continuous']] = scaler.transform(trainbinarynorm[kwargs['continuous']].values)

    for value in differenceclasses:
        subtrainbinary = trainbinarynorm.loc[trainbinary.difference == value, colsbinary].values
        hc = linkage(subtrainbinary, kwargs['linkagetype'])
        regions.loc[regions.difference == value, 'region'] = fcluster(hc, t=kwargs['t'][value], criterion='distance')

    sampled = regions.groupby([kwargs['classname'], 'region'], group_keys=False).sample(n=1, random_state=kwargs['random_state'])
    sampled.drop(columns = 'region', inplace=True)

    return sampled

def getglobal(data: pd.DataFrame, classname: str, npoints: int, populationsize: int,
              discrete, continuous, modelA, modelB,
              method = randominstances, random_state = 1, t = None, linkagetype='ward', gp_kwargs: dict = None):
    r"""
    Synthetic global neighborhood generation using DiRo2C

    :param data:
        Data to sample from
    :type data: ``pandas.DataFrame``
    :param classname:
        Name of stratification variable found in data
    :type classname: ``str``
    :param npoints:
        Number of instances to sample; Only relevant for method = randominstances or stratifiedrandominstances
    :type npoints: ``int``
    :param populationsize:
        Size (Number of instances) of the neighborhood to be generated
    :type populationsize: ``int``
    :param discrete:
        Discrete feature names of data for One-Hot encoding
    :type discrete: ``list``
    :param continuous:
        Continuous feature names of data for scaling
    :type continuous: ``list``
    :param modelA:
       First classification model
   :param modelB:
       Second classification model
    :param method:
        Method to sample instances from data
    :param random_state:
        Seed for random number generator
    :type random_state: ``int``
    :param t:
        used only for method = clusterbasedsampling
        key: '<prediction model A>|<prediction model B>', value: t in scipy.cluster.hierarchy.fcluster for criteria 'distance'
    :type t: ``dict``
    :param linkagetype:
        used only for method = clusterbasedsampling, one of 'ward', 'single
    :type t: ``str``

    :return: DataFrame of synthetic neighborhood for sampled instances in index
    """

    data[classname] = data[classname].astype(str)
    dataset = prepare_df(data, 'intern', classname, discrete, continuous)
    discrete_no_class = list(dataset['discrete'])
    discrete_no_class.remove(classname)

    features = dataset['columns'].copy()
    features.remove(classname)

    instances = method(data, random_state = random_state, classname = classname, t = t, discrete = discrete_no_class,
                       continuous = continuous, npoints = npoints, cols = features, linkagetype = linkagetype)

    geneticneighborhood = dict.fromkeys(instances.index)
    X = np.array(data[features])
    Z_to_rec_diff = cPickle.loads(cPickle.dumps(X))
    feature_values = calculate_feature_values(
        X, dataset['columns'], classname, dataset['discrete'], dataset['continuous'], len(data)
    )

    discrete_no_class = list(dataset['discrete'])
    discrete_no_class.remove(classname)

    def distance_function(x0, x1, discrete, continuous, class_name):
        return mixed_distance(x0, x1, discrete, continuous, class_name,
                              ddist=simple_match_distance,
                              cdist=normalized_euclidean_distance)

    # allow overriding GP generator parameters via gp_kwargs to speed up runs
    default_gp = dict(
        neigtype={'ss': 0.5, 'sd': 0.5},
        population_size=populationsize,
        halloffame_ratio=0.05,
        alpha1=0.5,
        alpha2=0.5,
        eta1=1,
        eta2=0.0,
        tournsize=3,
        cxpb=0.2,
        mutpb=0.3,
        ngen=100,
        return_logbook=False,
        max_steps=10,
        is_unique=True
    )
    if gp_kwargs is None:
        gp_kwargs = {}

    for indexinstance in instances.index:
        instance = Z_to_rec_diff[indexinstance,:]
        # merge defaults with provided overrides
        gp_call_args = default_gp.copy()
        gp_call_args.update(gp_kwargs)

        Z = modified_gpdatagenerator.generate_modified_data(
            instance,
            feature_values,
            modelA,
            modelB,
            diff_classifier_method_type.multiclass_diff_classifier,
            discrete_no_class,
            dataset['continuous'],
            classname,
            dataset['idx_features'],
            distance_function,
            **gp_call_args
        )
        geneticneighborhood[indexinstance] = Z

    geneticneighborhood = pd.concat({k:pd.DataFrame(v) for k,v in geneticneighborhood.items()})
    return geneticneighborhood

def getexplainer(geneticneighborhood, modelA, modelB, **kwargs):
    r"""
   Determines decision differences and fits explainer for geneticneighborhood

   :param geneticneighborhood:

   :type data: ``pandas.DataFrame``
   :param modelA:
       First classification model
   :param modelB:
       Second classification model
   :param \**kwargs:
        Arguments passed on to sklearn.tree.DecisionTreeClassifier

   :return: trained sklearn.tree.DecisionTreeClassifier
   """
    predA = modelA.predict(geneticneighborhood).astype(str)
    predB = modelB.predict(geneticneighborhood).astype(str)
    difference = pd.Series(np.char.add(np.char.add(predA, '|'), predB))
    dtc = DecisionTreeClassifier(random_state=0, **kwargs)
    dtc.fit(geneticneighborhood, difference)
    return dtc