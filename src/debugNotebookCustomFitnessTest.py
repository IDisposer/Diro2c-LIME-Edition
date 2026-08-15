import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from data.getdata import loaddata, prepare_df
from data.split3fold import split3fold
import pickle
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder, OneHotEncoder
import minisom
from collections import defaultdict
from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette, fcluster
from matplotlib.legend import Legend
from matplotlib import colors
import getLimeExplainer
from lime.lime_tabular import LimeTabularExplainer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin

class VerticalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param
    def predict(self, X):
        closest = 1*(X[:,0]>np.pi/2)
        self.y_ = closest
        return self.y_
    
class CornerToCornerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param
    def predict(self, X):
        closest = np.zeros(X.shape[0])
        x = X[:,0]
        y = X[:,1]
        for i in range(len(x)):
            if (x[i] < -5):
                closest[i] = 1*(y[i] > (-2 * x[i] + 2))
            elif (x[i] < 3):   
                closest[i] = 1*(y[i] > 2)
            else: 
                closest[i] = 1*(y[i] > -2 * x[i])

        self.y_ = closest.astype(np.int64)
        return self.y_

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

import warnings
warnings.filterwarnings("ignore")

plt.style.use('ggplot')

from vars import plot_colors, color_dict, classes_dict, make_meshgrid, make_meshgridLinspace, plot_contours,get_rules
from FromLocalToGlobal import *
data = 'bankmarketing'
dataA, dataB, cols, discrete, continuous, le = loaddata(data)
blackboxtrainA, trainA, testA = split3fold(dataA, 0.4, 0.2, random_state=1)
blackboxtrainB, trainB, testB = split3fold(dataB, 0.4, 0.2, random_state=1)
modelA = pickle.load(open('blackboxes/'+data+'A.sav', 'rb'))
modelB = pickle.load(open('blackboxes/'+data+'B.sav', 'rb'))
#create difference detection dataset:
train = pd.concat([trainA, trainB])
train['predA'] = modelA.predict(train[cols].values)
train['predB'] = modelB.predict(train[cols].values)
train['difference'] = train.apply(lambda row: str(int(row['predA'])) + '|' + str(int(row['predB'])), axis = 1)
train.drop(columns=['predA', 'predB', 'y'], inplace=True)
train = train.reset_index(drop = True, inplace = False)
test = pd.concat([testA, testB])
test['predA'] = modelA.predict(test[cols].values)
test['predB'] = modelB.predict(test[cols].values)
test['difference'] = test.apply(lambda row: str(int(row['predA'])) + '|' + str(int(row['predB'])), axis = 1)
test.drop(columns=['predA', 'predB', 'y'], inplace=True)
discrete.append('difference')
#### Approach 1.3:
##### Dendrograms:
differenceclasses = train.difference.unique()
differenceclasses = differenceclasses[~np.isin(differenceclasses, ['0|0', '1|1', '2|2'])]
differenceclasses.sort()

discrete_woclassname = discrete.copy()
discrete_woclassname.remove('difference')

d = defaultdict(lambda: OneHotEncoder(drop = 'first'))
trainbinary = train.copy()
testbinary = test.copy()
colsbinary = cols.copy()
for feature in discrete_woclassname:
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

trainbinarynorm = trainbinary[colsbinary].copy()
if(len(continuous)>0):
    d = StandardScaler()
    scaler = d.fit(trainbinarynorm[continuous].values)
    trainbinarynorm[continuous] = scaler.transform(trainbinarynorm[continuous].values)
linkagetype = 'ward'
linked = []
for value in differenceclasses:
    subtrainbinarynorm = trainbinarynorm.loc[trainbinary.difference == value, colsbinary].values
    linked.append(linkage(subtrainbinarynorm, linkagetype))
#number of instances for which a neighborhood is generated:
if data == 'compas':
    t = {'2|0': 4, '0|1': 15,
         '2|1': 10, '1|2': 4.5,
         '0|2': 6, '1|0': 10
         } #ward linkage 26
elif data == 'bankmarketing':
    t = {'0|1': 25, '1|0': 10} #ward linkage 21

ninstances = 0
for index, value in enumerate(differenceclasses):
    clusterarr = pd.DataFrame(fcluster(linked[index], t=t[value], criterion='distance'), columns=['cluster'])
    ncluster = len(np.unique(clusterarr))
    ninstances +=  ncluster
    print("{dclass}: {n}".format(dclass = value, n=ncluster))
print(ninstances)
if data == 'compas':
    gs = gridspec.GridSpec(2, 3)
    fig = plt.figure(figsize=(12, 6))
    t = {'2|0': 4, '0|1': 15,
         '2|1': 10, '1|2': 4.5,
         '0|2': 6, '1|0': 10
         } #26 instances
elif data == 'bankmarketing':
    gs = gridspec.GridSpec(1, 2)
    fig = plt.figure(figsize=(12, 3))

for index, value in enumerate(differenceclasses):
    col = int(np.mod(index, 3))
    row = int(np.floor(index/3))
    ax = pl.subplot(gs[row, col])
    set_link_color_palette(plot_colors)
    D = dendrogram(linked[index], ax=ax, no_labels=True, color_threshold=t[value],
                   above_threshold_color='k')
    ax.axhline(y=t[value], c = 'black', linestyle = 'dotted')
    ax.set_facecolor('#FFFFFF')
    ax.set_title('Region ($M_A|M_B$): ' + classes_dict[data][value], fontsize = 10)
    ax.grid(False)
### Obtain explaining model
random_state = 0
# To avoid very long runs, pass very small GA params via gp_kwargs (override defaults)
gp_kwargs = {'ngen': 2, 'population_size': 20, 'max_steps': 1}
globaldata = getglobal(train, classname = 'difference', npoints = 1, populationsize=1000, method = clusterbasedinstances, random_state = random_state, discrete = discrete, continuous=continuous, modelA=modelA, modelB=modelB, t=t, gp_kwargs=gp_kwargs)
## Running example plots
performancefile = 'docout/sections/localtoglobal/resultsCustomFitnessTest/FromLocalToGlobalrunning.txt'
data = 'running1'
train, cols = loaddata(data)
modelA = pickle.load(open('blackboxes/'+data+'A.sav', 'rb'))
modelB = pickle.load(open('blackboxes/'+data+'B.sav', 'rb'))

#modelA = VerticalClassifier()
#modelB = CornerToCornerClassifier()

train['yA'] = modelA.predict(train[['x1', 'x2']].values)
train['yB'] = modelB.predict(train[['x1', 'x2']].values)
print(type(train['yA'][0]))
print(type(train['yB'][0]))
train['difference'] = train.apply(lambda row: '%g' % row['yA'] + '|' + '%g' % row['yB'], axis = 1)
train.drop(columns=['yA', 'yB'], inplace=True)
#load test data:
test, cols = loaddata(data + 'test')
test['yA'] = modelA.predict(test[cols].values)
test['yB'] = modelB.predict(test[cols].values)
test['difference'] = test.apply(lambda row: '%g' % row['yA'] + '|' + '%g' % row['yB'], axis=1)
test.drop(columns=['yA', 'yB'], inplace=True)
from matplotlib.colors import to_rgb
from sklearn.tree import plot_tree
import matplotlib
import re
import pyperclip


def replace_text(obj):
    if type(obj) == matplotlib.text.Annotation:
        txt = obj.get_text()
        txt = re.sub("samples[^$]*class", "class", txt)
        obj.set_text(txt)
    return obj
### Plot for Approach 1.1

gdTrain = getLimeExplainer.expandDataset(train[['x1', 'x2']].values, 5, 0.5, 34)
gdTest = getLimeExplainer.expandDataset(test[['x1', 'x2']].values, 5, 0.5, 34)

class_names_2ClassProblem = ['difference']
class_names_4ClassProblem = ['A:0', 'A:1', 'B:0', 'B:1']

print(str(gdTrain[0]))

predA = modelA.predict(gdTrain)
predB = modelB.predict(gdTrain)

y_meta = (predA * 2 + predB)

train_values_dict = {}
for idx, (x, y) in enumerate(gdTrain):
    train_values_dict[(x, y)] = predA[idx] * 2 + predB[idx]

print(type(train_values_dict))

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors
import time
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from itertools import combinations
from collections import defaultdict

X_aug = gdTrain

global_sigma = 0.5
temp_sigma = global_sigma
neighborsDict = {}

while temp_sigma <= global_sigma * 20:
    temp_ngbrs = NearestNeighbors(radius=temp_sigma)
    temp_ngbrs.fit(X_aug)
    neighborsDict[temp_sigma] = temp_ngbrs
    temp_sigma = math.ceil((temp_sigma + 0.1)* 10) / 10   #avoid floating point precision issues

pointFreqMap = defaultdict(lambda: [0, 0, 0, 0])

linepoints = []

def setBasePlot():
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 8))

    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')

    X0, X1 = train.x1, train.x2
    xx, yy = make_meshgrid(X0, X1, h=0.005)

    cntr1 = plot_contours(ax, modelA, xx, yy, levels=1, colors='black', linewidths=2, linestyles='dotted')
    cntr2 = plot_contours(ax, modelB, xx, yy, levels=1, colors='black', linewidths=1)

    h = [plt.plot([], [], ls=i, color='black')[0] for i in ['dotted', 'solid']]
    ax.legend(handles=h, labels=['Decision Boundary $M_A$', 'Decision Boundary $M_B$'],
            loc='lower left', title='', frameon=False, bbox_to_anchor=(0, -0.12),
            ncol=4)
    plt.grid(True, color='#F3F3F3')

    ax.set_xlim(X0.min(),X0.max())
    ax.set_ylim(train.x2.min(),train.x2.max())

def getColorForTrainingPoint(point):
    if point in train_values_dict:
        value = train_values_dict[point]
        if value == 0:
            return 'blue'
        elif value == 1:
            return 'red'
        elif value == 2:
            return 'green'
        elif value == 3:
            return 'purple'
    return 'black'

#setting base plot
setBasePlot()
#end of setting base plot


X0, X1 = gdTest[:, 0], gdTest[:, 1]
xx, yy = make_meshgridLinspace(X0, X1, h=0.09)
xravel = xx.ravel()
yravel = yy.ravel()
print(bcolors.WARNING + str(xravel.size) + bcolors.ENDC)
grid = np.c_[xravel, yravel]

print(bcolors.OKBLUE + "Starting local predictions on grid")
start_time = time.perf_counter()
print("Time before predictions: " + str(start_time))
#Z, C = getLimeExplainer.local_predict_combined(grid, 0.5, 0, neighborsDict, 10.0, predA, predB, X_aug, False, xx, yy, getColorForTrainingPoint)
end_time = time.perf_counter()
print("Time after predictions: " + str(end_time))
print("Time taken for local predictions: " + str(end_time - start_time) + " seconds")
print("Finished local predictions on grid" + bcolors.ENDC)
#Z = Z.reshape(xx.shape)
#C = C.reshape(xx.shape)

predA = modelA.predict(gdTest)
predB = modelB.predict(gdTest)

#y_meta = (predA * 2 + predB)

cmap = ListedColormap(["blue", "red", "green", "purple"])
labels = ["0|0", "0|1", "1|0", "1|1"]
colors = ["blue", "red", "green", "purple"]

patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(4)]

#plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
#plt.scatter(xx.ravel(), yy.ravel(), c=Z.ravel(), cmap=cmap, alpha=0.6)
#plt.contourf(xx, yy, C, levels=20, cmap="viridis", alpha=0.6)

#plt.colorbar(label="Confidence")
plt.scatter(gdTrain[:, 0], gdTrain[:, 1], c=y_meta, cmap=cmap, edgecolor="k")
#plt.scatter([p[0] for p in linepoints], [p[1] for p in linepoints], c='orange', edgecolor="k")
plt.legend(handles=patches)

plt.title("Regions where models disagree")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
plt.savefig('docout/sections/localtoglobal/resultsCustomFitnessTest/Approach1_decisionsurfaceexplainer_' + data + '.jpg', dpi=150, bbox_inches='tight', transparent=True, pad_inches=0)


def drawHeatmap():
    #setting base plot
    setBasePlot()
    #end of setting base plot

    plt.scatter(xravel, yravel, c=C.ravel(), cmap="viridis", alpha=0.6)

    plt.colorbar(label="Confidence")

    plt.title("Confidence of the explainer")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.savefig('docout/sections/localtoglobal/resultsCustomFitnessTest/Heatmap_' + data + '.jpg', dpi=150, bbox_inches='tight', transparent=True, pad_inches=0)

def drawMostFrequentClass():
    #setting base plot
    setBasePlot()
    #end of setting base plot
    def getMostFrequentClass():
        return [np.argmax(values) for values in pointFreqMap.values()]

    plt.scatter(xravel, yravel, c=getMostFrequentClass(), cmap=cmap, alpha=0.6)

    plt.legend(handles=patches)

    plt.title("The other thing")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.savefig('docout/sections/localtoglobal/resultsCustomFitnessTest/OtherThing_' + data + '.jpg', dpi=150, bbox_inches='tight', transparent=True, pad_inches=0)

#drawHeatmap()

#drawMostFrequentClass()
