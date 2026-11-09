from datetime import datetime
import math
import sys

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
### Obtain explaining model
random_state = 0
# To avoid very long runs, pass very small GA params via gp_kwargs (override defaults)
#gp_kwargs = {'ngen': 2, 'population_size': 20, 'max_steps': 1}
#globaldata = getglobal(train, classname = 'difference', npoints = 1, populationsize=1000, method = clusterbasedinstances, random_state = random_state, discrete = discrete, continuous=continuous, modelA=modelA, modelB=modelB, t=t, gp_kwargs=gp_kwargs)
## Running example plots
performancefile = 'docout/sections/localtoglobal/resultsSingleCustomSimple/FromLocalToGlobalrunning.txt'
data = 'running1'
train, cols = loaddata(data)
#modelA = pickle.load(open('blackboxes/'+data+'A.sav', 'rb'))
#modelB = pickle.load(open('blackboxes/'+data+'B.sav', 'rb'))

modelA = VerticalClassifier()
modelB = CornerToCornerClassifier()

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
        return value
    return 4

#setting base plot
setBasePlot()
#end of setting base plot


X0, X1 = gdTest[:, 0], gdTest[:, 1]
xx, yy = make_meshgridLinspace(X0, X1, h=0.5)
xravel = xx.ravel()
yravel = yy.ravel()
print(bcolors.WARNING + str(xravel.size) + bcolors.ENDC)
grid = np.c_[xravel, yravel]

print(bcolors.OKBLUE + "Starting local predictions on grid")
start_time = time.perf_counter()
print("Time before predictions: " + str(start_time))
orig_stdout = sys.stdout
f = open(f'docout/sections/localtoglobal/resultsSingleCustomSimple/out_{datetime.now().strftime("%Y-%m-%d %H%M%S")}.txt', 'w+')
sys.stdout = f
Z, C = getLimeExplainer.local_predict_improved(grid, 0.5, 100, neighborsDict, 1.8, predA, predB, 
                                               X_aug, True, xx, yy, getColorForTrainingPoint, 
                                               [654], 
                                                [0, 1, 2, 3]
                                               )
sys.stdout = orig_stdout
f.close()
end_time = time.perf_counter()
print("Time after predictions: " + str(end_time))
print("Time taken for local predictions: " + str(end_time - start_time) + " seconds")
print("Finished local predictions on grid" + bcolors.ENDC)
Z = Z.reshape(xx.shape)
C = C.reshape(xx.shape)

cmap = ListedColormap(["blue", "red", "green", "purple", "white"])
labels = ["0|0", "0|1", "1|0", "1|1", "white"]
colors = ["blue", "red", "green", "purple", "white"]

patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(5)]

#plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
plt.scatter(xx.ravel(), yy.ravel(), c=Z.ravel(), cmap=cmap, alpha=0.6)
#plt.contourf(xx, yy, C, levels=20, cmap="viridis", alpha=0.6)

#plt.colorbar(label="Confidence")
#plt.scatter(gdTrain.x1, gdTrain.x2, c=y_meta, cmap=cmap, edgecolor="k")
#plt.scatter([p[0] for p in linepoints], [p[1] for p in linepoints], c='orange', edgecolor="k")
#plt.legend(handles=patches)

plt.title("Regions where models disagree")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
plt.savefig('docout/sections/localtoglobal/resultsSingleCustomSimple/SingleCustom_' + data + '.jpg', dpi=150, bbox_inches='tight', transparent=True, pad_inches=0)


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
    plt.savefig('docout/sections/localtoglobal/resultsSingleCustomSimple/Heatmap_' + data + '.jpg', dpi=150, bbox_inches='tight', transparent=True, pad_inches=0)

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
    plt.savefig('docout/sections/localtoglobal/resultsSingleCustomSimple/OtherThing_' + data + '.jpg', dpi=150, bbox_inches='tight', transparent=True, pad_inches=0)

#drawHeatmap()

#drawMostFrequentClass()
