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
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import minisom
from collections import defaultdict
from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette, fcluster
from matplotlib.legend import Legend
from matplotlib import colors

from sklearn.tree import plot_tree
import matplotlib
import re
import pyperclip
import getLimeExplainer
import math
from lime.lime_tabular import LimeTabularExplainer

plt.style.use('ggplot')

from vars import plot_colors, color_dict, classes_dict, make_meshgrid, plot_contours,get_rules
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

#number of instances for which a neighborhood is generated:
if data == 'compas':
    t = {'2|0': 4, '0|1': 15,
         '2|1': 10, '1|2': 4.5,
         '0|2': 6, '1|0': 10
         } #ward linkage 26
elif data == 'bankmarketing':
    t = {'0|1': 25, '1|0': 10} #ward linkage 21

random_state = 0
gp_kwargs = {'ngen': 2, 'population_size': 20, 'max_steps': 1}
globaldata = getglobal(train, classname = 'difference', npoints = 1, populationsize=1000, method = clusterbasedinstances, random_state = random_state, discrete = discrete, continuous=continuous, modelA=modelA, modelB=modelB, t=t, gp_kwargs=gp_kwargs)
performancefile = 'docout/sections/localtoglobal/results/FromLocalToGlobalrunning.txt'
data = 'running1'
train, cols = loaddata(data)
modelA = pickle.load(open('blackboxes/'+data+'A.sav', 'rb'))
modelB = pickle.load(open('blackboxes/'+data+'B.sav', 'rb'))
train['yA'] = modelA.predict(train[['x1', 'x2']].values)
train['yB'] = modelB.predict(train[['x1', 'x2']].values)
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

populationsize = 1000
gp_kwargs = {'ngen': 2, 'population_size': 1000, 'max_steps': 1}
globaldata = getglobal(train, modelA=modelA, modelB=modelB, classname='difference', npoints=8,
                       populationsize=populationsize, method=randominstances,
                       random_state=0, discrete=['difference'],
                       continuous=['x1', 'x2'], gp_kwargs=gp_kwargs)
globaldata.columns = ['x1', 'x2']

with open('docout/sections/localtoglobal/res/approach1_neighborhood_' + data + '.pickle', 'wb') as handle:
    pickle.dump(globaldata, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('docout/sections/localtoglobal/res/approach1_neighborhood_' + data + '.pickle', 'rb') as handle:
    globaldata = pickle.load(handle)

fig, ax = plt.subplots(figsize=(10, 8))

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

X0, X1 = train.x1, train.x2
xx, yy = make_meshgrid(X0, X1, h=0.005)

z1 = modelA.predict(np.c_[xx.ravel(), yy.ravel()])
z1 = z1.reshape(xx.shape)
z2 = modelB.predict(np.c_[xx.ravel(), yy.ravel()])
z2 = z2.reshape(xx.shape)

cntr1 = plot_contours(ax, modelA, xx, yy, levels=1, colors='black', linewidths=2, linestyles='dotted')
cntr2 = plot_contours(ax, modelB, xx, yy, levels=1, colors='black', linewidths=1)

h = [plt.plot([], [], ls=i, color='black')[0] for i in ['dotted', 'solid']]
ax.legend(handles=h, labels=['Decision Boundary $M_A$', 'Decision Boundary $M_B$'],
          loc='lower left', title='', frameon=False, bbox_to_anchor=(0, -0.12),
          ncol=4)

ax.scatter(globaldata.x1, globaldata.x2, c='black', alpha=0.3, s=6)
choseninstancesindex = np.unique(globaldata.index.get_level_values(0))
choseninstances = train.iloc[choseninstancesindex]
ax.scatter(choseninstances.x1, choseninstances.x2, c='#D90429', s=150, marker='X')

ax.set_facecolor('#FFFFFF')
plt.tight_layout()
plt.grid(True, color='#F3F3F3')

ax.set_xlim(X0.min(),X0.max())
ax.set_ylim(train.x2.min(),train.x2.max())

plt.savefig('docout/sections/localtoglobal/results/Approach1_generatedneighborhoods_' + data + '.jpg', dpi=150,bbox_inches='tight',transparent=True, pad_inches=0)
predA = modelA.predict(globaldata.values)
predB = modelB.predict(globaldata.values)
difference = np.array([str(x1) + '|' + str(x2) for x1, x2 in zip(predA, predB)])
explainer = DecisionTreeClassifier(random_state=0)
explainer.fit(globaldata.values, difference)
X0, X1 = globaldata.x1, globaldata.x2
xx, yy = make_meshgrid(X0, X1, h=0.005)
print('!!SIZE' + str(xx.ravel().size) + '!!SIZE')
z = explainer.predict(np.c_[xx.ravel(), yy.ravel()])
d=LabelEncoder()
d.fit(np.array([x for x in color_dict[data].keys()]))
z = d.transform(z)
z = z.reshape(xx.shape)
ordering = [x for x in color_dict[data].keys()]
keys = list(ordering)
ordering.sort()
ordering = [keys.index(x) for x in ordering]

#Evaluation:
if data == 'running1':
    dataname = '"Sine"'
else:
    dataname = '"Spiral"'
pred = explainer.predict(test[cols].values)
with open(performancefile, 'a') as myfile:
    line = ' '.join([dataname,
                     '"Approach 1: Random sampling"',
                     str(explainer.get_depth()),
                     str(explainer.get_n_leaves()),
                     str(metrics.accuracy_score(test.difference, pred)),
                     str(metrics.precision_score(test.difference, pred, average='macro')),
                     str(metrics.recall_score(test.difference, pred, average='macro'))
                     ])
    myfile.write(line + '\n')

values = [x[0] for x in color_dict[data].values()]
orderedmap = [values[i] for i in ordering]
MyCmap=colors.ListedColormap(orderedmap)
fig, ax = plt.subplots(figsize=(10, 8))

cntr1 = plot_contours(ax, modelA, xx, yy, levels=1, colors='black', linewidths=2, linestyles='dotted')
cntr2 = plot_contours(ax, modelB, xx, yy, levels=1, colors='black', linewidths=1)

cp = ax.contourf(xx, yy, z+0.1, alpha=0.7, cmap=MyCmap)

h = [plt.plot([], [], color=i[0], linewidth=10, label=j)[0] for j, i in color_dict[data].items()]
ax.legend(handles=h, loc='lower left', title='Prediction Explainer', frameon=False, bbox_to_anchor=(0, -0.12),
          ncol=9)

ax.set_facecolor('#FFFFFF')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.grid(True, color='#F3F3F3')

ax.set_ylim(train.x2.min(), train.x2.max())
ax.set_xlim(train.x1.min(), train.x1.max())

plt.tight_layout()
plt.savefig('docout/sections/localtoglobal/results/Approach1_decisionsurfaceexplainer_' + data + '.jpg', dpi=150, bbox_inches='tight', transparent=True, pad_inches=0)

fig, ax = plt.subplots(figsize=(55, 10))
class_names = explainer.classes_
colorss = [color_dict[data][x][0] for x in class_names]
N = len(class_names)
artists = plot_tree(explainer, fontsize=8, ax=ax,
                    impurity=False, node_ids=True,
                    feature_names=cols, class_names=class_names)
ax.properties()['children'] = [replace_text(i) for i in ax.properties()['children']]
# for artist, impurity, value in zip(artists, explainer.tree_.impurity, explainer.tree_.value):
#     # let the max value decide the color; whiten the color depending on impurity (gini)
#     r, g, b = to_rgb(colorss[np.argmax(value)])
#     f = impurity * N / (N - 1) if N > 1 else 0
#     artist.get_bbox_patch().set_facecolor((f + (1 - f) * r, f + (1 - f) * g, f + (1 - f) * b))
#     artist.get_bbox_patch().set_edgecolor('black')
plt.savefig('docout/sections/localtoglobal/results/Approach1_' + data + '_explainer.jpg', dpi=300, bbox_inches='tight',transparent=True, pad_inches=0)

rules = get_rules(explainer, ['x_1', 'x_2'], class_names)
rules = '\n'.join(rules)
pyperclip.copy(rules)