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
from collections import defaultdict
from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette, fcluster
from matplotlib.legend import Legend
from matplotlib import colors
from matplotlib.colors import to_rgb
from sklearn.tree import plot_tree
import matplotlib
import re
import getLimeExplainer
from lime.lime_tabular import LimeTabularExplainer

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
performancefile = 'docout/sections/localtoglobal/resUSLIME/FromLocalToGlobalrunning.txt'
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
#test.drop(columns=['yA', 'yB'], inplace=True)

def replace_text(obj):
    if type(obj) == matplotlib.text.Annotation:
        txt = obj.get_text()
        txt = re.sub("samples[^$]*class", "class", txt)
        obj.set_text(txt)
    return obj
### Plot for Approach 1.1
populationsize = 100000
# use tiny GP params for a quick run (overrides defaults in getglobal)
gp_kwargs = {'ngen': 2, 'population_size': 1000, 'max_steps': 1}
globaldata = getglobal(train, modelA=modelA, modelB=modelB, classname='difference', npoints=8,
                       populationsize=populationsize, method=randominstances,
                       random_state=0, discrete=['difference'],
                       continuous=['x1', 'x2'], gp_kwargs=gp_kwargs)
globaldata.columns = ['x1', 'x2']
with open('docout/sections/localtoglobal/resUSLIME/approach1_neighborhood_' + data + '.pickle', 'wb') as handle:
    pickle.dump(globaldata, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('docout/sections/localtoglobal/resUSLIME/approach1_neighborhood_' + data + '.pickle', 'rb') as handle:
    globaldata = pickle.load(handle)
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

ax.scatter(globaldata.x1, globaldata.x2, c='black', alpha=0.3, s=6)
choseninstancesindex = np.unique(globaldata.index.get_level_values(0))
choseninstances = train.iloc[choseninstancesindex]
ax.scatter(choseninstances.x1, choseninstances.x2, c='#D90429', s=150, marker='X')

ax.set_facecolor('#FFFFFF')
plt.tight_layout()
plt.grid(True, color='#F3F3F3')

ax.set_xlim(X0.min(),X0.max())
ax.set_ylim(train.x2.min(),train.x2.max())

plt.savefig('docout/sections/localtoglobal/resultsUSLIME/Approach1_generatedneighborhoods_' + data + '.jpg', dpi=150,bbox_inches='tight',transparent=True, pad_inches=0)

k = int(len(globaldata.x1)/2)
gdTrain, gdTest = globaldata.iloc[:k, :], globaldata.iloc[k:, :]

explainer = getLimeExplainer.getUSLimeExplainer(n_samples=10000, n_neighbors=50, noise_scale=0.01, random_state=0)

def diff_predict_fn(X):
    #print("in diff_predict_fn")
    pa = modelA.predict(X)#[:, 1]
    pb = modelB.predict(X)#[:, 1]
    #print(pa)
    #print(pb)
    pb = pb * 2 #two differentiate between 0 - 0 and 1 - 1
    return (pa - pb)

explainer.fit(gdTrain.values, diff_predict_fn)
# explainer = LimeTabularExplainer(
#     training_data=gdTrain.values,
#     feature_names=list(gdTrain.columns),
#     kernel_width=0.25,
#     discretize_continuous=False,
#     class_names=['diff'],
#     mode='regression'
# )



def diffOfArrays(a1, a2):
    return np.subtract(a1, a2)

#X0, X1 = globaldata.x1, globaldata.x2
X0, X1 = gdTest.x1, gdTest.x2
print(bcolors.WARNING + str(X0) + bcolors.ENDC)
print(bcolors.WARNING + str(X1) + bcolors.ENDC)
print(bcolors.WARNING + "Before meshgrid" + bcolors.ENDC)
xx, yy = make_meshgridLinspace(X0, X1, h=0.5)
print(bcolors.WARNING + str(xx.size) + bcolors.ENDC)
print(bcolors.WARNING + str(yy.size) + bcolors.ENDC)
print(bcolors.WARNING + "Before data explanation" + bcolors.ENDC)
xravel = xx.ravel()
yravel = yy.ravel()
print(bcolors.WARNING + str(xravel.size) + bcolors.ENDC)
print(bcolors.WARNING + str(yravel.size) + bcolors.ENDC)
#print(bcolors.WARNING + str(np.c_[xravel, yravel]) + bcolors.ENDC)
print(bcolors.WARNING + str(diffOfArrays(xravel, yravel)[0]) + bcolors.ENDC)
print(globaldata[list(globaldata.columns)].iloc[0].values)
arrCompr = np.empty(xravel.size)
arrPred = np.empty(xravel.size)
arrDiff = np.empty(xravel.size)
lastCompr = None
lastScore = None
lastExp = None
print(bcolors.OKBLUE + "Starting explanation loop")
for i in range(xravel.size):
    compr, scaled_data, weights = explainer.explain(np.array([xravel[i], yravel[i]]), surrogate='rf')
    #arr[i] = compr.predict(np.c_([xravel[i],yravel[i]]))
    #print(bcolors.WARNING + "explanations round " + str(i) + bcolors.ENDC)
    #print(bcolors.WARNING + "" + bcolors.ENDC)
    #arr[i] = ret_exp.local_exp
    #arr[i] = compr.predict(np.c_[xravel, yravel])[i]
    print("In loop round " + str(i))
    arrCompr[i] = round(compr.predict(scaled_data)[0], 0)
    #print("Pred: " + str(arrPred[i]) + " | Compr: " + str(arrCompr[i]) + " | Instance: " + str(np.array([xravel[i], yravel[i]])) + " | Score: " + str(ret_exp.score))
    lastCompr = compr
    #print(bcolors.OKBLUE + str(ret_exp.predicted_value) + ' | ' + str(round(arr[i], 2)) + bcolors.ENDC)
    #arr[i] = ret_exp.predicted_value
print(bcolors.ENDC)
    #print(bcolors.OKGREEN + str(arr[i]) + bcolors.ENDC)
arrCompr = np.clip(arrCompr, -2, 1)
X0_np = X0.to_numpy()
X1_np = X1.to_numpy()

freq_map = dict(zip(*np.unique(arrCompr, return_counts=True)))
print(bcolors.OKGREEN)
print(freq_map)
print(bcolors.ENDC)



# mean1 = X0_np.mean()
# idx1 = np.abs(X0_np - mean1).argmin()

# mean2 = X1_np.mean()
# idx2 = np.abs(X1_np - mean2).argmin()

# explanationCenterValues = np.array([
#     X0_np[idx1],
#     X1_np[idx2]
# ])
# exp, scaled_data, inverse, compr = explainer.explain_instance(
#     explanationCenterValues,
#     diff_predict_fn,
#     num_features=2
# )
#lastExp = exp
#lastCompr = compr
#grid = np.c_[xravel, yravel]
#Z = np.round(compr.predict(scaled_data), 0)
#Z = np.clip(Z, -2, 1)
#Z = Z.reshape(xx.shape)
#print(bcolors.OKCYAN + str(Z) + bcolors.ENDC)
colours = ["yellow", "green", "blue", "red", "white"]
lime_color_dict = getLimeExplainer.getColorMap(4, ['0|1', '1|1', '0|0', '1|0'], default_colors=colours)
d=LabelEncoder()
#d.fit(np.array([x for x in color_dict[data].keys()]))
#d.fit(np.array(list(color_dict[data].keys()), dtype=float))
#z = d.transform(arr.astype(float))
#z = z.reshape(xx.shape)
z = arrCompr.reshape(xx.shape)
#z = arrPred.reshape(xx.shape)
ordering = [x for x in lime_color_dict[data].keys()]
keys = list(ordering)
ordering.sort()
ordering = [keys.index(x) for x in ordering]
#Evaluation:
if data == 'running1':
    dataname = '"Sine"'
else:
    dataname = '"Spiral"'
#pred = explainer.predict(test[cols].values)
#pred = lastCompr.predict(test[cols].values)
#with open(performancefile, 'a') as myfile:
#    line = ' '.join([dataname,
#                     '"Approach 1: Random sampling"',
#                     str(explainer.get_depth()),
#                     str(explainer.get_n_leaves()),
#                     str(metrics.accuracy_score(test.difference, pred)),
#                     str(metrics.precision_score(test.difference, pred, average='macro')),
#                     str(metrics.recall_score(test.difference, pred, average='macro'))
#                        str(lastScore),
#                     ])
#    myfile.write(line + '\n')




values = [x[0] for x in lime_color_dict[data].values()]
orderedmap = [values[i] for i in ordering]
MyCmap=colors.ListedColormap(orderedmap)
fig, ax = plt.subplots(figsize=(10, 8))

cntr1 = plot_contours(ax, modelA, xx, yy, levels=1, colors='black', linewidths=2, linestyles='dotted')
cntr2 = plot_contours(ax, modelB, xx, yy, levels=1, colors='black', linewidths=1)
#this puts in the data from the explainer
#ax.contour(xx, yy, Z, colors="orange")
#levels = np.linspace(z.min(), z.max(), 5)  # 4 contour levels
from matplotlib.colors import BoundaryNorm

levels = [-2.5, -1.5, -0.5, 0.5, 1.5]
norm = BoundaryNorm(levels, ncolors=4)

cp = ax.contourf(xx, yy, z, alpha=0.7, colors=colours, levels=levels, norm=norm)
cbar = plt.colorbar(cp, ticks=cp.levels)
print(bcolors.OKCYAN)
for i, collection in enumerate(cp.collections):
    print(f"Range {cp.levels[i]} to {cp.levels[i+1]}")
    print("Color:", collection.get_facecolor())
print(bcolors.ENDC)
#cp = ax.contourf(xx, yy, arr+0.1, alpha=0.7, cmap=MyCmap)
# ax.scatter(
#     explanationCenterValues[0], explanationCenterValues[1],
#     color="red",
#     s=80,
#     marker="x",
#     label="LIME anchor"
# )
def color_for_value(value):
    if value <= -1.5:
        return lime_color_dict[data]['0|1'][0]
    elif -1.5 < value <= -0.5:
        return lime_color_dict[data]['1|1'][0]
    elif -0.5 < value <= 0.5:
        return lime_color_dict[data]['0|0'][0]
    elif 0.5 < value <= 1.5:
        return lime_color_dict[data]['1|0'][0]
    else:
        return 'white'  # default color for out-of-range values

# for i in range(len(X1_np)):
#     ax.scatter(
#         X0_np[i], X1_np[i],
#         #color=color_for_value(arrCompr[i]),
#         color="black",
#         s=50,
#         marker="x"
#     )
# ax.scatter(
#     X0_np, X1_np,
#     color="black",
#     s=50,
#     marker="x"
# )

h = [plt.plot([], [], color=i[0], linewidth=10, label=j)[0] for j, i in lime_color_dict[data].items()]
ax.legend(handles=h, loc='lower left', title='Prediction Explainer', frameon=False, bbox_to_anchor=(0, -0.12),
          ncol=9)

ax.set_facecolor('#FFFFFF')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.grid(True, color='#F3F3F3')

ax.set_ylim(train.x2.min(), train.x2.max())
ax.set_xlim(train.x1.min(), train.x1.max())

plt.tight_layout()
plt.savefig('docout/sections/localtoglobal/resultsUSLIME/Approach1_decisionsurfaceexplainer_' + data + '.jpg', dpi=150, bbox_inches='tight', transparent=True, pad_inches=0)
fig, ax = plt.subplots(figsize=(55, 10))
class_names = ['difference']#explainer.classes_
colorss = ['#52854C']#[color_dict[data][x][0] for x in class_names]
N = len(class_names)
# artists = plot_tree(explainer, fontsize=8, ax=ax,
#                     impurity=False, node_ids=True,
#                     feature_names=cols, class_names=class_names)
# ax.properties()['children'] = [replace_text(i) for i in ax.properties()['children']]
# for artist, impurity, value in zip(artists, explainer.tree_.impurity, explainer.tree_.value):
#     # let the max value decide the color; whiten the color depending on impurity (gini)
#     r, g, b = to_rgb(colorss[np.argmax(value)])
#     f = impurity * N / (N - 1) if N > 1 else 0
#     artist.get_bbox_patch().set_facecolor((f + (1 - f) * r, f + (1 - f) * g, f + (1 - f) * b))
#     artist.get_bbox_patch().set_edgecolor('black')

# figure = lastExp.as_pyplot_figure()
# figure.savefig('docout/sections/localtoglobal/resultsUSLIME/Approach1_' + data + '_explainer.jpg')
#plt.savefig('docout/sections/localtoglobal/resultsUSLIME/Approach1_' + data + '_explainer.jpg', dpi=300, bbox_inches='tight',transparent=True, pad_inches=0)
# rules = get_rules(explainer, ['x_1', 'x_2'], class_names)
# rules = '\n'.join(rules)
# pyperclip.copy(rules)
### Plot for Approach 1.2
# populationsize = 1000
# # use tiny GP params for a quick run (overrides defaults in getglobal)
# gp_kwargs = {'ngen': 2, 'population_size': 20, 'max_steps': 1}
# globaldata = getglobal(train, classname='difference', npoints=8, modelA=modelA, modelB=modelB,
#                        populationsize=populationsize, method=stratifiedrandominstances,
#                        random_state=23, discrete=['difference'], #5
#                        continuous=['x1', 'x2'], gp_kwargs=gp_kwargs)
# globaldata.columns = ['x1', 'x2']
