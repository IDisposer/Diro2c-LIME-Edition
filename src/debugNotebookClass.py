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
import getLimeExplainer
from lime.lime_tabular import LimeTabularExplainer
from sklearn.linear_model import LogisticRegression

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
performancefile = 'docout/sections/localtoglobal/resLIMEClass/FromLocalToGlobalrunning.txt'
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
### Plot for Approach 1.1
populationsize = 100000
# use tiny GP params for a quick run (overrides defaults in getglobal)
gp_kwargs = {'ngen': 2, 'population_size': 1000, 'max_steps': 1}
globaldata = getglobal(train, modelA=modelA, modelB=modelB, classname='difference', npoints=8,
                       populationsize=populationsize, method=randominstances,
                       random_state=0, discrete=['difference'],
                       continuous=['x1', 'x2'], gp_kwargs=gp_kwargs)
globaldata.columns = ['x1', 'x2']
with open('docout/sections/localtoglobal/resLIMEClass/approach1_neighborhood_' + data + '.pickle', 'wb') as handle:
    pickle.dump(globaldata, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('docout/sections/localtoglobal/resLIMEClass/approach1_neighborhood_' + data + '.pickle', 'rb') as handle:
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

plt.savefig('docout/sections/localtoglobal/resultsLIMEClass/Approach1_generatedneighborhoods_' + data + '.jpg', dpi=150,bbox_inches='tight',transparent=True, pad_inches=0)

k = int(len(globaldata.x1)/2)
gdTrain, gdTest = globaldata.iloc[:k, :], globaldata.iloc[k:, :]

class_names_2ClassProblem = ['difference']
class_names_4ClassProblem = ['A:0', 'A:1', 'B:0', 'B:1']

# explainer = LimeTabularExplainer(
#     training_data=gdTrain.values,
#     feature_names=list(gdTrain.columns),
#     class_names=class_names_4ClassProblem,
#     kernel_width=0.75,
#     mode='classification',
#     discretize_continuous=True
# )

amount_features = len(gdTrain.columns)
kernel_width = kernel_width = np.sqrt(amount_features) * 0.75
explainer = LimeTabularExplainer(
    training_data=gdTrain.values,
    feature_names=list(gdTrain.columns),
    kernel_width=kernel_width,
    discretize_continuous=False,
    class_names=['0', '1'],
    mode="classification",
    random_state = 42
)
print(str(gdTrain.values[0]))

X0, X1 = gdTest.x1, gdTest.x2
xx, yy = make_meshgridLinspace(X0, X1, h=0.5)
xravel = xx.ravel()
yravel = yy.ravel()
print(bcolors.WARNING + str(xravel.size) + bcolors.ENDC)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def diffOfArrays(a1, a2):
    return np.subtract(a1, a2)

def lime_surrogate_predict(exp, x):
    scores = []
    
    for class_idx in exp.local_exp:
        intercept = exp.intercept[class_idx]
        weights = dict(exp.local_exp[class_idx])
        
        s = intercept
        for feature_idx, value in enumerate(x):
            if feature_idx in weights:
                s += weights[feature_idx] * value
        
        scores.append(s)
    
    return np.array(scores)

def lime_surrogate_exact_all_classes(exp):
    class_indices = sorted(exp.local_exp.keys())
    x = exp.domain_mapper.scaled_row
    
    scores = []
    for class_idx in class_indices:
        intercept = exp.intercept[class_idx]
        weights = dict(exp.local_exp[class_idx])

        s = intercept
        for fid, w in weights.items():
            s += w * x[fid]

        scores.append(s)

    return np.array(scores)

index = 0
AorB = 0
compareArr = np.full((xravel.size, 2), -1, dtype=int)

def smooth_predict(model, X, noise=0.05, n_samples=30):
    preds = []
    for _ in range(n_samples):
        noisy = X + np.random.normal(0, noise, X.shape)
        preds.append(model.predict(noisy))
    
    probs = np.mean(preds, axis=0)
    return np.vstack([1 - probs, probs]).T

def compare_results(p, pog):
    result = 0
    for i in range(len(p)):
        a = np.argmax(p[i])
        b = pog[i]
        a = 1 if a == 0 else 1  #first argument is bigger => that means true
        if a == b:
            result += 1
    compareArr[index, AorB] = result

def predict_proba_wrapper(preds):
    return np.vstack([1 - preds, preds]).T

def predict_A(X):
    # p = logRegModel1.predict_proba(X)
    # pog = modelA.predict(X)
    # compare_results(p, pog)
    # print("PredA: " + str(pog.size) + "\n" + str(p))
    #return predict_proba_wrapper(modelA.predict(X))
    return smooth_predict(modelA, X)

def predict_B(X):
    # p = logRegModel2.predict_proba(X)
    # pog = modelB.predict(X)
    # compare_results(p, pog)
    # print("PredB: " + str(pog.size) + "\n" + str(p))
    #return predict_proba_wrapper(modelB.predict(X))
    return smooth_predict(modelB, X)

def extract_lime_line(exp, class_id):
    intercept = exp.intercept[class_id]
    weights = dict(exp.local_exp[class_id])
    
    w_x = weights.get(0, 0.0)
    w_y = weights.get(1, 0.0)
    
    # avoid divide-by-zero
    if abs(w_y) < 1e-8:
        return None  # vertical line
    
    slope = -w_x / w_y
    intercept_y = -intercept / w_y
    
    return slope, intercept_y

x_vals = np.linspace(min(xravel), max(xravel), 100)
lineArrA = np.empty((xravel.size, 2), dtype=object)
lineArrB = np.empty((xravel.size, 2), dtype=object)
pointArr = np.empty((xravel.size, 2), dtype=object)
def getExplanationsPrediction(explainer, function, instance, aOrB, i):
    ret_exp, scaled_data, inverse, compr = explainer.explain_instance(instance, function, num_features=amount_features, labels=(0, 1), num_samples=10000)
    #scaled_instance = ret_exp.domain_mapper.scaled_row
    scores = lime_surrogate_exact_all_classes(ret_exp)
    print(scores)
    print(ret_exp.score)
    print(ret_exp.predict_proba)
    # line0 = extract_lime_line(ret_exp, 0)
    # line1 = extract_lime_line(ret_exp, 1)
    # lineArr = None
    # if aOrB == 'A':
    #     lineArr = lineArrA
    #     colors = ['red', 'blue']
    # else:
    #     lineArr = lineArrB
    #     colors = ['orange', 'purple']

    # if line0:
    #     lineArr[i, 0] = (line0, colors[0])

    # if line1:
    #     lineArr[i, 1] = (line1, colors[1])
    return np.argmax(scores)

def drawLimeLines(lineArr, ax):
    for i in range(lineArr.shape[0]):
        for j in range(lineArr.shape[1]):
            if lineArr[i, j] is not None:
                (m, c), color = lineArr[i, j]
                ax.plot(x_vals, m * x_vals + c, color=color)

def drawLimePoints(pointArr, ax):
    valid_points = [p for p in pointArr if p is not None]
    for x, y in valid_points:
        ax.scatter(x, y, color='blue', zorder=10)
        circle = plt.Circle((x, y), kernel_width, fill=False, color='blue', linewidth=2, zorder=9)
        ax.add_patch(circle)

def drawLimePoints2(pointArr, ax, z, cmap):
    points = np.array([p[0] for p in pointArr])
    scores = np.array([p[1] for p in pointArr])

    x = points[:, 0]
    y = points[:, 1]
    vmin = z.min()
    vmax = z.max()

    sc = ax.scatter(
        x, y,
        c=scores,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors='k'
    )


arr = np.full(xravel.size, -1, dtype=int)

mapping = {
    (0,0): 0,
    (1,0): 1,
    (0,1): 2,
    (1,1): 3
}

print(bcolors.OKBLUE + "Starting explanation loop")
nodesToCheck = np.full(1, 4000000, dtype=int) #np.random.choice(xravel.size, size=100, replace=False)
for i in range(xravel.size):
#for i in nodesToCheck:
    print("In loop round " + str(i) + " out of " + str(xravel.size) + ". kernel_width: " + str(kernel_width))
    index = i
    instance = np.array([xravel[i], yravel[i]])
    #print(instance)
    AorB = 0
    predA = getExplanationsPrediction(explainer, predict_A, instance, 'A', i)
    AorB = 1
    predB = getExplanationsPrediction(explainer, predict_B, instance, 'B', i)
    arr[i] = mapping[(predA, predB)]
    pointArr[i] = ((instance[0], instance[1]), arr[i])
    # ret_exp, scaled_data, inverse, compr = explainer.explain_instance(instance, 
    #                                                                   diff_predict_fn_4Class, 
    #                                                                   num_features=4,
    #                                                                   labels=(0, 1, 2, 3))
    #preds = compr.predict(scaled_data[0].reshape(1, -1))
    #scaled_instance = ret_exp.domain_mapper.scaled_row
    #scores = lime_surrogate_predict(ret_exp, scaled_instance)
    #print(scores)
    #arr[i] = np.argmax(scores)
    #arr[i] = np.argmax(preds)
    #print(bcolors.OKGREEN + str(arr[i]) + bcolors.ENDC)
print(bcolors.ENDC)
#arr = arr[arr != -1]
freq_map = dict(zip(*np.unique(arr, return_counts=True)))
print(bcolors.OKGREEN)
print(freq_map)
for i in range(compareArr.shape[0]):
    if (compareArr[i, 0] != -1):
        print("Index: " + str(i) + " A correct: " + str(compareArr[i, 0]) + " B correct: " + str(compareArr[i, 1]))
print(bcolors.ENDC)

d=LabelEncoder()
#d.fit(np.array([x for x in color_dict[data].keys()]))
#d.fit(np.array(list(color_dict[data].keys()), dtype=float))
#z = d.transform(arr.astype(float))
#z = z.reshape(xx.shape)
z = arr.reshape(xx.shape)
z = np.ma.masked_where(z == -1, z)
ordering = [x for x in color_dict[data].keys()]
keys = list(ordering)
ordering.sort()
ordering = [keys.index(x) for x in ordering]
#Evaluation:
if data == 'running1':
    dataname = '"Sine"'
else:
    dataname = '"Spiral"'
#pred = explainer.predict(test[cols].values)
values = [x[0] for x in color_dict[data].values()]
orderedmap = [values[i] for i in ordering]
MyCmap=colors.ListedColormap(orderedmap)
fig, ax = plt.subplots(figsize=(10, 8))

cntr1 = plot_contours(ax, modelA, xx, yy, levels=1, colors='black', linewidths=2, linestyles='dotted')
cntr2 = plot_contours(ax, modelB, xx, yy, levels=1, colors='black', linewidths=1)

from matplotlib.colors import BoundaryNorm

levels = [-0.5, 0.5, 1.5, 2.5, 3.5]
colours = ["yellow", "green", "blue", "red", "white"]
norm = BoundaryNorm(levels, ncolors=4)

#cp = ax.contourf(xx, yy, z, alpha=0.7, cmap=MyCmap)
drawLimePoints2(pointArr, ax, z, MyCmap)
# drawLimeLines(lineArrA, ax)
# drawLimeLines(lineArrB, ax)
# drawLimePoints(pointArr, ax)
#cbar = plt.colorbar(cp, ticks=cp.levels)
#cp = ax.contourf(xx, yy, arr+0.1, alpha=0.7, cmap=MyCmap)

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
plt.savefig('docout/sections/localtoglobal/resultsLIMEClass/Approach1_decisionsurfaceexplainer_' + data + '.jpg', dpi=150, bbox_inches='tight', transparent=True, pad_inches=0)
fig, ax = plt.subplots(figsize=(55, 10))
class_names = ['difference']#explainer.classes_
colorss = ['#52854C']#[color_dict[data][x][0] for x in class_names]
N = len(class_names)

#figure = lastExp.as_pyplot_figure()
#figure.savefig('docout/sections/localtoglobal/resultsLIMEClass/Approach1_' + data + '_explainer.jpg')