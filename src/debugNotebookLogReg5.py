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


performancefile = 'docout/sections/localtoglobal/resLogReg5/FromLocalToGlobalrunning.txt'
data = 'running1'
train, cols = loaddata(data)
modelA = pickle.load(open('blackboxes/'+data+'A.sav', 'rb'))
modelB = pickle.load(open('blackboxes/'+data+'B.sav', 'rb'))

train['yA'] = modelA.predict(train[['x1', 'x2']].values)
train['yB'] = modelB.predict(train[['x1', 'x2']].values)
train['difference'] = train.apply(lambda row: '%g' % row['yA'] + '|' + '%g' % row['yB'], axis = 1)
train.drop(columns=['yA', 'yB'], inplace=True)

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
gp_kwargs = {'ngen': 2, 'population_size': 5000, 'max_steps': 1}
globaldata = getglobal(train, modelA=modelA, modelB=modelB, classname='difference', npoints=8,
                       populationsize=populationsize, method=randominstances,
                       random_state=3, discrete=['difference'],
                       continuous=['x1', 'x2'], gp_kwargs=gp_kwargs)
globaldata.columns = ['x1', 'x2']
with open('docout/sections/localtoglobal/resLogReg5/approach1_neighborhood_' + data + '.pickle', 'wb') as handle:
    pickle.dump(globaldata, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('docout/sections/localtoglobal/resLogReg5/approach1_neighborhood_' + data + '.pickle', 'rb') as handle:
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

plt.savefig('docout/sections/localtoglobal/resultsLogReg5/Approach1_generatedneighborhoods_' + data + '.jpg', dpi=150,bbox_inches='tight',transparent=True, pad_inches=0)

k = int(len(globaldata.x1)/2)
gdTrain, gdTest = globaldata.iloc[:k, :], globaldata.iloc[k:, :]

class_names_2ClassProblem = ['difference']
class_names_4ClassProblem = ['A:0', 'A:1', 'B:0', 'B:1']

print(str(gdTrain.values[0]))

predA = modelA.predict(gdTrain.values)
predB = modelB.predict(gdTrain.values)

y_meta = (predA * 2 + predB)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors
import time
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from itertools import combinations

def kernel(dist, bandwidth, kind="gaussian"):
    u = dist / bandwidth

    if kind == "gaussian":
        return np.exp(-0.5 * u**2)

    elif kind == "uniform":
        return (np.abs(u) <= 1).astype(float)

    elif kind == "triangular":
        return np.maximum(0, 1 - np.abs(u))

    elif kind == "epanechnikov":
        return np.maximum(0, 1 - u**2)

    elif kind == "quartic":
        return np.maximum(0, (1 - u**2)**2)

    else:
        raise ValueError("Unknown kernel")

X_aug = gdTrain.values
global_sigma = 0.8
nbrs = NearestNeighbors(radius=global_sigma)
nbrs.fit(X_aug)
def local_predict_combined(X_query, sigma=global_sigma):
    Xq_aug = X_query
    preds = []
    confs = []
    cnt = 0
    cnt2 = 0
    max = len(Xq_aug)
    for xq in Xq_aug:
        cnt += 1
        if (cnt != 879): # 401, 1400, 900*, 894*, 881.
            preds.append(4)
            continue
        print("Predicting for query point " + str(cnt) + " out of " + str(max))
        dist, idx = nbrs.radius_neighbors([xq])

        if len(idx[0]) == 0:
            print("in zeroth if")
            dist_knn, idx_knn = nbrs.kneighbors([xq], n_neighbors=1)
            nearest_idx = idx_knn[0][0]

            pA = predA[nearest_idx]
            pB = predB[nearest_idx]
            preds.append(pA * 2 + pB)
            #confs.append(1.0)
            continue
        #print(predA[idx[0]])
        X_local = X_aug[idx[0]]
        y_local = predA[idx[0]] * 2 + predB[idx[0]]
        print(y_local)
        w = kernel(dist[0], sigma, kind="epanechnikov")

        mask = w > 0
        if not np.any(mask):
            print("in first if")
            nearest_idx = idx[0][0]
            pA = predA[nearest_idx]
            pB = predB[nearest_idx]
            preds.append(pA * 2 + pB)
            #confs.append(1.0)
            continue

        X_local = X_local[mask]
        y_local = y_local[mask]
        w = w[mask]

        if len(np.unique(y_local)) < 2:
            p = np.bincount(y_local).argmax()
            print("in second if")
            confidence = 1.0
        else:
            cnt2 += 1
            models_local = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced")
            )
            models_local.fit(X_local, y_local, logisticregression__sample_weight=w)
            p = models_local.predict([xq])[0]
            drawlineforpoint4(models_local, xq)
            displayDataForPoint(xq)
            #probs = models_local.predict_proba([xq])[0]
            #p = np.argmax(probs)
            #confidence = np.max(probs)

        preds.append(p)
        #confs.append(confidence)
    print("number of times in else block: " + str(cnt2))
    return np.array(preds)#, np.array(confs)

def drawlineforpoint4(models_local, point):
    model = models_local.named_steps['logisticregression']
    scaler = models_local.named_steps['standardscaler']

    class_names = {
        0: "0|0",
        1: "0|1",
        2: "1|0",
        3: "1|1"
    }

    classes = model.classes_
    x_vals = np.linspace(xx.min(), xx.max(), 300)

    # 1. Binary Case
    if model.coef_.shape[0] == 1:
        w = model.coef_[0]
        b = model.intercept_[0]

        mean = scaler.mean_
        scale = scaler.scale_

        w_orig = w / scale
        b_orig = b - np.sum((w * mean) / scale)

        if abs(w_orig[1]) > 1e-12:
            y_vals = -(w_orig[0] * x_vals + b_orig) / w_orig[1]
            c0, c1 = classes

            # Calculate k and d for y = kx + d
            k = -w_orig[0] / w_orig[1]
            d = -b_orig / w_orig[1]
            sign = "+" if d >= 0 else "-"
            pointValue = k * point[0] + d
            pointValueStr = f"=> {pointValue:.2f}"
            eq_str = f" [y = {k:.2f}x {sign} {abs(d):.2f}] {pointValueStr}"

            if w_orig[1] > 0:
                label = f"Above: {class_names[c1]} ↔ Below: {class_names[c0]}" + eq_str
            else:
                label = f"Above: {class_names[c0]} ↔ Below: {class_names[c1]}" + eq_str

            plt.plot(x_vals, y_vals, linewidth=2, color="black", label=label)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        return

    # 2. Multiclass Case
    mean = scaler.mean_
    scale = scaler.scale_

    W = model.coef_ / scale
    B = model.intercept_ - np.sum(model.coef_ * mean / scale, axis=1)

    pair_colors = {
        (0, 1): "magenta",
        (0, 2): "cyan",
        (0, 3): "blue",
        (1, 2): "orange",
        (1, 3): "red",
        (2, 3): "green",
    }

    for i, j in combinations(range(len(classes)), 2):
        wi, wj = W[i], W[j]
        bi, bj = B[i], B[j]

        w_diff = wi - wj
        b_diff = bi - bj

        if abs(w_diff[1]) < 1e-12:
            continue

        y_vals = -(w_diff[0] * x_vals + b_diff) / w_diff[1]
        ci, cj = classes[i], classes[j]

        color = pair_colors.get(tuple(sorted((ci, cj))), "black")

        # Calculate k and d for y = kx + d
        k = -w_diff[0] / w_diff[1]
        d = -b_diff / w_diff[1]
        sign = "+" if d >= 0 else "-"
        pointValue = k * point[0] + d
        pointValueStr = f"=> {pointValue:.2f}"
        eq_str = f" [y = {k:.2f}x {sign} {abs(d):.2f}] {pointValueStr}"

        if w_diff[1] > 0:
            label = f"Above: {class_names[ci]} ↔ Below: {class_names[cj]}" + eq_str
        else:
            label = f"Above: {class_names[cj]} ↔ Below: {class_names[ci]}" + eq_str

        plt.plot(x_vals, y_vals, linewidth=2, color=color, alpha=0.8, label=label)

    plt.legend()

def displayDataForPoint(point):
    info_text = "Query Point: ({:.4f}, {:.4f})\n".format(point[0], point[1])

    plt.gca().text(
        0.8, 0.8,                  # X and Y positions (5% from left, 95% from bottom)
        info_text, 
        transform=plt.gca().transAxes, # Lock position to the plot box layout, not data coordinates
        fontsize=10, 
        verticalalignment='top',     # Anchor text from its top edge
        horizontalalignment='left',   # Anchor text from its left edge
        bbox=dict(                   # Style the box container
            boxstyle="round,pad=0.5", 
            facecolor="white", 
            edgecolor="gray", 
            alpha=0.9
        )
    )

#setting base plot
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
#end of setting base plot


X0, X1 = gdTest.x1, gdTest.x2
xx, yy = make_meshgridLinspace(X0, X1, h=0.5)
xravel = xx.ravel()
yravel = yy.ravel()
print(bcolors.WARNING + str(xravel.size) + bcolors.ENDC)
grid = np.c_[xravel, yravel]

grid_aug = grid


print(bcolors.OKBLUE + "Starting local predictions on grid")
start_time = time.perf_counter()
print("Time before predictions: " + str(start_time))
Z = local_predict_combined(grid)
end_time = time.perf_counter()
print("Time after predictions: " + str(end_time))
print("Time taken for local predictions: " + str(end_time - start_time) + " seconds")
print("Finished local predictions on grid" + bcolors.ENDC)
Z = Z.reshape(xx.shape)
#C = C.reshape(xx.shape)

predA = modelA.predict(gdTest.values)
predB = modelB.predict(gdTest.values)

#y_meta = (predA * 2 + predB)

cmap = ListedColormap(["blue", "red", "green", "purple", "white"])
labels = ["0|0", "0|1", "1|0", "1|1", "na"]
colors = ["blue", "red", "green", "purple", "white"]

patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(5)]

#plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
plt.scatter(xx.ravel(), yy.ravel(), c=Z.ravel(), cmap=cmap, alpha=0.6)
#plt.contourf(xx, yy, C, levels=20, cmap="viridis", alpha=0.6)

#plt.colorbar(label="Confidence")
plt.scatter(gdTrain.x1, gdTrain.x2, c=y_meta, cmap=cmap, edgecolor="k")
#plt.scatter([p[0] for p in linepoints], [p[1] for p in linepoints], c='orange', edgecolor="k")
#plt.legend(handles=patches)

plt.title("Regions where models disagree")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
plt.savefig('docout/sections/localtoglobal/resultsLogReg5/Approach1_decisionsurfaceexplainer_' + data + '.jpg', dpi=150, bbox_inches='tight', transparent=True, pad_inches=0)

