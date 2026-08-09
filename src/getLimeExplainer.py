import math

from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import string
from USLime import USLime

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from itertools import combinations

def getLimeExplainer(lime_train, feature_names, class_names):
    r"""
    Get LIME explainer for comparison

    :return: None
    """

    explainer = LimeTabularExplainer(lime_train,
                        feature_names=feature_names,
                        class_names=class_names,
                        discretize_continuous=True)

    return explainer

def getColorMap(numOfColors, labels=None, default_colors=None, label='running1'):
    if default_colors is not None:
        return {label: dict(zip(labels, default_colors[:numOfColors]))}
    # Generate 100 gradient colors from a colormap
    cmap = plt.cm.plasma
    colors = [mcolors.to_hex(cmap(i)) for i in np.linspace(0, 1, numOfColors)]

    # Create 100 labels (a0, a1, ..., a99)
    if labels is None:
        labels = [f'a{i}' for i in range(numOfColors)]

    # Reuse marker styles (cycle through them)
    markers = ['o', 'v', '^', 's', 'P', 'D', '_', '|', '1']
    edge_colors = ['white', 'black']

    marker_styles = [
        [markers[i % len(markers)], edge_colors[i % len(edge_colors)]]
        for i in range(numOfColors)
    ]

    return {
        'running1': dict(zip(labels, zip(colors, marker_styles)))
    }


def getUSLimeExplainer(n_samples = 1000, n_neighbors = 50, noise_scale = 0.01, random_state = None):
    explainer = USLime(n_neighbors=n_neighbors, 
                       n_samples=n_samples, 
                       noise_scale=noise_scale, 
                       random_state=random_state)

    return explainer

#workingdata, sigma, min_neighbors, neighbors_dict, max_sigma, predA Train data, predB Train data, X_aug total Train data
def local_predict_combined(X_query, sigma=0.5, min_neighbors=3, neighbors_dict=None, max_sigma=1.0, 
                           predA=None, predB=None, X_aug=None, draw_point_scpecific = False, xx=None, yy=None, 
                           color_func=None, lines_to_draw=None):
    if neighbors_dict is None or predA is None or predB is None or X_aug is None or X_query is None or xx is None or yy is None or color_func is None:
        return None, None
    Xq_aug = X_query
    preds = []
    confs = []
    cnt = 0
    cnt2 = 0
    max = len(Xq_aug)
    base_sigma = sigma
    for xq in Xq_aug:
        cnt += 1
        sigma = base_sigma
        if (lines_to_draw is not None and cnt not in lines_to_draw): # 401, 1400, 900*, 894*, 881. | 1175*
            preds.append(4)
            confs.append(1.0)
            continue
        print("Predicting for query point " + str(cnt) + " out of " + str(max))
        dist, idx = neighbors_dict[sigma].radius_neighbors([xq])

        #print("Neighborhood size for sigma = " + str(sigma) + ": " + str(len(idx[0])))
        while len(idx[0]) <= min_neighbors and sigma < max_sigma: 
            #print("Neighborhood (" + str(len(idx[0])) + ") too small for sigma = " + str(sigma) + ", increasing sigma")
            sigma = math.ceil((sigma + 0.1)* 10) / 10
            dist, idx = neighbors_dict[sigma].radius_neighbors([xq])

        if len(idx[0]) == 0:
            dist_knn, idx_knn = neighbors_dict[sigma].kneighbors([xq], n_neighbors=1)
            nearest_idx = idx_knn[0][0]
            pA = predA[nearest_idx]
            pB = predB[nearest_idx]
            preds.append(pA * 2 + pB)
            confs.append(1.0)
            continue

        #print(predA[idx[0]])
        X_local = X_aug[idx[0]]
        y_local = predA[idx[0]] * 2 + predB[idx[0]]
        print(y_local)
        w = kernel(dist[0], sigma)

        mask = w > 0
        if not np.any(mask):
            nearest_idx = idx[0][0]
            pA = predA[nearest_idx]
            pB = predB[nearest_idx]
            preds.append(pA * 2 + pB)
            confs.append(1.0)
            continue

        X_local = X_local[mask]
        y_local = y_local[mask]
        w = w[mask]

        if len(np.unique(y_local)) < 2:
            p = np.bincount(y_local).argmax()
            confidence = 1.0
        else:
            cnt2 += 1
            models_local = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced")
            )
            models_local.fit(X_local, y_local, logisticregression__sample_weight=w)
            p = models_local.predict([xq])[0]
            print("p: " + str(p))
            if (draw_point_scpecific):
                drawlineforpoint(models_local, xq, xx)
                displayDataForPoint(xq, sigma)
                drawCircleForSigmaAtPoint(sigma, xq[0], xq[1])
                drawPoints(X_local, color_func)
            probs = models_local.predict_proba([xq])[0]
            #p = np.argmax(probs)
            confidence = np.max(probs)
            print("c: " + str(confidence))

        preds.append(p)
        confs.append(confidence)
    print("number of times in else block: " + str(cnt2))
    return np.array(preds), np.array(confs)

def kernel(dist, sigma):
    #distance normalized by the maximum distance (sigma)
    u = dist / sigma
    #epanechnikov
    return 0.75 * (1 - u**2) * (np.abs(u) <= 1)

def drawlineforpoint(models_local, point, xx=None):
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

            eq_str = getEquationString(w_orig, b_orig, point[0])
            above_below_label = getAboveBelowClasses(w_orig, class_names, c0, c1)
            label = above_below_label + " " + eq_str
            plt.plot(x_vals, y_vals, linewidth=2, color="black", label=label)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        return

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
        eq_str = getEquationString(w_diff, b_diff, point[0])
        above_below_label = getAboveBelowClasses(w_diff, class_names, cj, ci)
        label = above_below_label + " " + eq_str 
        plt.plot(x_vals, y_vals, linewidth=2, color=color, alpha=0.8, label=label)
    plt.legend()

def getAboveBelowClasses(w, class_names, c0, c1):
    # Determine which class is above and which is below based on the sign of w[1]
    # regression line: z = w[0] * x + w[1] * y + b
    # w[1] > 0 means that as y increases, z increases
    # z > 0 corresponds to class c1, and z < 0 corresponds to class c0
    # so z increasing with y increasing means c1 is above the line and c0 is below
    if w[1] > 0:
        return f"Above: {class_names[c1]} ↔ Below: {class_names[c0]}"
    else:
        return f"Above: {class_names[c0]} ↔ Below: {class_names[c1]}"
    
def getEquationString(w, b, x_val):
    # Calculate k and d for y = kx + d
    # decision boundary: 0 = w[0] * x + w[1] * y + b
    # solve for y: y = (-w[0] * x - b) / w[1]
    # split it up: y = (-w[0] / w[1]) * x + (-b / w[1])
    # =>           y =        k       * x -     d
    k = -w[0] / w[1]
    d = -b / w[1]
    point_value = k * x_val + d
    point_value_str = f"=> {point_value:.2f}"
    sign = "+" if d >= 0 else "-"
    return f"[y = {k:.2f}x {sign} {abs(d):.2f}] {point_value_str}"

def displayDataForPoint(point, sigma=None):
    info_text = "Query Point: ({:.4f}, {:.4f})\n".format(point[0], point[1])
    if sigma is not None:
        info_text += f"Sigma: {sigma:.2f}\n"

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

def drawCircleForSigmaAtPoint(sigma, pointX, pointY):
    circle = plt.Circle((pointX, pointY), sigma, color='gray', fill=False, linestyle='--', linewidth=1.5)
    plt.gca().add_artist(circle)

def drawPoints(points, color_func=None):
    for point in points:
        if color_func:
            color = color_func((point[0], point[1]))
        else:
            color = 'black'
        plt.plot(point[0], point[1], marker = 'x', color=color)