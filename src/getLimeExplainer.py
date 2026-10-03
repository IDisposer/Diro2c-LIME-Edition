import math

import sklearn
from sklearn.multiclass import OneVsRestClassifier

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

def local_predict_improved(X_query, sigma=0.5, min_neighbors=3, neighbors_dict=None, max_sigma=1.0, 
                           predA=None, predB=None, X_aug=None, draw_point_specific = False, xx=None, yy=None, 
                           color_func=None, lines_to_draw=None, allowed_classes=None):
    if neighbors_dict is None or predA is None or predB is None or X_aug is None or X_query is None or xx is None or yy is None or color_func is None:
        return None, None
    #sklearn.set_config(enable_metadata_routing=True)
    Xq_aug = X_query
    preds = []
    confs = []
    cnt = 0
    cnt2 = 0
    max = len(Xq_aug)
    base_sigma = sigma
    for xq in Xq_aug:
        cnt += 1
        #print(str(cnt) + " at [" + str(xq[0]) + ", " + str(xq[1]) + "]")
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
        #print(y_local)
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
            #logReg = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear").set_fit_request(sample_weight=True)
            #logReg = LogisticRegression(max_iter=2000, class_weight="balanced")
            models_local = make_pipeline(
                StandardScaler(),#.set_fit_request(sample_weight=False),
                LogisticRegression(max_iter=2000, class_weight="balanced")
            )
            #models_local = OneVsRestClassifier(models_local)
            #models_local.fit(X_local, y_local, sample_weight=w)
            models_local.fit(X_local, y_local, logisticregression__sample_weight=w)
            p = models_local.predict([xq])[0]
            print("p: " + str(p))
            if (draw_point_specific):
                if (lines_to_draw is not None and len(lines_to_draw) == 1):
                    displayDataForPoint(xq, sigma)
                #drawlineforpointOneVsRest(models_local, xq, X_local, color_func)
                extract_multinomial_ranges_universal(models_local, X_local, allowed_classes)
                drawCircleForSigmaAtPoint(sigma, xq[0], xq[1])
                #drawPoints(X_local, color_func)
            probs = models_local.predict_proba([xq])[0]
            #p = np.argmax(probs)
            confidence = np.max(probs)
            #print("c: " + str(confidence))

        preds.append(p)
        confs.append(confidence)
    print("number of times in else block: " + str(cnt2))
    plt.legend()
    return np.array(preds), np.array(confs)

def local_predict_improved_provide_models(X_query, sigma=0.5, min_neighbors=3, neighbors_dict=None, max_sigma=1.0, 
                           predA=None, predB=None, X_aug=None, draw_point_specific = False, xx=None, yy=None, 
                           color_func=None, lines_to_draw=None, allowed_classes=None):
    if neighbors_dict is None or predA is None or predB is None or X_aug is None or X_query is None or xx is None or yy is None or color_func is None:
        return None, None
    #sklearn.set_config(enable_metadata_routing=True)
    Xq_aug = X_query
    preds = []
    confs = []
    models_used = []
    cnt = 0
    cnt2 = 0
    max = len(Xq_aug)
    base_sigma = sigma
    for xq in Xq_aug:
        cnt += 1
        #print(str(cnt) + " at [" + str(xq[0]) + ", " + str(xq[1]) + "]")
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
        #print(y_local)
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
            #logReg = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear").set_fit_request(sample_weight=True)
            #logReg = LogisticRegression(max_iter=2000, class_weight="balanced")
            models_local = make_pipeline(
                StandardScaler(),#.set_fit_request(sample_weight=False),
                LogisticRegression(max_iter=2000, class_weight="balanced")
            )
            #models_local = OneVsRestClassifier(models_local)
            #models_local.fit(X_local, y_local, sample_weight=w)
            models_local.fit(X_local, y_local, logisticregression__sample_weight=w)
            p = models_local.predict([xq])[0]
            #print("p: " + str(p))
            if (draw_point_specific):
                #displayDataForPoint(xq, sigma)
                #drawlineforpointOneVsRest(models_local, xq, X_local, color_func)
                extract_multinomial_ranges_universal(models_local, X_local, allowed_classes)
                drawCircleForSigmaAtPoint(sigma, xq[0], xq[1])
                #drawPoints(X_local, color_func)
            probs = models_local.predict_proba([xq])[0]
            #p = np.argmax(probs)
            confidence = np.max(probs)
            #print("c: " + str(confidence))
            models_used.append(models_local)

        preds.append(p)
        confs.append(confidence)
    print("number of times in else block: " + str(cnt2))
    plt.legend()
    return np.array(preds), np.array(confs), models_used

def local_predict_moredimensional(X_query, sigma=0.5, min_neighbors=3, neighbors_dict=None, max_sigma=1.0, 
                                  X_local=None, Y_local=None, lines_to_draw=None):
    models_used = {}
    Xq_aug = X_query
    preds = []
    confs = []
    cnt = 0
    cnt2 = 0
    max = len(Xq_aug)
    base_sigma = sigma
    for xq in Xq_aug:
        cnt += 1
        #print(str(cnt) + " at [" + str(xq[0]) + ", " + str(xq[1]) + "]")
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
            preds.append(Y_local[nearest_idx])
            confs.append(1.0)
            continue

        #print(predA[idx[0]])
        x_local = X_local[idx[0]]
        y_local = Y_local[idx[0]]
        print(y_local)
        w = kernel(dist[0], sigma)

        mask = w > 0
        if not np.any(mask):
            nearest_idx = idx[0][0]
            preds.append(Y_local[nearest_idx])
            confs.append(1.0)
            continue

        x_local = x_local[mask]
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
            models_local.fit(x_local, y_local, logisticregression__sample_weight=w)
            models_used[cnt] = {'model': models_local, 'x_local': x_local}
            p = models_local.predict([xq])[0]
            print("p: " + str(p))
            probs = models_local.predict_proba([xq])[0]
            #p = np.argmax(probs)
            confidence = np.max(probs)
            print("c: " + str(confidence))

        preds.append(p)
        confs.append(confidence)
    print("number of times in else block: " + str(cnt2))
    return np.array(preds), np.array(confs), models_used

def kernel(dist, sigma):
    #distance normalized by the maximum distance (sigma)
    u = dist / sigma
    #epanechnikov
    #return 0.75 * (1 - u**2) * (np.abs(u) <= 1)
    #gaussian
    return np.exp(-0.5 * u**2)

def drawlineforpointOneVsRest(models_local, point, X_local, pred_func):
    class_names = {
            0: "0|0",
            1: "0|1",
            2: "1|0",
            3: "1|1"
        }
    raw_lines = {}
    n_estimators = len(models_local.estimators_)

    # --- SECTION 1: FIXING THE EXTRACTION LOGIC ---
    if n_estimators == 1:
        # BINARY CASE (2 classes in data): They share one estimator, but signs are flipped
        trained_pipeline = models_local.estimators_[0]
        scaler = trained_pipeline.named_steps['standardscaler']
        lr_model = trained_pipeline.named_steps['logisticregression']
        
        # Flatten using .ravel() to guarantee 1D arrays
        w_scaled = lr_model.coef_.ravel()
        b_scaled = lr_model.intercept_.ravel()
        
        mu = scaler.mean_
        sigma = scaler.scale_
        w_raw = w_scaled / sigma
        b_raw = b_scaled - np.sum((w_scaled * mu) / sigma)
        
        class_0 = models_local.classes_[0]
        class_1 = models_local.classes_[1]
        
        # Cast explicitly to float() to strip away numpy array wrapping
        raw_lines[class_0] = {'w1': float(-w_raw[0]), 'w2': float(-w_raw[1]), 'b': float(-b_raw[0])}
        raw_lines[class_1] = {'w1': float(w_raw[0]), 'w2': float(w_raw[1]), 'b': float(b_raw[0])}

    else:
        # MULTICLASS CASE (3+ classes in data): 1 estimator per class
        for idx, class_label in enumerate(models_local.classes_):
            trained_pipeline = models_local.estimators_[idx]
            scaler = trained_pipeline.named_steps['standardscaler']
            lr_model = trained_pipeline.named_steps['logisticregression']
            
            w_scaled = lr_model.coef_.ravel()
            b_scaled = lr_model.intercept_.ravel()
            
            mu = scaler.mean_
            sigma = scaler.scale_
            w_raw = w_scaled / sigma
            b_raw = b_scaled - np.sum((w_scaled * mu) / sigma)
            
            # Cast explicitly to float() to make 'scores' comparison work smoothly
            raw_lines[class_label] = {'w1': float(w_raw[0]), 'w2': float(w_raw[1]), 'b': float(b_raw[0])}


    # --- SECTION 2: GRID EVALUATION ---
    x1_min, x1_max = X_local[:, 0].min() - 1, X_local[:, 0].max() + 1
    x2_min, x2_max = X_local[:, 1].min() - 1, X_local[:, 1].max() + 1
    x1_vals = np.linspace(x1_min, x1_max, 1000)

    y_pred_func = np.array([])
    for x in X_local:
        y_pred_func = np.append(y_pred_func, pred_func((x[0], x[1])))

    for c1, c2 in combinations(models_local.classes_, 2):
        line1 = raw_lines[c1]
        line2 = raw_lines[c2]
        
        dw1 = line1['w1'] - line2['w1']
        dw2 = line1['w2'] - line2['w2']
        db  = line1['b'] - line2['b']
        
        if np.isclose(dw2, 0):
            continue
            
        valid_x1 = []
        valid_x2 = []

        k = -dw1 / dw2
        d = -db / dw2
        
        for x1 in x1_vals:
            x2 = (k * x1) + d
            if x2 < x2_min or x2 > x2_max:
                continue
            
            # This dictionary now contains pure floats, making max() and tie checks valid
            scores = {c: raw_lines[c]['w1']*x1 + raw_lines[c]['w2']*x2 + raw_lines[c]['b'] 
                    for c in models_local.classes_}
            
            highest_score = max(scores.values())
            
            # Verify the tie
            if np.isclose(scores[c1], highest_score, atol=1e-5) and np.isclose(scores[c2], highest_score, atol=1e-5):
                valid_x1.append(x1)
                valid_x2.append(x2)
                
        if valid_x1:
            x1_start, x1_end = min(valid_x1), max(valid_x1)
            
            # --- Integrated Error Calculation Step ---
            wrong_points_in_range = 0
            
            for i in range(len(X_local)):
                px1 = X_local[i, 0]
                px2 = X_local[i, 1]
                
                # Check if the data point falls into this line's valid horizontal range slice
                if x1_start <= px1 <= x1_end:
                    # Calculate model's preferred winner at this specific spot
                    pt_scores = {c: raw_lines[c]['w1']*px1 + raw_lines[c]['w2']*px2 + raw_lines[c]['b'] 
                                 for c in models_local.classes_}
                    line_winner = max(pt_scores, key=pt_scores.get)
                    
                    # If the formula prediction doesn't match your custom function, it's a mistake
                    if line_winner != y_pred_func[i]:
                        wrong_points_in_range += 1


            sign = "+" if d >= 0 else "-"
            formula_str = f"y = {k:.2f}x {sign} {abs(d):.2f}"

            lbl = f'{class_names[c1]} vs {class_names[c2]} | [{min(valid_x1):.2f}, {max(valid_x1):.2f}] | {formula_str} | {point[0]*k + d:.2f} | Errors: {wrong_points_in_range}'
            plt.plot(valid_x1, valid_x2, label=lbl, linewidth=3)
            print(f"Boundary line between Class {class_names[c1]} and {class_names[c2]} is valid for X1 range: [{min(valid_x1):.2f}, {max(valid_x1):.2f}]")
    plt.legend()

def extract_multinomial_ranges_universal(pipeline_model, X_local, allowed_classes=None):
    class_names = {0: "0|0", 1: "0|1", 2: "1|0", 3: "1|1"}
    if allowed_classes is None:
        allowed_classes = [0, 1, 2, 3]
    
    scaler = pipeline_model.named_steps['standardscaler']
    lr_model = pipeline_model.named_steps['logisticregression']
    
    classes = lr_model.classes_
    n_classes = len(classes)
    
    mu = scaler.mean_
    sigma = scaler.scale_
    x1_min, x1_max = X_local[:, 0].min(), X_local[:, 0].max()
    x2_min, x2_max = X_local[:, 1].min(), X_local[:, 1].max()
    if n_classes == 2:
        c1, c2 = classes[0], classes[1]
        if c1 not in allowed_classes or c2 not in allowed_classes:
            return
        w_scaled = lr_model.coef_.ravel()
        b_scaled = lr_model.intercept_.ravel()
        w_raw = w_scaled / sigma
        b_raw = b_scaled - np.sum((w_scaled * mu) / sigma)
        
        dw1, dw2, db = w_raw[0], w_raw[1], b_raw[0]
        
        if not np.isclose(dw2, 0):
            k = -dw1 / dw2
            d = -db / dw2

            sign = "+" if d >= 0 else "-"
            label_str = f"{class_names[c1]} vs {class_names[c2]} | y = {k:.2f}x {sign} {abs(d):.2f} | Range: [({x1_min:.2f}, {x2_min:.2f}), ({x1_max:.2f}, {x2_max:.2f})]"
            plot_x1 = np.linspace(x1_min, x1_max, 20)
            plot_x1 = plot_x1[((k * plot_x1) + d >= x2_min) & ((k * plot_x1) + d <= x2_max)]
            plot_x2 = (k * plot_x1) + d
            
            label=label_str
            plt.plot(plot_x1, plot_x2, linewidth=3, label=label)
            print(f"Binary Boundary [{class_names[c1]} vs {class_names[c2]}] spans full X1: [{x1_min:.2f}, {x1_max:.2f}]")

    else:
        coef_scaled = lr_model.coef_
        intercept_scaled = lr_model.intercept_
        
        w_raw = coef_scaled / sigma
        b_raw = intercept_scaled - np.sum((coef_scaled * mu) / sigma, axis=1)
        
        for c1_idx, c2_idx in combinations(range(n_classes), 2):
            c1, c2 = classes[c1_idx], classes[c2_idx]
            if c1 not in allowed_classes or c2 not in allowed_classes:
                continue

            #two regression lines that have different multipliers and results
            #the decision boundary is where the two lines have the same value 
            #set the formulas equal and move all terms to one side to get the equation of the line

            dw1 = w_raw[c1_idx, 0] - w_raw[c2_idx, 0]
            dw2 = w_raw[c1_idx, 1] - w_raw[c2_idx, 1]
            db  = b_raw[c1_idx] - b_raw[c2_idx]
            
            if np.isclose(dw2, 0):
                continue
                
            k = -dw1 / dw2
            d = -db / dw2
            
            valid_segments_x1 = []
            test_x1_points = np.linspace(x1_min, x1_max, 500)
            
            for x1 in test_x1_points:
                x2 = (k * x1) + d
                if x2 < x2_min or x2 > x2_max:
                    continue
                scores = [w_raw[i, 0]*x1 + w_raw[i, 1]*x2 + b_raw[i] for i in range(n_classes)]
                highest_score = max(scores)
                
                if np.isclose(scores[c1_idx], highest_score, atol=1e-4) and np.isclose(scores[c2_idx], highest_score, atol=1e-4):
                    valid_segments_x1.append(x1)
            print("Valid segments amount: " + str(len(valid_segments_x1)))
            if valid_segments_x1:
                start_x1 = min(valid_segments_x1)
                end_x1 = max(valid_segments_x1)
                
                sign = "+" if d >= 0 else "-"
                label_str = f"{class_names[c1]} vs {class_names[c2]} | y = {k:.2f}x {sign} {abs(d):.2f} | Range: [({start_x1:.2f}, {x2_min:.2f}), ({end_x1:.2f}, {x2_max:.2f})]"
                
                plot_x1 = np.linspace(start_x1, end_x1, 20)
                plot_x2 = (k * plot_x1) + d

                label=label_str
                plt.plot(plot_x1, plot_x2, linewidth=3, label=label)
                print(f"Boundary [{class_names[c1]} vs {class_names[c2]}] is strictly valid for X1: [{start_x1:.2f}, {end_x1:.2f}]")
        plt.legend()

def extract_multinomial_ranges_universal_multimodel(pipeline_model, X_local, allowed_classes=None, models_to_compare=[]):
    class_names = {0: "0|0", 1: "0|1", 2: "1|0", 3: "1|1"}
    if allowed_classes is None:
        allowed_classes = [0, 1, 2, 3]
    
    scaler = pipeline_model.named_steps['standardscaler']
    lr_model = pipeline_model.named_steps['logisticregression']
    
    classes = lr_model.classes_
    n_classes = len(classes)
    
    mu = scaler.mean_
    sigma = scaler.scale_
    x1_min, x1_max = X_local[:, 0].min(), X_local[:, 0].max()
    x2_min, x2_max = X_local[:, 1].min(), X_local[:, 1].max()
    k = 0
    d = 0
    if n_classes == 2:
        c1, c2 = classes[0], classes[1]
        if c1 not in allowed_classes or c2 not in allowed_classes:
            return
        w_scaled = lr_model.coef_.ravel()
        b_scaled = lr_model.intercept_.ravel()
        w_raw = w_scaled / sigma
        b_raw = b_scaled - np.sum((w_scaled * mu) / sigma)
        
        dw1, dw2, db = w_raw[0], w_raw[1], b_raw[0]
        
        if not np.isclose(dw2, 0):
            k = -dw1 / dw2
            d = -db / dw2

    else:
        coef_scaled = lr_model.coef_
        intercept_scaled = lr_model.intercept_
        
        w_raw = coef_scaled / sigma
        b_raw = intercept_scaled - np.sum((coef_scaled * mu) / sigma, axis=1)
        
        for c1_idx, c2_idx in combinations(range(n_classes), 2):
            c1, c2 = classes[c1_idx], classes[c2_idx]
            if c1 not in allowed_classes or c2 not in allowed_classes:
                continue

            #two regression lines that have different multipliers and results
            #the decision boundary is where the two lines have the same value 
            #set the formulas equal and move all terms to one side to get the equation of the line

            dw1 = w_raw[c1_idx, 0] - w_raw[c2_idx, 0]
            dw2 = w_raw[c1_idx, 1] - w_raw[c2_idx, 1]
            db  = b_raw[c1_idx] - b_raw[c2_idx]
            
            if np.isclose(dw2, 0):
                continue
                
            k = -dw1 / dw2
            d = -db / dw2

        valid_segments_x1 = []
        test_x1_points = np.linspace(x1_min, x1_max, 500)
        
        for x1 in test_x1_points:
            x2 = (k * x1) + d
            if x2 < x2_min or x2 > x2_max:
                continue
            scores = [w_raw[i, 0]*x1 + w_raw[i, 1]*x2 + b_raw[i] for i in range(n_classes)]
            highest_score = max(scores)
            hightest_score_across_models = get_max_score_from_all_models(x1, x2, models_to_compare)
            highest_score = max(highest_score, hightest_score_across_models)
            
            if np.isclose(scores[c1_idx], highest_score, atol=1e-4) and np.isclose(scores[c2_idx], highest_score, atol=1e-4):
                valid_segments_x1.append(x1)
        if valid_segments_x1:
            start_x1 = min(valid_segments_x1)
            end_x1 = max(valid_segments_x1)
            
            sign = "+" if d >= 0 else "-"
            label_str = f"{class_names[c1]} vs {class_names[c2]} | y = {k:.2f}x {sign} {abs(d):.2f} | Range: [{start_x1:.2f}, {end_x1:.2f}]"
            
            plot_x1 = np.linspace(start_x1, end_x1, 10)
            plot_x2 = (k * plot_x1) + d

            label=label_str
            plt.plot(plot_x1, plot_x2, linewidth=3)

def get_highscore_from_model(x, y, model):
    scaler = model.named_steps['standardscaler']
    lr_model = model.named_steps['logisticregression']
    
    mu = scaler.mean_
    sigma = scaler.scale_
    
    w_scaled = lr_model.coef_
    b_scaled = lr_model.intercept_
    
    w_raw = w_scaled / sigma
    b_raw = b_scaled - np.sum((w_scaled * mu) / sigma, axis=1)
    
    scores = [x * w_raw[i, 0] + y * w_raw[i, 1] + b_raw[i] for i in range(len(b_raw))]
    return max(scores)

def get_max_score_from_all_models(x, y, models):
    max_score = float('-inf')
    for model in models:
        score = get_highscore_from_model(x, y, model)
        if score > max_score:
            max_score = score
    return max_score

def extract_multinomial_ranges_universal_moredimensional(pipeline_model, X_local, allowed_classes=None):
    if allowed_classes is None:
        allowed_classes = [0, 1, 2, 3]
    
    scaler = pipeline_model.named_steps['standardscaler']
    lr_model = pipeline_model.named_steps['logisticregression']
    
    classes = lr_model.classes_
    n_classes = len(classes)
    
    mu = scaler.mean_
    sigma = scaler.scale_
    formulas = {}
    if n_classes == 2:
        c1, c2 = classes[0], classes[1]
        if c1 not in allowed_classes or c2 not in allowed_classes:
            return
        w_scaled = lr_model.coef_.ravel()
        b_scaled = lr_model.intercept_.ravel()
        w_raw = w_scaled / sigma
        b_raw = b_scaled - np.sum((w_scaled * mu) / sigma)
        
        formula_terms = [f"{coeff:.4f} * x{j}" for j, coeff in enumerate(w_raw)]
        formula = " + ".join(formula_terms) + f" + {b_raw[0]:.4f}"
        formulas[(c1, c2)] = formula    

    else:
        coef_scaled = lr_model.coef_
        intercept_scaled = lr_model.intercept_
        
        w_raw = coef_scaled / sigma
        b_raw = intercept_scaled - np.sum((coef_scaled * mu) / sigma, axis=1)
        
        for c1_idx, c2_idx in combinations(range(n_classes), 2):
            c1, c2 = classes[c1_idx], classes[c2_idx]
            if c1 not in allowed_classes or c2 not in allowed_classes:
                continue

            coeffs = w_raw[c1_idx] - w_raw[c2_idx]
            intercept = b_raw[c1_idx] - b_raw[c2_idx]
            formula_terms = [f"{coeff:.4f} * x{j}" for j, coeff in enumerate(coeffs)]
            formula = " + ".join(formula_terms) + f" + {intercept:.4f}"
            formulas[(c1, c2)] = formula  
    return formulas, getMaxAndMinOfEachFeature(X_local)

def getMaxAndMinOfEachFeature(x_local):
    max_min_dict = {}
    for i in range(x_local.shape[1]):
        feature_values = x_local[:, i]
        max_value = np.max(feature_values)
        min_value = np.min(feature_values)
        max_min_dict[i] = (min_value, max_value)
    return max_min_dict      

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
        0.1, 0.95,                  # X and Y positions (5% from left, 95% from bottom)
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
    color_map = {
        0: 'blue',
        1: 'red',
        2: 'green',
        3: 'purple',
        4: 'black'
    }
    for point in points:
        if color_func:
            color = color_map[color_func((point[0], point[1]))]
        else:
            color = 'black'
        plt.plot(point[0], point[1], marker = 'x', color=color)

def expandDataset(data, m, scale=0.1, seed=None):
    data = np.asarray(data)
    num_points, dimensions = data.shape
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=scale, size=(num_points, m, dimensions))
    synthetic_points = data[:, np.newaxis, :] + noise
    original_points_expanded = data[:, np.newaxis, :]
    combined_clusters = np.concatenate([original_points_expanded, synthetic_points], axis=1)
    return combined_clusters.reshape(-1, dimensions)