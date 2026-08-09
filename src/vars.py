import numpy as np
from sklearn.tree import _tree

plot_colors = ["#2f4b7c","#ce2029","#ffa600", '#85B832', "#84cac4", "#42858C","#DFE667", "#52854C", "#ff7c43","#f95d6a","#d45087", "#7B1E7A", '#606c38', '#283618', '#8d99ae', '#dda15e', '#bc6c25']

color_dict = {'compas': dict(zip(['0|0', '1|1', '2|2',
                                  '0|1', '1|0', '0|2',
                                  '2|0', '1|2', '2|1'],
                                 zip(["#52854C", '#85B832', '#B9E868',
                                      '#9A1D25', '#ce2029', '#ffa600',
                                      "#E86202", '#42858C', '#84cac4'],
                                     [['o', 'white'], ['v', 'black'], ['^', 'black'],
                                      ['s', 'white'], ['P', 'black'], ['D', 'black'],
                                      ['_', 'black'], ['|', 'black'], ['1', 'black']]))),
              'bankmarketing': dict(
                  zip(['0|0','1|1', '0|1', '1|0'],
                      zip(['#52854C','#85B832', '#176675', '#54C2CC'],
                          [['o', 'white'], ['x', 'black'], ['v', 'white'],['s', 'black']]))),
              'running1': dict(
                  zip(#['0|0','1|1', '0|1', '1|0'],
                      ['0','1', '2', '3'],
                      zip(['#52854C','#85B832', '#176675', '#54C2CC'],
                          [['o', 'white'], ['x', 'black'], ['v', 'white'],['s', 'black']]))),
              'running2': dict(zip(['0|0', '1|1', '2|2',
                                    '0|1', '1|0', '0|2',
                                    '2|0', '1|2', '2|1'],
                                   zip(["#52854C", '#85B832', '#B9E868',
                                        '#9A1D25', '#ce2029', '#ffa600',
                                        "#E86202", '#42858C', '#84cac4'],
                                       [['.', 'white'], ['v', 'black'], ['^', 'black'],
                                        ['s', 'black'], ['+', 'black'], ['D', 'black'],
                                        ['_', 'black'], ['|', 'black'], ['X', 'black']])))
              }

classes_dict = {'compas':dict(zip(['0|0', '1|0', '1|1',
                                   '0|2', '2|2', '2|0',
                                   '2|1', '0|1', '1|2'],
                                  ['Low|Low', 'Medium|Low', 'Medium|Medium',
                                   'Low|High', 'High|High', 'High|Low',
                                   'High|Medium', 'Low|Medium', 'Medium|High'])),
                'bankmarketing':dict(zip(['0|0', '0|1', '1|0', '1|1'],
                                         ['No|No', 'No|Yes', 'Yes|No', 'Yes|Yes']))
                }

def make_meshgrid(x, y, h=.05):
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def make_meshgridLinspace(x, y, h=.005):
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    stepsX = round(abs((x_min - x_max) / h))
    stepsY = round(abs((y_min - y_max) / h))
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, num=stepsX), np.linspace(y_min, y_max, num=stepsY))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contour(xx, yy, Z, **params)
    return out


def get_rules(tree, feature_names, class_names):
    #original source: https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"(${name} \leq {np.round(threshold, 3)}$)"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"(${name} > {np.round(threshold, 3)}$)"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    rulecounter = 1

    rules = []
    for path in paths:
        rule = "if "

        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        classes = path[-1][0][0]
        l = np.argmax(classes)
        rule += f"class: {class_names[l]}"
        #if class_names[l] not in ['0|0', '1|1', '2|2']:
        rule = f"Rule {rulecounter}: " + rule
        rulecounter += 1
        rules += [rule]

    return rules