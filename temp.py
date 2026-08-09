import pickle
from sklearn.dummy import DummyClassifier
clf = DummyClassifier(strategy='most_frequent').fit([[0],[1],[0]], [0,1,0])
with open('blackboxes/compasA.sav','wb') as f: pickle.dump(clf,f)
with open('blackboxes/compasB.sav','wb') as f: pickle.dump(clf,f)
print('Dummy models written')
