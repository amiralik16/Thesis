import numpy as np
import pandas as pd
from sklearn import feature_selection
from sklearn import svm
from sklearn import ensemble
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

class Experiment:

    def __init__(self,model=svm.SVC(), feature_selection_scheme=None, cv=None, train_set=None, test_set=None):
        '''
        model: sklearn model
        feature_selection_scheme :
        cv : cross validation object for train_set
        train_set = data used during training/feature_selection
        test_set = data used for final validation
        '''
        self.model = model
        self.feature_selection_scheme = feature_selection_scheme
        self.train_set = train_set
        self.test_set = test_set
        self.cv = cv




class Feature_Selection:

    def __init__(self, X=None, y=None, feature_selection_scheme=None, cv=None):
        #scheme = Tree, FeedForward, RFE
        #train set should be preprocessed
        self.feature_selection_scheme = feature_selection_scheme
        self.cv = cv
        self.fittedout = None
        self.X = X
        self.y = y

    def load_data(self,X,y,cv):
        self.X = X
        self.y = y
        self.cv = cv

    def _tree_selection(self,X,y):
        forest = ensemble.ExtraTreesClassifier()
        forest.fit(X, y)
        model = feature_selection.SelectFromModel(forest, prefit=True)
        return model.get_support()

    def _rfe_selection(self,X,y):
        svc = svm.LinearSVC()
        model = feature_selection.RFECV(estimator=svc,cv=self.cv,scoring="accuracy",n_jobs=-1,verbose=2)
        model.fit(X,y)
        return model.support_
    
    def _idx2mask(self,idxs, X):
        '''
        convert the index given by ff to the support mask format of the tree and rfe method
        '''
        length = X.shape[1]
        mask = [False]*length
        for idx in idxs:
            mask[idx] = True
        return mask


    def _ff_selection(self,model,X,y,cv):
        sfs = SFS(model, 
           k_features=(1,30), 
           forward=True, 
           floating=False, 
           verbose=2,
           scoring='accuracy',
           cv=self.cv,n_jobs=-1)
        sfs = sfs.fit(X, y)
        return self._idx2mask(sfs.k_feature_idx_,X)
         
    def selectfeatures(self,feature_selection_scheme=None,model=None):

        if feature_selection_scheme == None:
            scheme = self.feature_selection_scheme
        else:
            scheme = feature_selection_scheme

        if scheme == 'tree':
            return self._tree_selection(self.X,self.y)
        elif scheme == 'rfe':
            return self._rfe_selection(self.X,self.y)
        elif scheme == 'ff':
            return self._ff_selection(model,self.X,self.y,self.cv)

    def sth(self):
        pass
