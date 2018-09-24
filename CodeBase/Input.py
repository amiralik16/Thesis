import numpy as np
import pandas as pd

class Abide_Input:

    def __init__(self,loc_demographics=None, fol_features=None):
        '''
        loc_demographics: Demographic csv file
        fol_features: Folder where feature excels are
        '''
        self.loc_demographics = loc_demographics
        self.fol_features = fol_features
        self.fd = 0
        self.demo = pd.DataFrame()
        self.features = []

    
    def update_path(self,loc_demographics,fol_features):
        '''
        Redundant for now...maybe delete
        '''
        self.loc_demographics = loc_demographics
        self.fol_features = fol_features
    
    def read_files(self):
        self.demo = pd.read_csv('Variables.csv',index_col=0)

    def read_features(self):
        pass

    def preprocess_variables(self):
        pass
        
