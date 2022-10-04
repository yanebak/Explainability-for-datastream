from numpy import sqrt, abs, round
from sklearn.utils import shuffle
from pysad.utils import ArrayStreamer
from sklearn.utils import shuffle
from pysad.evaluation import AUROCMetric
from pysad.models.integrations import ReferenceWindowModel
from pysad import models
from tqdm import tqdm
import numpy as np
import pandas as pd
from datetime import datetime

class RobustRandomCutForest :
    def __init__(self,num_trees,window_size,tree_size) :
      self.num_trees = num_trees
      self.window_size = window_size
      self.tree_size = tree_size
      self.iteration = 0
      self.model = None

    
    def fit(self, X):
        self.iteration += 1
        if self.model == None:
            self.model = models.RobustRandomCutForest(self.num_trees,self.window_size,self.tree_size)
            self.model.fit_partial(X)
        else :
            self.model.fit_partial(X)
        # print("Model fitted successfuly")
    

    def score(self, X):
        if self.model != None :
            return self.model.score_partial(X)
        else :
            return 0
