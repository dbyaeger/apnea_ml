#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:53:53 2020

@author: danielyaeger
"""
import numpy as np

class LossMonitor():
    
    def __init__(self, patience: int = 3):
        self.patience = patience
    
    def stop_training(self,loss):
        """
          Inputs: Loss, list of scalars
              
          Returns False if:
            a) the length of the loss list is less than patience.
            b) the next consecutive patience entries do not have greater loss 
               than the ith entry.
           Returns True:
               if all of the next consecutive patience entries have greater loss 
               than the ith entry
        """
        if len(loss) <= (self.patience + 1):
            return False
        else:
            criterion = loss[-self.patience - 1]*np.ones(self.patience)
            test = np.array(loss[-self.patience:])
            if (test > criterion).all():
                return True
            return False