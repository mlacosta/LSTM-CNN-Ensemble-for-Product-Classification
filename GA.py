# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 15:47:35 2019

@author: Deep Marian
"""
import random
import numpy as np
from sklearn.metrics import balanced_accuracy_score

class individual():
    
    def __init__(self,weights):
        self.weights = weights
        self.fitness = 0
    
    def set_weights(self,weights):
        self.weights = weights

    def set_fitness(self,fitness):
        self.fitness = fitness
        


class genetic_algo():
    
    def __init__(self,models,population= 5,parents = 2):
        
        self.models = models
        self.population = population
        self.parents = 2
        
    def init_weights(self):
        
        return np.random.random(size=len(self.models))
        
    
    def init_population(self):
        
        self.individuals = []
        
        for inx in range(self.population):
            
            self.individuals.append(individual(self.init_weights))
    
    def calculate_fitness(self,x_data,y_data):
        
        for ind in self.individuals:
            
            y_predictions = []
            
            for model in self.models:
                
                y_predictions.append(model.predict(x_data,batch_size=128, verbose=1))
            
            y_weight = np.log(y_predictions[0]) * ind.weights[0]
            
            for inx in range(1,len(y_predictions)):
                
                y_weight += np.log(y_predictions[inx]) * ind.weights[inx]
            
            y_weight = y_weight/np.sum(ind.weights)
            
            y_weight = np.exp(y_weight)
            
            y_weight = np.argmax(y_weight, axis=1)
    
            ind.fitness = balanced_accuracy_score(y_data, y_weight)
    
    def show_fitness(self):
        
        for ind in self.individuals:
            
            print('fitness: %f\n'%(ind.fitness))
    
    def choose_parents
                
                
            
            
            
    
    
    
    
        
        