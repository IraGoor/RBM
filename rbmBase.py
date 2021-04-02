# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 21:14:01 2020

@author: eden_
"""

# importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from Dataset import SetGenerator
from torch.utils import data
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt



"""Find two attributes with the higest Variance """
def findTwoAttrWithHigestVariance(data):
    variances =[]
    for i in range(data.shape[1]):
        variances.append(np.var(data[:,i]))
    
    index_max=np.argmax(variances)
    variances[index_max]=-1
    index_submax=np.argmax(variances)
    
    return index_max,index_submax
    
    """"Print scatter train plot"""
def printScatterTrainPlot(X,Y):
    plt.plot()
    ax =plt.subplot()
    plt.title('training set')
  
    
    ### Calculate x_num and y_num by largest variance later
    x_num,y_num = findTwoAttrWithHigestVariance(X)
    
    ### From here, do not touch - it works
    x_min= min(np.min(X[:,x_num])*1.25 , 0.1)

    x_max=max(np.max(X[:,x_num])*1.25 , 0.2)
    y_min=min(np.min(X[:,y_num])*1.25  , 0.1)
    y_max=max(np.max(X[:,y_num])*1.25  , 0.2)
    plt.axis([x_min,x_max,y_min,y_max])
    
    plt.xlabel('hidden value {} '.format(x_num))
    plt.ylabel('hidden value {} '.format(y_num))
    scatter=plt.scatter(X[:,x_num] ,X[:,y_num] ,s=10, c=Y)

    plt.show()

def setGraphTitles(graphTitle,X_Axis_Title,Y_Axis_Title):
    names=[]
    names.insert(0, graphTitle)
    names.insert(1, X_Axis_Title)
    names.insert(2, Y_Axis_Title)
    return names
    """"adds test results to training scatter plot"""
def addToScatterTestPoints(Test_set,SVM_pred,real_names):
    colorss =np.zeros(len(real_names))
    test_colors=[]
    for i in range(len(real_names)):
        if real_names[i]==0:
            test_colors.insert(i, 'cyan')
        elif real_names[i]==1:
             test_colors.insert(i, 'green')
    plt.scatter(Test_set[:,2],Test_set[:,3], c=test_colors)
    blue_x = mlines.Line2D([], [], color='blue', marker='x',
                          markersize=15, label='0')
    red_circle=mlines.Line2D([], [], color='red', marker='o',markerfacecolor='none',
                          markersize=15, label='1')
    plt.legend(handles=[blue_x,red_circle])

    
    for i, point in enumerate(SVM_pred):
        if SVM_pred[i]==0:
            plt.scatter(Test_set[i][2] ,Test_set[i][3] , marker="x",s=80,c='blue')
        if SVM_pred[i]==1:
            plt.scatter(Test_set[i][2] ,Test_set[i][3] , marker="o",s=120,edgecolors='red',facecolors = 'none')
    plt.show()



torch.set_default_dtype(torch.double)

class RBM():

   def __init__(self, 
        visible_size=100, hidden_size=120, weights_init=None, hidden_bias_init=None, visible_bias_init=None, 
        learning_rate=1e-4, momentum=0.5, n_epoch=30, batch_size=100, visible_std_init=10.0,
        n_gibbs_sampling=1, use_cuda=False):

        self.visible_size = visible_size
        self.hidden_size = hidden_size
        self.weights_init = weights_init
        self.hidden_bias_init = hidden_bias_init
        self.visible_bias_init = visible_bias_init
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.n_gibbs_sampling = n_gibbs_sampling
        self.visible_std = torch.full((1, visible_size), visible_std_init)

        self.weights = torch.empty(visible_size, hidden_size)
        if weights_init is None:
            self.weights.normal_(mean=0, std=0.01)
        else:
            self.weights = torch.from_numpy(weights_init)

        self.hidden_bias = torch.empty(hidden_size)
        if hidden_bias_init is None:
            #set bi
            #sets hidden biases at exponental distrubutions define lambda as 2
            self.hidden_bias.exponential_(lambd=2)
        else:
            self.hidden_bias = torch.from_numpy(hidden_bias_init)
        
        self.visible_bias = torch.empty(visible_size)
        if visible_bias_init is None:
            #sets visible biases to zeros
            self.visible_bias.zero_()
        else:
            self.visible_bias = torch.from_numpy(visible_bias_init)

        #momentum its how a change during the training will affect this value 
        self.visible_bias_momentum = self.visible_bias.clone()
        self.hidden_bias_momentum = self.hidden_bias.clone()
        self.weights_momentum = self.weights.clone()
        self.visible_std_momentum = self.visible_std.clone()
# =============================================================================
#   function description:
#   step create
#
#
#
#
#   Params:
#   x: array of visable node
# =============================================================================

   def sample_hidden(self, visible_layer):

        wx = torch.mm(visible_layer, self.weights)
        
        activation = wx + self.hidden_bias.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

   def sample_visible(self, hidden_layer):
# =============================================================================
#        
#         Sample visible layer based on hidden layer
#         hidden_layer: (hidden_size) sized tensor
#             the hidden layer used to sample the visible layer
#         
#         Returns
#             tensor containing sampled visible layer
#         
# =============================================================================
       visible_normal_mean = torch.mul(torch.mm(hidden_layer, self.weights.transpose(0, 1)), torch.pow(self.visible_std, 2)) + self.visible_bias
       visible_layer = torch.normal(mean=visible_normal_mean, std=self.visible_std.expand(self.batch_size, -1))
       return visible_layer


        
   

   def contrastive_divergence(self, batch):
# =============================================================================
#             """
#         Perform one iteration of contrastive divergence on bias and weights using gibbs sampling with mini-batches
#         batch: (batch_size, visible_size) sized tensor
#             the batch used in gradient descent
#         """
# 
#         ''' 1. set visable layer with batch input, 3-dim: [batch_size][number of visible nodes][data] 
#             2. get array of activeted hidden layer before Forward Pass
#         '''
# =============================================================================
        visible_layer = batch.clone()

        
        _,hidden_layer = self.sample_hidden(visible_layer)

        ''' calculate batch of input with original values hidden layers multiplication, hidden association, and std
            (before Forward Pass )
                  '''
        positive_association = torch.bmm(visible_layer.unsqueeze(2), hidden_layer.unsqueeze(1))
        positive_hidden_association = hidden_layer
        positive_std_association = torch.pow(visible_layer - self.visible_bias, 2) 

        ''' start Forward Pass '''
        for i in range(self.n_gibbs_sampling):
            visible_layer = self.sample_visible(hidden_layer)
            _,hidden_layer = self.sample_hidden(visible_layer)
# =============================================================================
# 
#                 ''' calculate batch of input with learned values hidden layers multiplication
#             (after Forward Pass )
#                   '''
# =============================================================================
        negative_association = torch.bmm(visible_layer.unsqueeze(2), hidden_layer.unsqueeze(1))

        ''' calculate gradient decent after sampeling on weights and biases '''
        self.weights_momentum = self.weights_momentum * self.momentum + torch.sum(positive_association - negative_association, dim=0)
        self.visible_bias_momentum = self.visible_bias_momentum * self.momentum + torch.sum(batch - visible_layer, dim=0)
        self.hidden_bias_momentum = self.hidden_bias_momentum * self.momentum + torch.sum(positive_hidden_association - hidden_layer, dim=0)

        ''' update rbm's  weights and biases'''
        self.weights += self.weights_momentum * self.learning_rate / self.batch_size
        self.visible_bias += self.visible_bias_momentum * self.learning_rate / self.batch_size
        self.hidden_bias += self.hidden_bias_momentum * self.learning_rate / self.batch_size

        ''' update visable layer std momentum and  '''
        negative_std_association = torch.pow(visible_layer - self.visible_bias, 2)  
        self.visible_std_momentum = self.visible_std_momentum * self.momentum + torch.div(torch.sum(negative_std_association - positive_std_association, dim=0), torch.pow(self.visible_std, 3))
        self.visible_std += self.visible_std_momentum * self.learning_rate / self.batch_size
        error = torch.sum((batch - visible_layer) ** 2) / (self.batch_size * self.visible_size)
        print("             Error: " + str(error.item()), end='')


   def train(self,setGenerator,validation_size):       
# =============================================================================
#         """
#         Train RBM with input data
#         input_data: numpy array
#             numpy array containing the training data
#         """
#         
# =============================================================================
        datasetG=setGenerator
        generator=setGenerator
        training_set = generator.get_training_set()
        testing_set = generator.get_test_set()
        num_batch_in_input = len(training_set)//self.batch_size
        
        #create an array holds all of the values of the free energy during the contrastive divergence
        free_energy_gaps = torch.empty(self.n_epoch * num_batch_in_input) 
        training_generator= data.DataLoader(training_set, sampler=data.RandomSampler(training_set), batch_size=self.batch_size, drop_last=True, pin_memory=True)
        training_representative_generator = data.DataLoader(training_set, sampler=data.RandomSampler(training_set), batch_size=self.batch_size, drop_last=True, pin_memory=True)
        validation_generator = data.DataLoader(testing_set, sampler=data.RandomSampler(testing_set), batch_size=self.batch_size, drop_last=True, pin_memory=True)

        # save a random  sample from the training set to use to stop over fitting
        training_set_representative = next(iter(training_representative_generator))
        batch_num = 1
        for i in range(self.n_epoch):
            for batch,_ in training_generator:
                print("Batch #" + str(batch_num), end='')

                self.contrastive_divergence(batch) 
                #compare the free energy of the pre sampled batch from the training with the free energy of a batch from the test set  
                #to prevent overfitting if the free energy is grows very positive it's a sign for overfitting
                feg = self.free_energy(training_set_representative) - self.free_energy(next(iter(validation_generator)))
                free_energy_gaps[batch_num - 1] = feg
                print("             FEG: {:.7f}".format(feg.item()))
                batch_num += 1

        self.datasetG=generator
        self.training_set =generator.get_training_set()
        self.testing_set = generator.get_test_set()
        
   def free_energy(self, visible_layers):
        """
        Calculate average free energy of given visible layers
        visible_layer: (validation_set_size, visible_size) sized tensor
            visible layer used to calculate free energy
        Returns
            average free energy of the given input
        """
        
        return (-1 * torch.sum(torch.mv(visible_layers[0], self.visible_bias)) + torch.sum(torch.log1p(torch.exp(torch.matmul(visible_layers[0], self.weights) + self.hidden_bias))))/self.batch_size

