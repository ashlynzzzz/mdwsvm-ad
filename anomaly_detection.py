import numpy as np
import matplotlib.pyplot as plt
from emnist import extract_training_samples

from vertices_generator import vertices
from mdwsvm_ad import mdwsvm_ad
from hybrid import hybrid

#TODO: Data preprocessing
# 1. load data
# 2. extract training data and test data
# 3. change lower-case classes as one label 

# Load data
digits_images, digits_labels = extract_training_samples('digits')
letters_images, letters_labels = extract_training_samples('letters')

X_train = np.zeros((4000,28,28))
y_train = np.zeros((4000))
X_test = np.zeros((40000,28,28))
y_test = np.zeros((40000))

X_train[0:4000,:,:] = digits_images[0:4000,:,:] # 4000 digits training data
y_train[0:4000] = digits_labels[0:4000] # 4000 digits training label

X_test[0:40000,:,:] = letters_images[0:40000,:,:] 
X_test[0:2000,:,:] = digits_images[4000:6000,:,:]  # change the first 2000 to digits data
y_test[0:40000] = letters_labels[0:40000] 
y_test[0:2000] = digits_labels[4000:6000]   # change the first 2000 to digits label


# Use cross validation to choose C based on X_train
# Define values for cross_validation
c_values = [2**i for i in range(-3,13)]
w = vertices(11)
            
            
            
#TODO: Compare model
# use cross validation to choose hyperparameter
# mdwsvm
# mdwsvm_ad
# hybrid: use the same hyperparameter chosen in mdwsvm

#TODO:
# show result