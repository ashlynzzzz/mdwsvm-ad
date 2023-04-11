import numpy as np
import matplotlib.pyplot as plt
from emnist import extract_training_samples
from metric import within_class_error

from vertices_generator import vertices
from mdwsvm_ad import mdwsvm_ad
from mdwsvm import mdwsvm
from cross_validation import cross_validation

err_ad = np.zeros((6,14,6)) # error value 6 sigma value, 14 c value, 6 v value

# Load data
digits_images, digits_labels = extract_training_samples('digits')
letters_images, letters_labels = extract_training_samples('byclass')

X_train = np.zeros((4000,28,28))
y_train = np.zeros((4000), dtype=int)
X_test = np.zeros((40000,28,28))
y_test = np.zeros((40000), dtype=int)

# 4000 digits normalized training data 
X_train[0:4000,:,:] = digits_images[0:4000,:,:] / 255
X_train = X_train.reshape(4000,784).T 
y_train[0:4000] = digits_labels[0:4000] # 4000 digits training label

# Get 2000 digits for test X
X_test[0:2000,:,:] = digits_images[4000:6000,:,:] / 255
y_test[0:2000] = digits_labels[4000:6000]
# Get 38000 lowercase letters
count = 2000
current_i = 0
while True:
    if count == 40000:
        break
    
    if(letters_labels[current_i] >= 36): # Get lower case letter
        X_test[count,:,:] = letters_images[current_i,:,:] / 255
        y_test[count] = letters_labels[current_i]
        count += 1
        
    current_i += 1
# 2000 digits and 38000 letters normalized data, 0-9 are 0-9, 36-61 are a-z
X_test = X_test.reshape(40000,784).T

# Use cross validation to choose C based on X_train
# Define values for cross_validation
c_values = [2**i for i in range(-3,10)]
v_values = [0.001, 0.01, 0.1, 0.3, 0.6, 0.9]
sig_values = [0.001, 0.01, 0.1, 1, 10, 100]
w = vertices(10)
size = 4000
num_folds = 5

# MDWSVM
err_simple, best_c_simple = cross_validation(c_values, 5, 4000, w, X_train, X_test, y_test, y_train, mdwsvm)

w = vertices(11)
# MDWSVM_ad
folder_size = int(size / num_folds)
# Loop over each value and perform cross-validation
best_c = 0
best_v = 0
best_sig = 0
best_score = -1
isig = 0
ic = 0
iv = 0

for sig in sig_values:
    k = lambda x, y: np.exp(-np.linalg.norm(x - y)**2)/(2 * sig**2)
    ic = 0
    for c in c_values:
        iv = 0
        for v in v_values:
            scores = np.zeros(5)
            # Perform cross-validation and calculate the average score
            for i in range(num_folds):
                # Get testing set    
                testx = X_train[:, i*folder_size:(i+1)*folder_size]
                testy = y_train[i*folder_size:(i+1)*folder_size]
                # Get training set    
                trainx = np.hstack((X_train[:, 0:(i)*folder_size], X_train[:, (i+1)*folder_size:size]))
                trainy = np.hstack((y_train[0:(i)*folder_size], y_train[(i+1)*folder_size:size]))
                # Create method object
                method = mdwsvm_ad(trainx, trainy, w, c, v, k)
                
                pred_y = method.predict(testx)
                score = 1 - within_class_error(y_ture = testy, y_pred = pred_y)
                scores[i] = score
                
            # Check if the current value of c is the best so far
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_c = c
                best_v = v
                best_sig = sig
                best_score = avg_score
            
            # Record the current error and all the parameter values
            err_ad[isig, ic, iv] = 1 - avg_score
            iv += 1
        ic += 1
    isig += 1

            
# use the optimal C to train X_train and get the final classifier
k = lambda x, y: np.exp(-np.linalg.norm(x - y)**2)/(2 * best_sig**2)
method_666 = mdwsvm_ad(X_train, y_train, w, best_c, best_v, k)

# perform the final classifier on X_test
pred_y = method_666.predict(X_test)
err_ad_best = within_class_error(y_ture = y_test, y_pred = pred_y)   # store the value for the best error


# Plot
fig, axs = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(8, 3))

axs[0].set_ylim([0, 1])
axs[0].set_ylabel('Error')
axs[0].set_title('c')
for i in range(6):
    for j in range(6):
        axs[0].plot(c_values, err_ad[i, :, j], color = 'black', linestyle = '-', label = 'c')

axs[1].set_title('v')
for i in range(14):
    for j in range(6):
        axs[1].plot(v_values, err_ad[j, i, :], color = 'black', linestyle = '-', label = 'v')

axs[2].set_title('sig')
for i in range(14):
    for j in range(6):
        axs[2].plot(sig_values, err_ad[:, i, j], color = 'black', linestyle = '-', label = 'sig')

plt.tight_layout()
plt.show()





            
#TODO: Compare model
# use cross validation to choose hyperparameter
# mdwsvm
# mdwsvm_ad
# hybrid: use the same hyperparameter chosen in mdwsvm

#TODO:
# show result