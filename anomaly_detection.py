import numpy as np
import matplotlib.pyplot as plt
from emnist import extract_training_samples
from sklearn.utils import shuffle
from functools import partial
import time

from vertices_generator import vertices
from kernel import Gaussian_kernel
from mdwsvm import mdwsvm
from mdwsvm_ad import mdwsvm_ad
from one_class_svm import one_class_svm
from hybrid import hybrid
from metric import within_class_error

# Load data
digits_images, digits_labels = extract_training_samples('digits')
letters_images, letters_labels = extract_training_samples('byclass')

# Get number 
mask_1 = (digits_labels == 1)
digits_images_1 = digits_images[mask_1]
digits_labels_1 = digits_labels[mask_1]

mask_3 = (digits_labels == 3)
digits_images_3 = digits_images[mask_3]
digits_labels_3 = digits_labels[mask_3]

mask_5 = (digits_labels == 5)
digits_images_5 = digits_images[mask_5]
digits_labels_5 = digits_labels[mask_5]

mask_7 = (digits_labels == 7)
digits_images_7 = digits_images[mask_7]
digits_labels_7 = digits_labels[mask_7]

# Get letter u, v, w, x, y, z
mask_uvwxyz = (letters_labels == 56) | (letters_labels == 57) | (letters_labels == 58) | (letters_labels == 59) | (letters_labels == 60) | (letters_labels == 61)
letters_images = letters_images[mask_uvwxyz]
letters_labels = letters_labels[mask_uvwxyz]

# Get training and testing data
X_train = np.zeros((800,28,28))
y_train = np.zeros((800), dtype=int)
X_val = np.zeros((8000,28,28))
y_val = np.zeros((8000), dtype=int)
X_test = np.zeros((8000,28,28))
y_test = np.zeros((8000), dtype=int)

# 800 digits normalized training data 
X_train[0:150,:,:] = digits_images_1[0:150,:,:] / 255
X_train[150:300,:,:] = digits_images_3[0:150,:,:] / 255
X_train[300:550,:,:] = digits_images_5[0:250,:,:] / 255
X_train[550:800,:,:] = digits_images_7[0:250,:,:] / 255
X_train = X_train.reshape(800,784).T 
# 800 digits training label
y_train[0:150] = digits_labels_1[0:150] - 1
y_train[150:300] = digits_labels_3[0:150] - 2
y_train[300:550] = digits_labels_5[0:250] - 3
y_train[550:800] = digits_labels_7[0:250] - 4

# Used for hybrid
# Get 400 digits for validation X
X_val[0:100,:,:] = digits_images_1[1000:1100,:,:] / 255
X_val[100:200,:,:] = digits_images_3[1000:1100,:,:] / 255
X_val[200:300,:,:] = digits_images_5[1000:1100,:,:] / 255
X_val[300:400,:,:] = digits_images_7[1000:1100,:,:] / 255
# 400 digits validation label
y_val[0:100] = digits_labels_1[1000:1100] - 1
y_val[100:200] = digits_labels_3[1000:1100] - 2
y_val[200:300] = digits_labels_5[1000:1100] - 3
y_val[300:400] = digits_labels_7[1000:1100] - 4
# Get 7600 lowercase letters
X_val[400:8000,:,:] = letters_images[0:7600,:,:] / 255
y_val[400:8000] = letters_labels[0:7600]
# Get true y label to calculate hybrid error
y_val_true_hybrid = -np.ones((8000), dtype=int)
y_val_true_hybrid[0:400] = y_val[0:400]
# Get true y label to calculate mdwsvm_ad error
y_val_true_mdwsvm_ad = 4 * np.ones((8000), dtype=int)
y_val_true_mdwsvm_ad[0:400] = y_val[0:400]
# 400 digits and 7600 letters normalized data
X_val = X_val.reshape(8000,784).T

# Get 400 digits for test X
X_test[0:100,:,:] = digits_images_1[1100:1200,:,:] / 255
X_test[100:200,:,:] = digits_images_3[1100:1200,:,:] / 255
X_test[200:300,:,:] = digits_images_5[1100:1200,:,:] / 255
X_test[300:400,:,:] = digits_images_7[1100:1200,:,:] / 255
# 400 digits test label
y_test[0:100] = digits_labels_1[1100:1200] - 1
y_test[100:200] = digits_labels_3[1100:1200] - 2
y_test[200:300] = digits_labels_5[1100:1200] - 3
y_test[300:400] = digits_labels_7[1100:1200] - 4
# Get 7600 lowercase letters
X_test[400:8000,:,:] = letters_images[8000:15600,:,:] / 255
y_test[400:8000] = letters_labels[8000:15600]
# Get true y label to calculate hybrid error
y_test_true_hybrid = -np.ones((8000), dtype=int)
y_test_true_hybrid[0:400] = y_test[0:400]
# Get true y label to calculate mdwsvm_ad error
y_test_true_mdwsvm_ad = 4 * np.ones((8000), dtype=int)
y_test_true_mdwsvm_ad[0:400] = y_test[0:400]
# 400 digits and 7600 letters normalized data
X_test = X_test.reshape(8000,784).T

# y_test: 0,1,2,3,56-61
# y_test_true_hybrid: -1,0,1,2,3
# y_test_true_mdwsvm_ad: 0,1,2,3,4

# Cross Validation for MDWSVM
c_values = [2**i for i in range(-3,13)]
w1 = vertices(4)
size = X_train.shape[1]
num_folds = 5

X_train_new, y_train_new = shuffle(X_train.T, y_train, random_state=42)
X_train_new = X_train_new.T
folder_size = int(size / num_folds)
# Loop over each value of c and perform cross-validation
best_c = 0
best_score = -1
for c in c_values:
    scores = np.zeros(num_folds)
    # Perform cross-validation and calculate the average score
    for i in range(num_folds):
        # Get testing set    
        testx = X_train_new[:, i*folder_size:(i+1)*folder_size]
        testy = y_train_new[i*folder_size:(i+1)*folder_size]
        # Get training set    
        trainx = np.hstack((X_train_new[:, 0:(i)*folder_size], X_train_new[:, (i+1)*folder_size:size]))
        trainy = np.hstack((y_train_new[0:(i)*folder_size], y_train_new[(i+1)*folder_size:size]))
        method = mdwsvm(trainx, trainy, w1, c)
        
        pred_y = method.predict(testx)
        score = 1 - within_class_error(testy, pred_y)
        scores[i] = score
    # Check if the current value of c is the best so far
    avg_score = np.mean(scores)
    if avg_score > best_score:
        best_c = c
        best_score = avg_score
        
print('MDWSVM best c is', best_c)
print('MDWSVM best score is', best_score)

# Use test data
model1 = mdwsvm(X_train, y_train, w1, best_c)
y_pred_1 = model1.predict(X_test)
print('MDWSVM error is', within_class_error(y_test, y_pred_1))


# Hybrid Method
v_values = [0.1, 0.3, 0.5, 0.7, 0.9]
sigma2_values = [0.001, 0.01, 0.05, 0.1, 0.3, 0.6, 1, 1.5]

best_c = 1
best_v = 0
best_sigma2 = 0
best_score = -1
time_record_1 = []
for v in v_values:
    for s in sigma2_values:
        print('-'*5+'Start: v=%.2f'%v+' sigma2=%.3f'%s+'-'*5)
        k = partial(Gaussian_kernel, sigma2=s)
        st = time.time()
        y_pred = hybrid(X_train, y_train, X_val, v, w1, best_c, k)
        et = time.time()
        time_record_1.append(et - st)
        score = 1 - within_class_error(y_val_true_hybrid, y_pred)
        print(score)
        if score > best_score:
            best_v = v
            best_sigma2 = s
            best_score = score
            
print('Hybrid best v is', best_v)
print('Hybrid best sigma2 is', best_sigma2)
print('Hybrid best score is', best_score)

# Use test data
best_k = partial(Gaussian_kernel, sigma2=best_sigma2)
y_pred_2 = hybrid(X_train, y_train, X_test, best_v, w1, best_c, best_k)
print('Hybrid error is', within_class_error(y_test_true_hybrid, y_pred_2))


# MDWSVM-AD Method
v_values = [0.1, 0.3, 0.5, 0.7, 0.9]
sigma2_values = [10, 12, 14, 16, 18, 55]
C_values = [1,5,8,9,10,11]
w2 = vertices(5)

best_v_2 = 0
best_sigma2_2 = 0
best_c_2 = 0
best_score_2 = -1
time_record_2 = []
for v in v_values:
    for s in sigma2_values:
        for c in C_values:
            print('-'*5+'Start: v=%.3f'%v+' sigma2=%.3f'%s+' c=%.3f'%c+'-'*5)
            k = partial(Gaussian_kernel, sigma2=s)
            st = time.time()
            method = mdwsvm_ad(X_train, y_train, w2, c, v, k)
            y_pred = method.predict(X_val)
            et = time.time()
            time_record_2.append(et - st)
            score = 1 - within_class_error(y_val_true_mdwsvm_ad, y_pred)
            print(score)
            if score > best_score_2:
                best_v_2 = v
                best_sigma2_2 = s
                best_c_2 = c
                best_score_2 = score
                
print('MDWSVM-AD best v is', best_v_2)
print('MDWSVM-AD best sigma2 is', best_sigma2_2)
print('MDWSVM-AD best C is', best_c_2)
print('MDWSVM-AD best score is', best_score_2)

# Use test data
best_k_2 = partial(Gaussian_kernel, sigma2=best_sigma2_2)
model3 = mdwsvm_ad(X_train, y_train, w2, best_c_2, best_v_2, best_k_2)
y_pred_3 = model3.predict(X_test)
print('MDWSVM-AD error is', within_class_error(y_test_true_mdwsvm_ad, y_pred_3))