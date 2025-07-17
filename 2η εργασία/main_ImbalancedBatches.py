import pickle
import numpy as np
import os
import time
from Support_Vector_Machine import SupportVectorMachine
from PCA import pca

start_time =  time.time()
# Φόρτωση δεδομένων
all_data_tr = []
labels_tr = []

for i in range(1,6):
    with open(f'data_batch_{i}', 'rb') as file_tr:
        batch_tr = pickle.load(file_tr, encoding='bytes')
        all_data_tr.append(batch_tr[b'data'])
        labels_tr.extend(batch_tr[b'labels'])

data_tr = np.concatenate(all_data_tr, axis=0)
data_tr = data_tr.astype(np.float64)
data_tr = data_tr/255.0

with open('test_batch', 'rb') as file_check:
    batch_check = pickle.load(file_check, encoding='bytes')

data_test = batch_check[b'data']
data_test = data_test.astype(np.float64)
data_test = data_test/255.0
labels_test = batch_check[b'labels']

# Ορισμός πλήθους test samples
n = len(data_test)

#Ορισμός πλήθους training samples
batch_size = 500
batches_number = 500 // batch_size

#Αρχικοποιήσεις
predictions_of_each_batch = np.zeros((batches_number, n))
prediction_for_each_label = np.zeros((10, n))
prediction_for_each_batch_tr = np.zeros(batches_number * batch_size)
prediction_for_each_label_tr = np.zeros((10, batches_number * batch_size))

for label in range(0, 10):
    print("ΚΛΑΣΗ", label)
    for batch in range(0, batches_number):
        print("Batch:", batch + 1)
        X_tr = []
        Y_tr = np.zeros(batch_size)
        cnt = 0
        j = 0
        for i in range(batch * batch_size, (batch + 1) * batch_size):
            X_tr.append(data_tr[i])
            if labels_tr[i] == label:
                Y_tr[j] = 1
                cnt= cnt + 1
            else:
                Y_tr[j] = -1
            j = j + 1
        X_tr = np.array(X_tr)
        print("Πλήθος δειγμάτων +1 στο batch:",cnt)

        Y_test = np.zeros(n)
        for i in range(0, n):
            if labels_test[i] == label:
                Y_test[i] = 1
            else:
                Y_test[i] = -1
        X_test = []
        for i in range(0, n):
            X_test.append(data_test[i])
        X_test = np.array(X_test)

        X_tr_mean = np.mean(X_tr, axis=0)
        X_tr, eigenvectors, eigenvalues, selected_eigenvectors, variance_explained = pca(X_tr, 0.9)
        X_test_centered = X_test - X_tr_mean
        X_test = np.dot(X_test_centered, selected_eigenvectors)
        svm = SupportVectorMachine(2, 0.1)
        svm.train(X_tr, Y_tr)
        predictions_of_each_batch[batch, :] = svm.predict(X_test)
        prediction_for_each_batch_tr[(batch * batch_size):((batch + 1) * batch_size)] = svm.predict(X_tr)

    for i in range(0, n):
        prediction_for_each_label[label, i] = np.mean(predictions_of_each_batch[:, i])
    print("Προβλέψεις για την κλάση: \n",prediction_for_each_label[label])

    prediction_for_each_label_tr[label, :] = prediction_for_each_batch_tr


test_predictions = np.argmax(prediction_for_each_label, axis=0)
counter_of_positive_predictions = 0
j=0
for i in test_predictions:
    if prediction_for_each_label[i,j] > 0:
        counter_of_positive_predictions = counter_of_positive_predictions + 1
    j = j + 1
print("Ποσοστό θετικών προβλέψεων στο test %:",counter_of_positive_predictions * 100 / len(test_predictions))
print("Προβλέψεις για τις κλάσεις των test δειγμάτων:")
print(test_predictions)

labels_test_true = np.zeros(n)
for i in range(0, n):
    labels_test_true[i] = labels_test[i]
labels_test_true = labels_test_true.astype(int)
print("Πραγματικές κλάσεις των test δειγμάτων:")
print(labels_test_true)

labels_test_true = np.zeros(n)
for i in range(0, n):
    labels_test_true[i] = labels_test[i]
labels_test_true = labels_test_true.astype(int)

accuracy = 0
for i in range(0, n):
    if test_predictions[i] == labels_test[i]:
        accuracy = accuracy + 1
print("Ποσοστό ακρίβειας % στο test: ", accuracy * 100 / n)

print("------------")

training_predictions = np.argmax(prediction_for_each_label_tr, axis=0)

accuracy = 0
for i in range(0, batches_number * batch_size):
    if training_predictions[i] == labels_tr[i]:
        accuracy = accuracy + 1
print("Ποσοστό ακρίβειας % στο training: ", accuracy * 100 / (batches_number * batch_size))


end_time = time.time()
print((end_time - start_time) / 60)






