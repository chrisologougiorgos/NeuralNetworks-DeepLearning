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

data_tr_mean = np.mean(data_tr, axis=0)
data_tr, eigenvectors, eigenvalues, selected_eigenvectors, variance_explained = pca(data_tr, 0.9)
data_test_centered = data_test - data_tr_mean
data_test = np.dot(data_test_centered, selected_eigenvectors)


# Ορισμός πλήθους test samples
n = len(data_test)

#Ορισμός πλήθους training samples
batch_size = 500
batches_number = 1
m = 1000
#Αρχικοποιήσεις
predictions_of_each_batch = np.zeros((batches_number, n))
prediction_for_each_label = np.zeros((10, n))
prediction_for_each_batch_tr = np.zeros((batches_number, m))
prediction_for_each_label_tr = np.zeros((10, m))
all_training_labels = []


validation_samples = data_tr[:m, :]
validation_labels = np.zeros(m)

for label in range(0, 10):
    print("ΚΛΑΣΗ", label)

    positive_samples = []
    negative_samples = []
    negative_indices = []
    for i in range(0, len(data_tr)):
        if labels_tr[i] == label:
            positive_samples.append(data_tr[i])
        else:
            negative_samples.append(data_tr[i])
            negative_indices.append(i)
    positive_samples = np.array(positive_samples)
    negative_samples = np.array(negative_samples)
    negative_indices = np.array(negative_indices)

    for i in range(0, m):
        if labels_tr[i] == label:
            validation_labels[i] = 1
        else:
            validation_labels[i] = -1
    for batch in range(0, batches_number):
        print("Batch:", batch + 1)
        # Επιλογή τυχαίων δειγμάτων
        pos_indices = np.random.choice(positive_samples.shape[0], batch_size // 2, replace=(label >= 10))  # Τυχαίοι δείκτες
        neg_indices = np.random.choice(negative_samples.shape[0], batch_size // 2, replace=(label >= 10))  # Τυχαίοι δείκτες
        # Επιλογή δειγμάτων βάσει δεικτών
        pos_batch = positive_samples[pos_indices]
        neg_batch = negative_samples[neg_indices]
        X_tr = np.vstack((pos_batch, neg_batch))  # Συνένωση δεδομένων
        Y_tr = np.hstack((np.ones(len(pos_batch)), -np.ones(len(neg_batch))))  # Ετικέτες (+1 για θετικά, -1 για αρνητικά)

        # Ανακάτεμα (shuffle)
        indices = np.arange(X_tr.shape[0])  # Δημιουργία δεικτών
        np.random.shuffle(indices)  # Ανακάτεμα δεικτών
        X_tr = X_tr[indices]  # Αναδιάταξη δεδομένων
        Y_tr = Y_tr[indices]  # Αναδιάταξη ετικετών

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
        svm = SupportVectorMachine(1, 0.1)
        svm.train(X_tr, Y_tr)
        predictions_of_each_batch[batch, :] = svm.predict(X_test)
        prediction_for_each_batch_tr[batch, :] = svm.predict(validation_samples)
    for i in range(0, n):
        prediction_for_each_label[label, i] = np.mean(predictions_of_each_batch[:, i])
    print("Προβλέψεις για την κλάση: \n",prediction_for_each_label[label])
    for i in range(0, m):
        prediction_for_each_label_tr[label, i] = np.mean(prediction_for_each_batch_tr[:, i])


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

accuracy = 0
for i in range(0, n):
    if test_predictions[i] == labels_test[i]:
        accuracy = accuracy + 1
print("Ποσοστό ακρίβειας % στο test: ", accuracy * 100 / n)

print("------------")

training_predictions = np.argmax(prediction_for_each_label_tr, axis=0)

accuracy = 0
for i in range(0, m):
    if training_predictions[i] == labels_tr[i]:
        accuracy = accuracy + 1
print("Ποσοστό ακρίβειας % στο training: ", accuracy * 100 / m)


end_time = time.time()
print((end_time - start_time) / 60)
