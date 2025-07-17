import pickle
import numpy as np
import time


all_data_tr = []
labels_tr = []

for i in range(1,6):
    with open(f'data_batch_{i}', 'rb') as file_tr:
        batch_tr = pickle.load(file_tr, encoding='bytes')
        all_data_tr.append(batch_tr[b'data'])
        labels_tr.extend(batch_tr[b'labels'])

data_tr = np.concatenate(all_data_tr, axis=0)
data_tr = data_tr.astype(np.float64)




with open('test_batch', 'rb') as file_check:
    batch_check = pickle.load(file_check, encoding='bytes')

data_test = batch_check[b'data']
data_test = data_test.astype(np.float64)
labels_test = batch_check[b'labels']

start_time =  time.time()
mispredictions = 0

for test_sample in range(0, len(data_test)):
    min_distance = np.sqrt(np.sum((data_test[test_sample] - data_tr[0]) ** 2))
    label_prediction = labels_tr[0]
    for tr_sample in range(1,len(data_tr)):
        distance = np.sqrt(np.sum((data_test[test_sample] - data_tr[tr_sample]) ** 2))
        if min_distance > distance:
            min_distance = distance
            label_prediction = labels_tr[tr_sample]
    if label_prediction != labels_test[test_sample]:
        mispredictions +=1

print(f"Ο αριθμός των λάθος προβλέψεων του 1-ΝΝ είναι {mispredictions} στα {len(data_test)} δείγματα ελέγχου.")
print(f"Ποσοστό αποτυχίας 1-ΝΝ: {(mispredictions / len(data_test)) * 100}%")
end_time = time.time()
print(f"Συνολικός χρόνος εκτέλεσης 1-ΝΝ: {end_time-start_time} δευτερόλεπτα / {(end_time-start_time) / 3600} ώρες")
print("--------------")


start_time =  time.time()
mispredictions = 0

for test_sample in range(0, len(data_test)):
    distances = []
    for tr_sample in range(0,len(data_tr)):
        distance = np.sqrt(np.sum((data_test[test_sample] - data_tr[tr_sample]) ** 2))
        distances.append((distance, labels_tr[tr_sample]))

    distances.sort(key=lambda x:x[0])
    nearest_neighbors = [distances[0],distances[1],distances[2]]


    if nearest_neighbors[0][1] == nearest_neighbors[1][1]:
        label_prediction = nearest_neighbors[0][1]
    elif nearest_neighbors[0][1] == nearest_neighbors[2][1]:
        label_prediction = nearest_neighbors[0][1]
    elif nearest_neighbors[1][1] == nearest_neighbors[2][1]:
        label_prediction = nearest_neighbors[1][1]
    else:
        label_prediction = nearest_neighbors[0][1]

    if label_prediction != labels_test[test_sample]:
        mispredictions += 1

print(f"Ο αριθμός των λάθος προβλέψεων του 3-ΝΝ είναι {mispredictions} στα {len(data_test)} δείγματα ελέγχου.")
print(f"Ποσοστό αποτυχίας 3-ΝΝ: {(mispredictions / len(data_test)) * 100}%")
end_time = time.time()
print(f"Συνολικός χρόνος εκτέλεσης 3-ΝΝ: {end_time-start_time} δευτερόλεπτα / {(end_time-start_time) / 3600} ώρες")
print("--------------")


start_time =  time.time()
centroids = []
mispredictions = 0

for label in range(0,10):
    class_points = []
    for tr_sample in range(0,len(data_tr)):
        if label == labels_tr[tr_sample]:
            class_points.append(data_tr[tr_sample])
    centroids.append(np.mean(class_points, axis = 0))

for test_sample in range(0, len(data_test)):
    min_distance = np.sqrt(np.sum((data_test[test_sample] - centroids[0]) ** 2))
    'min_distance = euclidean(data_check[ch_sample], centroids[0])'
    label_prediction = 0
    for centroid_label in range(1,len(centroids)):
        distance = np.sqrt(np.sum((data_test[test_sample] - centroids[centroid_label]) ** 2))
        'distance = euclidean(data_check[ch_sample], centroids[centroid_label])'
        if min_distance > distance:
            min_distance = distance
            label_prediction = centroid_label
    if label_prediction != labels_test[test_sample]:
        mispredictions += 1

print(f"Ο αριθμός των λάθος προβλέψεων του Nearest Centroid είναι {mispredictions} στα {len(data_test)} δείγματα ελέγχου.")
print(f"Ποσοστό αποτυχίας Nearest Centroid: {(mispredictions / len(data_test)) * 100}%")
end_time = time.time()
print(f"Συνολικός χρόνος εκτέλεσης 3-ΝΝ: {end_time-start_time} δευτερόλεπτα / {(end_time-start_time) / 3600} ώρες")





