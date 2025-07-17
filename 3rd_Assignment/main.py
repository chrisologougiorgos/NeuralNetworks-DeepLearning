import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import time
from RbfLayer import RbfLayer
from OuterLayer import OuterLayer
from Loss import cross_entropy_loss
from pca import pca
from Layer import Layer

start_time =  time.time()
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

#PCA
data_tr_mean = np.mean(data_tr, axis=0)
data_tr, eigenvectors, eigenvalues, selected_eigenvectors, variance_explained = pca(data_tr, 0.9)
data_test_centered = data_test - data_tr_mean
data_test = np.dot(data_test_centered, selected_eigenvectors)

d_train = np.zeros((len(data_tr), 10))
for i in range(len(data_tr)):
    d_train[i][labels_tr[i]] = 1

d_test = np.zeros((len(data_test), 10))
for i in range(len(data_test)):
    d_test[i][labels_test[i]] = 1

n_inputs = len(data_tr)
inputs = np.zeros((n_inputs, data_tr.shape[1]))
for i in range(0, n_inputs):
    inputs[i] = data_tr[i]
d_inputs = d_train[:n_inputs, :]
epochs = 3000
b = 0.02
hidden_layer = RbfLayer( n_inputs, data_tr.shape[1], 500)
#layer1 = Layer(512, 1000)
#layer2 = Layer(256, 512)
outer_layer = OuterLayer(10, 500)

rbf_start_time = time.time()
hidden_layer.train(inputs)
rbf_end_time =  time.time()
print("Χρόνος εκπαίδευσης rbf layer:", (rbf_end_time - rbf_start_time) / 60)

rbf_outputs_tr = hidden_layer.forward(inputs)
rbf_outputs_test = hidden_layer.forward(data_test)

# training με least squares
#'''
outer_layer.train_least_squares(rbf_outputs_tr, d_inputs)

outer_layer.forward(rbf_outputs_tr)
outer_layer.activation()

training_predictions = 0
for sample in range(0, len(inputs)):
    prediction = np.argmax(outer_layer.activated_outputs[sample])
    if prediction == np.argmax(d_inputs[sample]):
        training_predictions += 1

print("Τελικα αποτελέσματα:")
print(outer_layer.activated_outputs)
print("Τιμή σ: ", hidden_layer.sigma)
training_accuracy = (training_predictions / len(inputs)) * 100
print("Ποσοστό επιτυχίας στο training:", training_accuracy)

outer_layer.forward(rbf_outputs_test)
outer_layer.activation()

test_predictions = 0
for sample in range(0, len(data_test)):
    prediction = np.argmax(outer_layer.activated_outputs[sample])
    if prediction == np.argmax(d_test[sample]):
        test_predictions += 1

test_accuracy = (test_predictions / len(data_test)) * 100
print("Ποσοστό επιτυχίας στο test:", test_accuracy)
#'''

# training με backpropagation
'''
for epoch in range(1, epochs + 1):
    loss_of_epoch = 0
    training_predictions = 0

    # training forward phase σε απλό δίκτυο
    outer_layer.forward(rbf_outputs_tr)
    outer_layer.activation()

    # training forward phase σε πολύπλοκο δίκτυο
    #layer1.forward(rbf_outputs_tr)
    #layer1.activation()
    #layer2.forward(layer1.activated_outputs)
    #layer2.activation()
    #outer_layer.forward(layer2.activated_outputs)
    #outer_layer.activation()


    for sample in range(0, len(inputs)):
        prediction = np.argmax(outer_layer.activated_outputs[sample])
        if prediction == np.argmax(d_inputs[sample]):
            training_predictions += 1

    loss_of_epoch =cross_entropy_loss(outer_layer.activated_outputs, d_inputs)


    #backpropagation σε απλό δίκτυο
    outer_layer.backward(b, rbf_outputs_tr, d_inputs)

    # Backpropagation σε πολύπλοκο δίκτυο
    # outer_layer.backward(b, layer2.activated_outputs, d_inputs)
    # layer2.backward(b, layer1.activated_outputs, outer_layer)
    # layer1.backward(b, rbf_outputs_tr, layer2)
    
    
    #test forward phase σε απλό δίκτυο
    outer_layer.forward(rbf_outputs_test)
    outer_layer.activation()
    
    #test forward phase σε πολύπλοκο δίκτυο
    #layer1.forward(rbf_outputs_test)
    #layer1.activation()
    #layer2.forward(layer1.activated_outputs)
    #layer2.activation()
    #outer_layer.forward(layer2.activated_outputs)
    #outer_layer.activation()


    test_predictions = 0
    for sample in range(0, len(data_test)):
        prediction = np.argmax(outer_layer.activated_outputs[sample])
        if prediction == np.argmax(d_test[sample]):
            test_predictions += 1


    print("Εποχή:",epoch)
    print("Training loss εποχής:", loss_of_epoch)
    print("Ποσοστό επιτυχίας στο training:",(training_predictions / len(inputs)) * 100)
    print("Ποσοστό επιτυχίας στο test:",(test_predictions / len(data_test)) * 100)
'''

#παραδείγματα κατηγοριοποιήσης
'''
for i in range(0, 100):
    print("Test δείγμα ", i+1)
    #print(d_test[i])
    print("Έξοδος του outer layer για το δείγμα:")
    print(outer_layer.activated_outputs[i])
    print("Πρόβλεψη του δικτύου για το δείγμα: Κλάση ", np.argmax(outer_layer.activated_outputs[i]))
    print("Πραγματική κλάση του δείγματος: Κλάση ", np.argmax(d_test[i]))
'''

end_time = time.time()
print("Τιμή σ: ", hidden_layer.sigma)
print("Χρόνος εκπαίδευσης rbf layer:", (rbf_end_time - rbf_start_time) / 60)
print("Συνολικός χρόνος εκτέλεσης: ", (end_time - start_time) / 60)

