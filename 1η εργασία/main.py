import pickle
import numpy as np
from outer_layer import OuterLayer
from layer import Layer
from loss import cross_entropy_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time

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


def shuffle_data(X, d):
    indices = np.random.permutation(len(X))  # Τυχαία διάταξη δεικτών
    return X[indices], np.array(d)[indices]


# Δημιουργία batches
def create_batches(X, d, batch_size, n_batches):
    X, d = shuffle_data(X, d)  # Shuffling
    batches = []

    # Δημιουργούμε μέχρι τον απαιτούμενο αριθμό batches
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        X_batch = X[start_idx:end_idx]
        labels_batch = d[start_idx:end_idx]

        # One-hot encoding για τα labels του batch
        d_batch = np.zeros((len(labels_batch), 10))
        for j, label in enumerate(labels_batch):
            d_batch[j][label] = 1

        batches.append((X_batch, d_batch))

    return batches


batch_size = 256
n_batches = len(data_tr) // batch_size

batches = create_batches(data_tr, labels_tr, batch_size, n_batches)


layer1 = Layer(512,3072)
layer2=Layer(256,512)
layer3=Layer(128,256)
layer_outer = OuterLayer(10, 128)
epochs = 300
b = 0.003
loss_previous_epoch = 100
epoch_losses = []
accuracy_test = []
accuracy_training = []

for epoch in range(1,epochs+1):
    loss_of_epoch = 0
    training_predictions = 0
    for i, (X_batch, d_batch) in enumerate(batches):
        #φάση εμπρόσθιας τροφοδότησης
        layer1.forward(X_batch)
        layer1.activation()
        layer2.forward(layer1.activated_outputs)
        layer2.activation()
        layer3.forward(layer2.activated_outputs)
        layer3.activation()
        layer_outer.forward(layer3.activated_outputs)
        layer_outer.activation()

        for k in range(len(X_batch)):
            prediction = np.argmax(layer_outer.activated_outputs[k])
            if prediction == np.argmax(d_batch[k]):
                training_predictions += 1

        loss_of_epoch = loss_of_epoch + cross_entropy_loss(layer_outer.activated_outputs, d_batch)
        #φάση οπισθοδρόμησης
        layer_outer.backward(b, layer3.activated_outputs, d_batch)
        layer3.backward(b, layer2.activated_outputs,layer_outer)
        layer2.backward(b, layer1.activated_outputs,layer3)
        layer1.backward(b, X_batch, layer2)
    print("--------")
    print(epoch)
    print(b)
    print(loss_of_epoch / n_batches)
    'print("---------")'
    epoch_losses.append(loss_of_epoch / n_batches)

    training_accuracy = (training_predictions / (n_batches * batch_size)) * 100
    accuracy_training.append(training_accuracy)
    print(training_accuracy)

    Y = []
    d = np.zeros((10000, 10))
    for i in range(0, 10000):
        Y.append(data_test[i])
        d[i][labels_test[i]] = 1

    Y = np.array(Y)
    d = np.array(d).reshape(10000, 10)

    layer1.forward(Y)
    layer1.activation()
    layer2.forward(layer1.activated_outputs)
    layer2.activation()
    layer3.forward(layer2.activated_outputs)
    layer3.activation()
    layer_outer.forward(layer3.activated_outputs)
    layer_outer.activation()
    correct_predictions = 0
    for i in range(0, 10000):
        prediction = np.argmax(layer_outer.activated_outputs[i])
        if prediction == labels_test[i]:
            correct_predictions = correct_predictions + 1
    accuracy_test.append(correct_predictions  * 100 / 10000)
    print(correct_predictions  * 100 / 10000)
    print("---------")
    #Μεταβολή learning rate
    #'''
    if loss_of_epoch / n_batches > loss_previous_epoch :
        b = b * 0.5
    #'''

    '''
    if epoch % 30 == 0:
        b = b * 0.8
    '''
    loss_previous_epoch = loss_of_epoch / n_batches


matplotlib.pyplot.plot(range(1, epochs + 1), epoch_losses)
plt.title("Loss ανά εποχή")
plt.xlabel("Εποχή")
plt.grid()
file_name = "loss_plot.png"
plt.savefig(file_name)
os.startfile(file_name)

plt.clf()
matplotlib.pyplot.plot(range(1, epochs + 1), accuracy_test)
plt.title("Accuracy ανά εποχή")
plt.xlabel("Εποχή")
plt.grid()
file_name = "accuracy_plot.png"
plt.savefig(file_name)
os.startfile(file_name)

matplotlib.pyplot.plot(range(1, epochs + 1), accuracy_training)
plt.title("Accuracy ανά εποχή")
plt.xlabel("Εποχή")
plt.grid()
file_name = "accuracy_tr_plot.png"
plt.savefig(file_name)
os.startfile(file_name)

for i in range(len(accuracy_test)):
    print(accuracy_test[i])

end_time = time.time()
print((end_time - start_time) / 60)