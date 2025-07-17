import numpy as np

class RbfLayer:
    def __init__(self, n_inputs,input_size, n_centroids):
        self.n_inputs = n_inputs
        self.input_size = input_size
        self.n_centroids = n_centroids
        self.c = np.zeros((self.n_centroids, self.input_size))
        self.centroid_labels = np.zeros(self.n_inputs)
        self.sigma = 1e-6



    def forward(self, inputs):
        self.outputs = np.zeros((len(inputs), self.n_centroids))
        for i in range(0, len(inputs)):
            distance = np.linalg.norm(inputs[i] - self.c, axis=1)
            self.outputs[i] = np.exp(-distance ** 2 / (2 * self.sigma ** 2))
        return self.outputs



    def train(self, inputs):
        # τυχαία επιλογή κεντρών
        random_indices = np.random.choice(inputs.shape[0], self.n_centroids, replace=False)
        self.c = inputs[random_indices, :]
        scaling_factor = 40

        # επιλογή κεντρών μέσω κ-μέσων
        '''
        repetition = 0
        while True:
            repetition += 1
            print("Επανάληψη: ", repetition)
            new_centroids = np.zeros((self.n_centroids, self.input_size))
            for sample in range(0, inputs.shape[0]):
                min_distance = np.sqrt(np.sum((inputs[sample] - self.c[0]) ** 2))
                self.centroid_labels[sample] = 0
                for centroid in range(1, self.n_centroids):
                    distance = np.sqrt(np.sum((inputs[sample] - self.c[centroid]) ** 2))
                    if min_distance > distance:
                        min_distance = distance
                        self.centroid_labels[sample] = centroid

            for centroid in range(0, self.n_centroids):
                addition_of_samples = np.zeros(self.input_size)
                cnt = 0
                for sample in range(0, inputs.shape[0]):
                    if self.centroid_labels[sample] == centroid:
                        addition_of_samples = addition_of_samples + inputs[sample]
                        cnt = cnt + 1
                mean_of_samples = addition_of_samples / cnt
                new_centroids[centroid] = mean_of_samples

            if np.array_equal(self.c, new_centroids):
                break

            self.c = new_centroids
        '''

        #Απλή ανάθεση τιμής
        #self.sigma = 10

        # Υπολογισμός του σ μέσω του μέσου όρου των αποστάσεων μεταξύ των κέντρων
        pairwise_distances = np.linalg.norm(self.c[:, None, :] - self.c[None, :, :], axis=2)
        mean_distance = np.mean(pairwise_distances[pairwise_distances > 0])  # Αποφεύγουμε τις μηδενικές αποστάσεις
        self.sigma = mean_distance / np.sqrt(2 * self.n_centroids) * scaling_factor

        print("Τιμη του σ:", self.sigma)




