import numpy as np
import cvxpy as cp
import xmlrpc.client

from Kernel import Kernel


class SupportVectorMachine:
    def __init__(self, p, C):
        self.p = p
        self.C = C

    def train(self, X_tr, Y_tr):
        self.H = np.zeros((len(X_tr), len(X_tr)))
        self.X_tr = X_tr
        self.Y_tr = Y_tr
        self.ker = Kernel(self.p)

        for i in range(0, len(X_tr)):
            for j in range(0, len(X_tr)):
                self.H[i, j] = Y_tr[i] * Y_tr[j] * self.ker.calculate_kernel(X_tr[i], X_tr[j])
        self.H = self.H + np.eye(self.H.shape[0]) * 1e-5
        eigenvalues = np.linalg.eigvalsh(self.H)
        print("Minimum eigenvalue of H:", np.min(eigenvalues))
        self.H = cp.psd_wrap(self.H)

        c = np.ones(len(X_tr))
        self.a = cp.Variable(len(X_tr))
        objective = cp.Minimize(0.5 * cp.quad_form(self.a, self.H) - c.T @ self.a)
        constraints = [self.a >= 0, self.a <= self.C]
        problem = cp.Problem(objective, constraints)
        result = problem.solve()
        print("Solver status:", problem.status)
        self.a = self.a.value

        epsilon = 1e-6
        sv_indices = np.where((self.a > epsilon) & (self.a < self.C - epsilon))[0]
        print("Πλήθος support vectors:",len(sv_indices))
        if len(sv_indices) > 0:
            self.b = 0
            b_values = []  # Λίστα για αποθήκευση των τιμών b_{sv}
            for sv in sv_indices:
                b_sv = Y_tr[sv]  # Αρχικοποίηση με Y_tr[sv]
                for i in range(0, len(X_tr)):
                    b_sv -= self.a[i] * Y_tr[i] * self.ker.calculate_kernel(X_tr[sv], X_tr[i])
                b_values.append(b_sv)  # Αποθήκευση της τιμής b_{sv}
            self.b = np.mean(b_values)  # Υπολογισμός του μέσου όρου
            print(f"Bias (b) για την κλάση : {self.b}")
        else:
            print("error")


    def predict(self, X):
        W = np.zeros((len(X), len(self.X_tr)))
        for i in range(0, len(X)):
            for j in range(0, len(self.X_tr)):
                W[i, j] = self.Y_tr[j] * self.ker.calculate_kernel(X[i], self.X_tr[j])

        predictions = np.zeros(len(X))
        for i in range(0, len(X)):
             predictions[i] = np.dot(W[i,:], self.a) + self.b
        return predictions



