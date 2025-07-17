import numpy as np


def pca(X, var_explained=0.9):
    # Κεντροποίηση των δεδομένων (αφαίρεση του μέσου όρου)
    X_centered = X - np.mean(X, axis=0)

    # Υπολογισμός της συνδιακύμανσης
    covariance_matrix = np.cov(X_centered, rowvar=False)

    # Υπολογισμός των ιδιοτιμών και των ιδιοδιανυσμάτων
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Ταξινόμηση των ιδιοτιμών σε φθίνουσα σειρά
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Υπολογισμός του ποσοστού της συνολικής διασποράς
    total_variance = np.sum(eigenvalues)
    variance_explained = np.cumsum(eigenvalues) / total_variance

    # Εύρεση του αριθμού των συνιστωσών που εξηγούν τουλάχιστον `var_explained` της διασποράς
    num_components = np.where(variance_explained >= var_explained)[0][0] + 1

    # Επιλογή των πρώτων `num_components` ιδιοδιανυσμάτων
    selected_eigenvectors = eigenvectors[:, :num_components]

    # Εφαρμογή των κύριων συνιστωσών στα δεδομένα
    X_reduced = np.dot(X_centered, selected_eigenvectors)

    return X_reduced, eigenvectors, eigenvalues, selected_eigenvectors, variance_explained

