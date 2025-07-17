import numpy as np

def cross_entropy_loss(outer_layer_output, d):
    predictions = np.clip(outer_layer_output, 1e-15, 1 - 1e-15)
    sample_losses = -np.sum(d * np.log(predictions), axis=1)
    loss_mean = np.mean(sample_losses)
    return loss_mean
