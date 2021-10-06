# -*- coding: utf-8 -*-

"""Functions used to compute the loss."""

# computes MSE loss function
def compute_loss_mse(y, tx, w):
    e = y-tx.dot(w)
    return 1/(2*len(y)) * e.inner(e)

# computes MAE loss function
def compute_loss_mae(y, tx, w):
    e = y-tx.dot(w)
    return np.sum(np.abs(e))*1/(2*len(y))