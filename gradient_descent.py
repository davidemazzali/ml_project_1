# -*- coding: utf-8 -*-
"""Gradient Descent"""

# compute gradient of MSE loss
def compute_gradient_mse(y, tx, w):
    return -1/len(y) * np.transpose(tx).dot(y-tx.dot(w))

# gradient descent for MSE loss
def gradient_descent_mse(y, tx, initial_w, max_iters, gamma, print_steps = False, print_last_step = False):
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        loss = compute_loss_mse(y, tx, w)
        w = w - gamma*compute_gradient_mse(y,tx,w)
        
        ws.append(w)
        losses.append(loss)
        
        if print_steps:
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    
    if print_last_step:
        print(("Gradient Descent: loss={l}, w0={w0}, w1={w1}".format(l=losses[-1], w0=w[0], w1=w[1])))
    
    return losses, ws

# compute subgradient for MAE loss at w
def compute_subgradient_mae(y, tx, w):
    return -1/len(y) * np.transpose(tx).dot(np.sign(y-tx.dot(w)))

# subradient descent for MAE loss
def subgradient_descent_mae(y, tx, initial_w, max_iters, gamma, print_steps = False, print_last_step = False):
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        loss = compute_loss_mae(y, tx, w)
        w = w - gamma*compute_subgradient_mae(y,tx,w)
        
        ws.append(w)
        losses.append(loss)
        if print_steps:
            print("Subradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    
    if print_last_step:
        print(("Subgradient Descent: loss={l}, w0={w0}, w1={w1}".format(l=losses[-1], w0=w[0], w1=w[1])))
    return losses, ws