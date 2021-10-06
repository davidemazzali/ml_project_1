# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

# compute batched stochastic gradient for MSE loss
def compute_sg_mse_batched(y_b, tx_b, w):
    return -1*np.transpose(tx_b).dot(y_b-tx_b.dot(w))

# compute unif rand stochastic gradient for MSE loss
def compute_sg_mse_unifrand((y, tx, w):
    n = np.random.randint(len(y))
    return tx[n]*(-1)*(y[n]-tx[n].dot(w))

# batched SGD for MSE
def stochastic_gradient_descent_mse_batched(y, tx, initial_w, batch_size, max_iters, gamma, print_steps = False, print_last_step = False):
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for y_b, tx_b in batch_iter(y, tx, batch_size, max_iters):
        loss = compute_loss_mse(y, tx, w)
        w = w - gamma*compute_sg_mse_batched(y_b,tx_b,w)
        
        ws.append(w)
        losses.append(loss)
        
        if print_steps:
            print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(bi=0, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    if print_last_step:
        print(("Stochastic Gradient Descent: loss={l}, w0={w0}, w1={w1}".format(l=losses[-1], w0=w[0], w1=w[1])))
    
    return losses, ws

def stochastic_gradient_descent_mse_unifrand(y, tx, initial_w, max_iters, gamma, print_steps = False, print_last_step = False):
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for num_iter in range(max_iters):
        loss = compute_loss_mse(y, tx, w)
        
        w = w - gamma*compute_sg_mse_unifrand(y,tx,w)
        
        ws.append(w)
        losses.append(loss)
        
        if print_steps:
            print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(bi=0, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    if print_last_step:
        print(("Stochastic Gradient Descent: loss={l}, w0={w0}, w1={w1}".format(l=losses[-1], w0=w[0], w1=w[1])))
    
    return losses, ws