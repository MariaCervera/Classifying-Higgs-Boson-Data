from helpers import *
from costs import *


#We have all the optimization method in one file

#-------------------------------------------------------------------------------
#                               Gradient Descent
#-------------------------------------------------------------------------------
def least_squares_GD(y,tx,gamma,max_iters):
    """Linear regression using gradient descent
    gamma : step size
    max_iters : number of steps to run

    returns : final weights
    """
    # Initialisation
    initial_w = np.zeros(tx.shape[1])
    ws = [initial_w]
    losses = []
    w = initial_w
    threshold =  1e-8

    for n_iter in range(max_iters):
        gradient = compute_gradient_MSE(y,tx,w)
        loss = compute_loss(y,tx,w)
        w_new = w - gamma*gradient
        w = w_new
        # Store w and loss
        ws.append(w)
        losses.append(loss)
        # converge criteria
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return ws[-1]



def least_squares_GD_optimized(y,tx,gamma,beta,max_iters):
    """Linear regression using gradient descent where teh gamma is decreased iteratively
    gamma : step size
    max_iters : number of steps to run
    beta : ratio to decrease the gamma each step 

    returns : iterative MSE, iterative weights
    """
    # Initialisation

    #initialization
    initial_w = np.zeros(tx.shape[1])
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient = gradient_descent_Hugo.compute_gradient_MSE(y,tx,w)
        loss = costs.compute_loss(y,tx,w)
        w_new = w - gamma*gradient
        w = w_new
        gamma = gamma*beta
        # store w and loss
        ws.append(w)
        losses.append(loss)
        #print("Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))
    return losses, ws

def compute_gradient_MAE(y,tx,w):
    """Compute the gradient with MAE."""
    e = y - np.dot(tx,w.T)
    s = np.sign(e).reshape(-1,1)
    toReturn = np.sum(-tx*s,axis=0)/len(y) 
    return toReturn
            
def compute_gradient_MSE(y, tx, w):
    """Compute the gradient with MSE."""
    e = y - np.dot(tx,w.T)
    toReturn = -np.dot(tx.T,e) / y.shape[0]
    return toReturn

#-------------------------------------------------------------------------------
#                         Stochastic Gradient Descent
#-------------------------------------------------------------------------------
def least_squares_SGD(y, tx, gamma, max_iters,batch_size=1):
    """Linear regression using SGD
    gamma : step size
    max_iters : number of steps to run
    batch_size : by default 1

    returns : iterative losses and weights
    """
    #initialization
    initial_w = np.zeros(tx.shape[1])
    batches = batch_iter(y, tx, batch_size)
    
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        try:
            # get the next item
            y,tx = batches.__next__()
            
        except StopIteration:
            # No more elements in Iterator, we get a new one
            batches = batch_iter(y, tx, batch_size)
            y,tx = batches.__next__()
            
        gradient = compute_stoch_gradient(y,tx,w)
        loss = compute_loss(y,tx,w)
        w_new = w - gamma*gradient
        w = w_new
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_epochs - 1, l=loss, w0=w[0], w1=w[1]))
    return ws[-1]
    
def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    #We simply compute the normal gradient of this batch
    return compute_gradient_MSE(y,tx,w)


#-------------------------------------------------------------------------------
#                               Least Squares
#-------------------------------------------------------------------------------
def least_squares(y,tx):
    """Least squares regression using normal equations"""
    """calculate the least squares solution."""
    transpose = tx.T
    w = np.linalg.solve(np.dot(transpose,tx),np.dot(transpose,y))
    return w

#-------------------------------------------------------------------------------
#                               Ridge Regression
#-------------------------------------------------------------------------------
  
def ridge_regression(y, tx, lamb):
    """Ridge regression using normal equations"""
    transpose = tx.T
    lambdaIden = lamb/(2*y.shape[0])*np.eye(tx.shape[1])
    LHS = np.dot(transpose,tx)+lambdaIden
    RHS = np.dot(transpose,y)
    beta = np.linalg.solve(LHS,RHS)
    return beta

#-------------------------------------------------------------------------------
#                               Logisitc regression
#-------------------------------------------------------------------------------

def sigmoid(t):
    """apply sigmoid function on t."""
    return np.exp(t) / (1 + np.exp(t))

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    tmp = np.dot(tx,w)
    diff = np.log(1+np.exp(tmp)) - y*tmp
    loss = np.sum(diff)

    if(loss < 0):
        print(diff.shape)
        print(diff)
    return loss

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    s = sigmoid(tx.dot(w))
    tmp = s - y
    return (tx.T).dot(tmp)

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    sigmoid_txw = sigmoid(tx.dot(w))
    S = sigmoid_txw*(1-sigmoid_txw)*np.eye(len(y))
    H = (tx.T).dot(S.dot(tx))
    return H

def learning_by_gradient_descent(y, tx, w, alpha):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y,tx,w)
    gradient = calculate_gradient(y,tx,w)
    w = w - alpha*gradient
    return loss, w

def learning_by_newton_method(y, tx, w, alpha):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """
    loss = calculate_loss(y,tx,w)
    gradient = calculate_gradient(y,tx,w)
    hessian = calculate_hessian(y,tx,w)
    w = w - alpha*np.linalg.inv(hessian).dot(gradient)
    return loss, w

def adapt_y(y):
    """This function help us convert y where the labels are -1 and 1 to 0 and 1, to fit the logistic regression seen in class"""
    y_new = np.zeros((y.shape[0],y.shape[1]))
    y_new[np.where(y == 1)] = 1
    return y_new

def logistic_regression(y, tx, gamma,max_iters):
    """Logistic regression using GD or SGD"""
    threshold = 1e-6
    losses = []
    w = np.zeros((tx.shape[1], 1))
    #We reshape y because we expect it to be a 2D vector and not just 1D vector in our formulas
    y = adapt_y(y.reshape(y.shape[0],1))
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        #loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        loss, w = learning_by_newton_method(y,tx,w,gamma)
        
        # log info
        if(iter % 500 == 0):
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            print("Converged : " ,losses[-1] - losses[-2])
            break
        if loss > 10e6 :
            print("diverges")
            break
    return w


#-------------------------------------------------------------------------------
#                               Regularized Logistic
#-------------------------------------------------------------------------------


def learning_by_penalized_gradient(y, tx, w, alpha, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss = calculate_loss(y,tx,w) + lambda_*np.linalg.norm(w)**2
    gradient = calculate_gradient(y,tx,w) + 2*lambda_*w
    hessian = calculate_hessian(y,tx,w) + 2*lambda_*np.eye(tx.shape[1])
    w = w - alpha*(np.linalg.inv(hessian).dot(gradient))
    return loss, w
    
def reg_logistic_regression(y, tx, lambda_ , gamma, max_iters):
    threshold = 1e-8
    losses = []

    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, alpha, lambda_)
        # log info
        if iter % 500 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criteria
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w














