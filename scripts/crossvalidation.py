import numpy as np
from optimization import *
from costs import *
from helpers import *
import matplotlib.pyplot as plt

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    toReturn = np.power(x,1)
    for i in range(2,degree+1):
        toReturn = np.hstack([toReturn,np.power(x,i)])
    return standardize(toReturn)[0]


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_,model,degree=1,poly_built_already=False):
    """return the loss of the selected model"""
    test_y = y[k_indices[k]]
    test_x = x[k_indices[k]]
    train_k_indices = k_indices[[i for i in range(len(k_indices)) if i != 2]].ravel()
    train_y = y[train_k_indices]
    train_x = x[train_k_indices]

    if(not poly_built_already):
        phi_train_x = build_poly(train_x,degree)
        phi_test_x = build_poly(test_x,degree)
    else:
        phi_train_x = train_x
        phi_test_x = test_x

    if(model == 0):
        weight_training = least_squares_GD(train_y,phi_train_x,lambda_,300)
    elif(model == 1):
        weight_training = least_squares_SGD(train_y,phi_train_x,lambda_,500)
    elif(model == 2):
        weight_training = least_squares(train_y,phi_train_x)
    elif(model == 3):
        weight_training = ridge_regression(train_y, phi_train_x, lambda_)
    elif(model == 4):
        weight_training = logistic_regression(train_y, phi_train_x, lambda_,10000)
    else :
        weight_training = reg_logistic_regression(train_y, phi_train_x, lambda_,10000)

    e = train_y - np.dot(phi_train_x,weight_training) 
    loss_tr = compute_rmse(train_y,phi_train_x,weight_training)
    loss_te = compute_rmse(test_y,phi_test_x,weight_training)
    return loss_tr, loss_te



# ==================== METHODS TO OBTAIN OPTIMAL PARAMETERS =======================================
def get_best_parameters_GD(y,tx):
    # We limit ourselves in term of iteration to 300 and degree 1
    seed = 3
    k_fold = 6
    lambdas = 0.1 + np.logspace(-2, -1, 10)
    k_indices = build_k_indices(y, k_fold, seed)

    mse_tr = []
    mse_te = []
    for index_lamb,lamb in enumerate(lambdas):
        loss_tr_acc = 0
        loss_te_acc = 0
        for k in range(len(k_indices)):
            loss_tr, loss_te = cross_validation(y,tx, k_indices, k, lamb,0,1,True)
            loss_tr_acc += loss_tr
            loss_te_acc += loss_te
        mse_tr.append(loss_tr_acc/len(k_indices))
        mse_te.append(loss_te_acc/len(k_indices))
        print("for lamb = ",lamb," loss_tr = ",mse_tr[index_lamb]," loss_te = ",mse_te[index_lamb])
        if(mse_tr[index_lamb] > 10e4 or mse_te[index_lamb] > 10e4 ):
            print("Higher values of lambda will result in high divergence yielding infinite loss")
            break
    l_index = np.argmin(mse_te)
    optimal_loss_te = np.min(mse_te)
    optimal_loss_tr = np.min(mse_tr)
    optimal_lambda = lambdas[l_index]
    return optimal_lambda,optimal_loss_tr,optimal_loss_te

def get_best_parameters_SGD(y,tx):
    # We limit ourselves in term of iteration to 300 and degree 1
    seed = 3
    k_fold = 6
    lambdas = 0.002 + np.logspace(-4, -3, 10)
    k_indices = build_k_indices(y, k_fold, seed)
    mse_tr = []
    mse_te = []
    for index_lamb,lamb in enumerate(lambdas):
        loss_tr_acc = 0
        loss_te_acc = 0
        for k in range(len(k_indices)):
            loss_tr, loss_te = cross_validation(y,tx, k_indices, k, lamb,1,1,True)
            loss_tr_acc += loss_tr
            loss_te_acc += loss_te
        mse_tr.append(loss_tr_acc/len(k_indices))
        mse_te.append(loss_te_acc/len(k_indices))
        print("for lamb = ",lamb," loss_tr = ",mse_tr[index_lamb]," loss_te = ",mse_te[index_lamb])
        if(mse_tr[index_lamb] > 10e4 or mse_te[index_lamb] > 10e4 ):
            print("Higher values of lambda will result in high divergence yielding infinite loss")
            break
    l_index = np.argmin(mse_te)
    optimal_loss_te = np.min(mse_te)
    optimal_loss_tr = np.min(mse_tr)
    optimal_lambda = lambdas[l_index]
    return optimal_lambda,optimal_loss_tr,optimal_loss_te

def cross_validate_Least_Squares(y,tx):
    seed = 3
    k_fold = 6
    loss_tr_acc = 0
    loss_te_acc = 0
    k_indices = build_k_indices(y, k_fold, seed)
    for k in range(len(k_indices)):
        loss_tr, loss_te = cross_validation(y, tx, k_indices, k, 0,2,1,True)
        loss_tr_acc += loss_tr
        loss_te_acc += loss_te
    mse_tr = loss_tr_acc/len(k_indices)
    mse_te = loss_te_acc/len(k_indices)
    print(" loss_tr = ",mse_tr," loss_te = ",mse_te)

def get_best_parameters_Ridge(y,tx):
    # We limit ourselves in term of iteration to 300 and degree 1
    seed = 3
    k_fold = 6
    lambdas = 0.09 + np.logspace(-2, -1, 50)
    k_indices = build_k_indices(y, k_fold, seed)
    mse_tr = []
    mse_te = []
    for index_lamb,lamb in enumerate(lambdas):
        loss_tr_acc = 0
        loss_te_acc = 0
        for k in range(len(k_indices)):
            loss_tr, loss_te = cross_validation(y, tx, k_indices, k, lamb,3,1,True)
            loss_tr_acc += loss_tr
            loss_te_acc += loss_te
        mse_tr.append(loss_tr_acc/len(k_indices))
        mse_te.append(loss_te_acc/len(k_indices))
        #print("for lamb = ",lamb," loss_tr = ",mse_tr[index_lamb]," loss_te = ",mse_te[index_lamb])
        if(mse_tr[index_lamb] > 10e4 or mse_te[index_lamb] > 10e4 ):
            print("Higher values of lambda will result in high divergence yielding infinite loss")
            break
    l_index = np.argmin(mse_te)
    optimal_loss_te = np.min(mse_te)
    optimal_loss_tr = np.min(mse_tr)
    optimal_lambda = lambdas[l_index]
    return optimal_lambda,optimal_loss_tr,optimal_loss_te

def get_best_parameters_Logistic(y,tx):
    # We limit ourselves in term of iteration to 300 and degree 1
    seed = 3
    k_fold = 6
    lambdas = np.logspace(-5, 1, 10)
    k_indices = build_k_indices(y, k_fold, seed)
    mse_tr = []
    mse_te = []
    for index_lamb,lamb in enumerate(lambdas):
        loss_tr_acc = 0
        loss_te_acc = 0
        for k in range(len(k_indices)):
            loss_tr, loss_te = cross_validation(y, tx, k_indices, k, lamb,4,1,True)
            loss_tr_acc += loss_tr
            loss_te_acc += loss_te
        mse_tr.append(loss_tr_acc/len(k_indices))
        mse_te.append(loss_te_acc/len(k_indices))
        print("for lamb = ",lamb," loss_tr = ",mse_tr[index_lamb]," loss_te = ",mse_te[index_lamb])
        if(mse_tr[index_lamb] > 10e4 or mse_te[index_lamb] > 10e4 ):
            print("Higher values of lambda will result in high divergence yielding infinite loss")
            break
    l_index = np.argmin(mse_te)
    optimal_loss_te = np.min(mse_te)
    optimal_loss_tr = np.min(mse_tr)
    optimal_lambda = lambdas[l_index]
    return optimal_lambda,optimal_loss_tr,optimal_loss_te


# =====================================================================
# ========================For future improvements======================
# =====================================================================
def build_poly_with_feature_selection(x,degree):
    toReturn = np.power(x,1)
    for i in range(2,degree+1):
        toReturn = np.hstack([toReturn,np.power(x,i)])
    return standardize_feature_selection(toReturn)[0]

def cross_validate_feature_selection(y,x):
    seed = 1
    k_fold = 4
    lambdas = np.append([0],np.logspace(-5, 2, 30))
    degrees = [1]
    feature_selection = [False,True]
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    for f in feature_selection:
        
        if f :
            print("========> With feature Selection")
        else:
            print("========> NO feature Selection")
        mse_tr = np.zeros((lambdas.shape[0],len(degrees)))
        mse_te = np.zeros((lambdas.shape[0],len(degrees)))
        for index_degree,degree in enumerate(degrees):
            if feature_selection :
                phi_x = build_poly_with_feature_selection(x,degree)
            else :
                phi_x = build_poly(x,degree)
            for index_lamb,lamb in enumerate(lambdas):
                loss_tr_acc = 0
                loss_te_acc = 0
                for k in range(len(k_indices)):
                    loss_tr, loss_te = cross_validation(y, phi_x, k_indices, k, lamb,1,degree,True)
                    loss_tr_acc += loss_tr
                    loss_te_acc += loss_te
                mse_tr[index_lamb,index_degree] = loss_tr_acc/len(k_indices)
                mse_te[index_lamb,index_degree] = loss_te_acc/len(k_indices)
                print("for lamb,deg = ",lamb,",",degree," loss_te = ",mse_te[index_lamb,index_degree])
            indices = np.argmin(mse_te)
            l_index,d_index = (indices/len(mse_te[0]),indices%len(mse_te[0]))
            optimal_lambda = lambdas[l_index]
            optimal_degree = degrees[d_index]
            print("The optimal hyper-parameters are ",optimal_lambda,",",optimal_degree)
    return 0
# =====================================================================
# =====================================================================
# =====================================================================

