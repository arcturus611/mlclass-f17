#%% Code for Machine Learning homework 4 (Fall 2017)

#%% Import packages
from cvxpy import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
def gamma_choose_heuristic(X):
    # Function: For the rbf kernel, we have a heuristic 
    # to choose gamma which helps to narrow down the range of search. 
    # This function generates the best gamma acc. to the heuristic, so we
    # only search around it. 
    
    # Input: data set
    # Output: heuristic gamma
    
    #WARNING: input MUST be a vector of len > 1
    
    dists = np.square(X[:, None] - X[None, :])
    dists_vec = np.reshape(dists, len(X)*len(X), 1)
    heu_gamma = 1/np.median(dists_vec)
    return heu_gamma

#%%
def solve_for_alpha(K, lam, y, err_fn):
    # Function: Solve a regression problem with some input 
    # error function, data matrix and target values, by using cvxpy 
    # (as opposed to coding in the closed-form expression)
    
    # Input: Data matrix (K), parameter lambda, target value y and error function
    # Output: alpha (solved by cvxpy)
    
    n = len(y) 
    alpha = Variable(n)#(true soln (for ls err_fn): np.linalg.solve(K + lam*np.identity(n), y))
    if (err_fn == "huber"):    
        objective = Minimize(sum_entries(huber(K*alpha - y, 1)) + lam*quad_form(alpha, K))
    elif (err_fn == "ls"):
        objective = Minimize(sum_squares(K*alpha - y) + lam*quad_form(alpha, K))
    else: 
        print("Invalid error function selected, please select huber or ls")
    
    prob = Problem(objective)
    result = prob.solve()
    return alpha.value

#%% 
def generate_exp_kernel(X, Y, gamma):
    # Function: given "vectors" X, Y and param gamma, we compute [K_{ij}]= (exp(-gamma*(x_i - x_j)^2)
   
    # Input: X (dims1), Y (dims2), gamma (natural number)
    # Output: K (dims1 X dims2) matrix corr. to above kernel
    
    # WARNING: First input must be either vector (for training phase)
    # OR, a scalar (for testing)
    
    if(type(X)!=np.ndarray):
        K_exp = np.exp(-gamma*np.square(X - Y[None, :]))
        K_exp = K_exp[0]
    else:
        K_exp = np.exp(-gamma*np.square(X[:, None] - Y[None, :]))
    return K_exp 

#%% 
def leave_one_out_of_vec(X, idx):
    # Function: Return vector with element at X(idx) removed, 
    # and also that removed elemnt
    
    # Input: data structure from which idx-th element has to be removed
    # Output: removed stuff and stuff with idx-th element removed
    
    mask = np.ones(len(X), dtype = bool)
    mask[idx] = False
    vec = X[mask]
    elem = X[idx]
    return vec, elem

#%% 
def leave_one_out_of_mat(X, idx):
    # Input: matrix X, idx for which row and col are to be removed
    # Output: matrix Z, submatrix from X, removing idx-th row and col
    # Assuming it's a square matrix (of course)
    
    mask = np.ones(X.shape[0], dtype = bool)
    mask[idx] = False
    Z = X[np.ix_(mask, mask)]
    return Z

#%%
def predict_x_exp(train, test, gamma, alpha_hat):
    # Input: train points, test point, gamma(for exp kernel), model alpha
    return np.inner(alpha_hat.T, generate_exp_kernel(test, train, gamma))

#%%
def generate_data(n):
    # Function:  generate data according to given prompt in assignment, returning 
    # x, fx and y (which is just noisy fx)
    
    x = (np.arange(n))/(n-1)
    fx = compute_fx(x)
    eps =  np.random.randn(n)#additive gaussian noise
    y = fx + eps
    y[15] = 0 #outlier 
    return x, fx, y

#%%
def compute_fx(x):
    # Function: A step function. Generated using greater_equal fn of numpy
    # Note, if we don't multiply by 1, the datatype remains bool for each greater_equal
    # output, and True + True = 1, not 2, so we'll end up with just a single step function (which is wrong)
    
    return 10*(1*np.greater_equal(x, 1/5) + 1*np.greater_equal(x, 2/5) + 1*np.greater_equal(x, 3/5) + 1*np.greater_equal(x, 4/5))

#%% 
def compute_pred_error(pred_val, true_val):
    err = np.power(pred_val - true_val, 2)
    return err

#%% 
def learn_exp_kernel_by_cv(lam, gamma, x, y, fx, err_fn):
    # Input: lam and d are hyperparams; x is vector of input data and y are corrupt values and true fn vals
    # Note that err_fn is either huber or ls. 
    # Output: total error over all cross-validation runs using these hyperparams
    # We use "unused" to denote unused element 
    #%%
    K = generate_exp_kernel(x, x, gamma)
    total_err = 0
    #%%
    for i in range(len(x)):
        #%% 
        # Remember,i is the one you are leaving out (test data) 
        K_tr = leave_one_out_of_mat(K, i)
        (y_tr, y_test) = leave_one_out_of_vec(y, i) 
        fx_test = fx[i] #true (noiseless val of fn on x_test)
        #rememeber, we ned to use y, not fx, for training (noise incl)
        # also, remmber, y_test is not what we use 
        #%% 
        alpha_hat = solve_for_alpha(K_tr, lam, y_tr, err_fn)
        #print("in LOO loop, testing on data point i = {}".format(i))
        (data_tr, data_test) = leave_one_out_of_vec(x, i)
        pred_test = predict_x_exp(data_tr, data_test, gamma, alpha_hat)
        total_err+= compute_pred_error(pred_test, fx_test)
    #%%
    return total_err
#%% 
if __name__ == '__main__':
    #%% Generate data with desired input parameters 
    # Inputs: Number of data points and error function used
    n = 30 
    (x, fx, y) = generate_data(n)
    lam = np.linspace(.001, 5, 15) #hard-coded, based on trial-and-error 
    heu_gamma = gamma_choose_heuristic(x) #heuristic 
    print("heuristic computed value is {}".format(heu_gamma))
    gamma = np.linspace(heu_gamma-1, heu_gamma+60, 15) #400
    total_err = np.empty((len(lam), len(gamma)))
    err_fn = "ls" #choose "ls" or "huber"
    
    #%%learn the kernel for each hyperparam pair using loocv on input data
    for i in np.arange(len(lam)):
        print("i = {}".format(i))
        for j in np.arange(len(gamma)):
            print("~~~j = {}".format(j))
            total_err[i, j] = learn_exp_kernel_by_cv(lam[i], gamma[j], x, y, fx, err_fn)
    
    #%% based on total_err, we find the best lam and gamma
    (lam_min_idx, gamma_min_idx)= np.unravel_index(total_err.argmin(), total_err.shape)
    best_lam = lam[lam_min_idx]
    best_gamma = gamma[gamma_min_idx]
    print("(best_lam = {},  best_gamma = {}), and equals total_err[best_lam, best_gamma]= {}".format(best_lam, best_gamma, total_err[lam_min_idx, gamma_min_idx]))
    
    #%% Now we build the predictor
    K_learnt = generate_exp_kernel(x, x, best_gamma)
    alpha_learnt = solve_for_alpha(K_learnt, best_lam, y, err_fn) #This allows us to choose either huber or ls
    
    #%% using learnt hyperparams and alpha, eval on a range of points with unif. interval
    num_new_test_points = 200
    new_test_points = np.linspace(0, 1, num_new_test_points)
    fx_new_test_points = compute_fx(new_test_points)
    fxhat_new_test_points = np.empty(num_new_test_points)
    for i in range(num_new_test_points):
        fxhat_new_test_points[i] = np.inner(alpha_learnt.T, generate_exp_kernel(new_test_points[i], x, best_gamma))
    
    #%% plot everythng. 
    sns.set()
    
    fig = plt.figure(1)
    fig, ax = plt.subplots(1, 1)
    ax.scatter(x, y, c='k')
    ax.plot(new_test_points, fx_new_test_points, c='b')
    ax.plot(new_test_points, fxhat_new_test_points, c = 'r')
    ax.set_xlim([0,1])
    ax.set_ylim([-1.5, 41.5])
    ax.legend(['true fit', 'learnt fit', 'data'])
    ax.set_title('Learning exp. kernel hyperparameters with {} loss function'.format(err_fn))    
