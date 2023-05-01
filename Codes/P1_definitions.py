import numpy as np
import matplotlib.pyplot as plt  
import sys  

# For modeling 
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures  
from random import random, seed

# For Franke function plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Scaling
from sklearn.preprocessing import StandardScaler

# For bootstrap 
from sklearn.utils import resample

# For Gradient Descent
from numpy.linalg import norm


# ******************** FRANKE FUNCTION *************************

def FrankeFunction(x,y):
    """ Defining Franke Function (2D), defined for (x,y) in [0,1] """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def FrankeFunction_noise(x,y,coeff_noise):
    """ Defining Franke Function (2D) plus a noise, defined for (x,y) in [0,1] """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    n = len(x)                                # lenght x (number of element in the x vector)
    np.random.seed(19)
    noise = np.random.normal(0,1,n*n)         # 0=mean, 1=standard deviation, n*n=number of elements in array noise (it is a VECTOR!)
    noise = noise.reshape(len(x),len(x))      # matrix n \times n
    return term1 + term2 + term3 + term4 + coeff_noise*noise



# ******************** PLOTTING *************************

def plot3Dfunction(x,y,z):
    """ input: x (array) and y (array) --> points where the function is evaluated 
        output: z (array) --> values ​​of the function evaluated in (x,y)
        x,y,z must have the same length  """

    # Plot the surface.
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d') # ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    ax.set_xlabel('x')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')
   

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=15, pad=0.15) 

    return plt.show()



# ******************** DATA POINT ************************* 

def xy_data_equally(n):
    """ Input given by the space [0,1] EQUALLY divided 
        input: n (int) = number of data 
        outpuy: x,y (array) = two matrixes n \times n --> we have n^2 datapoint (x,y). """
    x = np.linspace(0,1,n)
    y = np.linspace(0,1,n)
    x,y = np.meshgrid(x,y)   #It makes a grid in the square (rectangle) x \times y.
    return x,y

def xy_data_uniform(n):
    """ Input from a UNIFORM distribution 
        input: n (int) = number of data 
        outpuy: x,y (array) = two matrixes n \times n --> we have n^2 datapoint (x,y). """
    x = np.sort(np.random.uniform(0, 1, n))
    y = np.sort(np.random.uniform(0, 1, n))
    x,y = np.meshgrid(x,y)
    return x,y



# ******************** MEAN SQUARED ERROR/ R2/ MEAN VALUE  ************************* 

def MSE_function(y_data, y_model, case):
    """ Compute the Mean Squared Error function of the model.
        input: y_data (array) = target values. Shape = (n_datapoints, 1)
               y_model (array)= model predictions. If case=='scalar' y_model.shape = (n_datapoints, 1)
                                                   If case=='vector' y_model.shape = (n_datapoints, n), with n= number of different models
               case (string) = 'scalar' or 'vector'. If the y_model is an array containing the prediction of just one model choose 'scalar'
                                otherwise 'vector' (y_model is a matrix with different predictions on each column).
        outpu: MSE (float or tuple) = sum of squared difference between target and the predicted values divided by total number of data."""
    
    if case == 'scalar':
        N=np.size(y_model)
        MSE = np.sum((y_data-y_model)**2)/N
    elif case == 'vector':
        MSE = np.mean((y_data - y_model)**2, axis=0, keepdims=True).ravel()
    else:
        print('Invalid case. If the prediction (y_model) is an array containing the prediction of just one model write \'scalar\' otherwise \'vector\'')
        sys.exit()
    return MSE

def R2_function(y_data, y_model, case):
    """ Compute the R2 score function of the model.
        Usually models with R^2 closer to 1 are better than those with R^2 closer to 0 (or being negative).
        input: y_data (array) = target values. Shape = (n_datapoints, 1)
               y_model (array) = model predictions. If case=='scalar' y_model.shape = (n_datapoints, 1)
                                                    If case=='vector' y_model.shape = (n_datapoints, n), with n= number of different models
               case (string) = 'scalar' or 'vector'. If the y_model is an array containing the prediction of just one model choose 'scalar'
                                otherwise 'vector' (y_model is a matrix with different predictions on each column).
        outpu: R2 (float or tuple) = model variance divided by total variance --> -inf < R^2 < 1. """

    if case == 'scalar':
        R2 = 1 - np.sum((y_data-y_model)**2)/np.sum((y_data- np.mean(y_data))**2)
    elif case == 'vector':
        R2 = np.zeros(y_model.shape[1])
        for i in range(y_model.shape[1]):
            R2[i] = 1 - np.sum((y_data.ravel()-y_model[:,i])**2)/np.sum((y_data.ravel() - np.mean(y_data.ravel()))**2)
    else:
        print('Invalid case. If the prediction (y_model) is an array containing the prediction of just one model write \'scalar\' otherwise \'vector\'')
        sys.exit()
    return R2

def mean_data(y_data):
    """ Compute the arithmetic mean value of the input data """
    N=np.size(y_data)
    return np.sum(y_data)/N



# ******************** DESIGM MATRIX/ OLS OPTIMAL BETA *************************

def Design_Matrix_calc(degree, xy):
    """ Compute the design matrix
        input: degree (int) = degree of the polynomial (complexity)
               xy (array) = dapoints (matrix n \times 2 whit x as first column and y as second column) 
        outpu: matrix (array) = design matrix [1 | x | y | x^2 | xy | y^2 | x^3 | x^2y | xy^2 | y^3 | ... ] """
    poly = PolynomialFeatures(degree)            # degree of the polynomial (complexity)
    matrix = poly.fit_transform(xy)              # data is the datapoints
    return matrix
        
def beta_calc(X, y):
    """ Computing the optimal beta for OLS 
        input: X (array) = design matrix obtain from data (x,y)
               y (array) = values of Franke function evaluated at (x,y), i.e. z = F(x,y)
        output: beta (array) = optimal beta (array) """
    beta = np.linalg.pinv(X.T@X) @X.T @ y
    return beta



# ******************** SCALING ***************************

def data_scaling(X_train, X_test):
    """ Computing the scaling of the features using StandardScaler from Scikitlearn.
        Standardizing (scaling) = zero mean and unit standard deviation (for each feature)--> scaled_xi = (xi - mean(x))/standarddeviation(x).
        Keep the intercept column to 1 and not scale the input y  

        input: X_train (array) = train design matrix obtain from data (x,y), which is used to fit the scaler.
               X_test (array) = test design matrix obtain from data (x,y). The fit scaler is used to transform train and test sets
               y (array) = values of Franke function evaluated at (x,y), i.e. z = F(x,y)
        output: scaled_X_train, scaled_X_test (array) = scaled training and test matrix """
    
    scaler_x = StandardScaler(with_mean=True, with_std=True)
    scaler_x.fit(X_train)
    scaled_X_train = scaler_x.transform(X_train)
    scaled_X_test = scaler_x.transform(X_test)
    # 1)keep the intercept
    scaled_X_train[:,0] = np.ones((X_train.shape[0],1)).ravel()
    scaled_X_test[:,0] = np.ones((X_test.shape[0],1)).ravel()

    return scaled_X_train, scaled_X_test



# ******************** BOOTSTRAP *************************

def bootstrap(n_boostraps, degree, x_train, y_train, x_test, y_test, case, lmb ):
    """ Computing the bootstrap for a fixed polynomial degree
        input: n_bootstrap (int) = number of bootstraps
               degree (int) = degree of the polynomial 
               x_train, y_train, x_test, y_test (array) = from the train_test_split of the data
               case (string) = any case between OLS, ridge and lasso
               lmb (float or int) = regularization parameter. For OLS put lmb = 0
        output: MSE_mean (float) = average of the mean square errors from all bootstrap
                y_pred (array) = predicted output of the model (vector) """

    y_pred = np.empty((y_test.shape[0], n_boostraps))
    # Define design matrices (train and test)
    design_matrix_train = Design_Matrix_calc(degree, x_train)
    design_matrix_test = Design_Matrix_calc(degree, x_test)

    for i in range(n_boostraps):
        X_, y_ = resample(design_matrix_train, y_train)
        #computing the beta
        if case == 'OLS' :
            beta = beta_calc(X_, y_)
            y_pred[:, i] = (design_matrix_test @ beta).ravel()   
        elif case == 'ridge' :
            beta = Ridge_beta_calc(X_, y_, lmb)
            y_pred[:, i] = (design_matrix_test @ beta).ravel() 
        elif case == 'lasso' :
            RegLasso = linear_model.Lasso(lmb, fit_intercept=True, max_iter = 1000000)
            RegLasso.fit(X_, y_)
            y_pred[:, i] = RegLasso.predict(design_matrix_test)
        else:
            print('The possible cases are: OLS, ridge and lasso. For OLS choose lmb =0, for the other two cases choose the lamba of interest')
            sys.exit("The specified case is not among the possible cases. Choose between 'OLS', 'ridge' or 'lasso'. ")

    MSE_mean = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True))
    return MSE_mean, y_pred



# ******************** BOOTSTRAP WITH SCALING *************************

def bootstrap_scaled(n_boostraps, degree, x_train, y_train, x_test, y_test, case, lmb ):
    """ Computing the bootstrap for a fixed polynomial degree. Models are evaluated on the same test data each time. Only the input are scaled.
        input: n_bootstrap (int) = number of bootstraps
               degree (int) = degree of the polynomial 
               x_train, y_train, x_test, y_test (array) = from the train_test_split of the data
               case (string) = any case between OLS, ridge and lasso
               lmb (float or int) = regularization parameter. For OLS put lmb = 0
        output: MSE_mean (int) = average of the mean square errors from all bootstrap
                y_pred (array) = predicted output of the model (vector) """
    
    y_pred = np.empty((y_test.shape[0], n_boostraps))
    # Define design matrices to apply scaling
    design_matrix_train = Design_Matrix_calc(degree, x_train)
    design_matrix_test = Design_Matrix_calc(degree, x_test)
    # Scaling the data 
    scaled_design_matrix_train, scaled_design_matrix_test = data_scaling(design_matrix_train, design_matrix_test)

    for i in range(n_boostraps):
        # Resample the data 
        X_, y_ = resample(scaled_design_matrix_train, y_train)
        # Computing the beta
        if case == 'OLS' :
            beta = beta_calc(X_, y_)
            y_pred[:, i] = (scaled_design_matrix_test @ beta).ravel() 
        elif case == 'ridge' :
            beta = Ridge_beta_calc(X_, y_, lmb)
            y_pred[:, i] = (scaled_design_matrix_test @ beta).ravel() 
        elif case == 'lasso' :
            RegLasso = linear_model.Lasso(lmb, fit_intercept=True, max_iter = 1000000)
            RegLasso.fit(X_, y_)
            y_pred[:, i] = RegLasso.predict(scaled_design_matrix_test)
        else:
            print('The possible cases are: OLS, ridge and lasso. For OLS choose lmb =0, for the other two cases choose the lamba of interest')
            sys.exit("The specified case is not among the possible cases. Choose between 'OLS', 'ridge' or 'lasso'. ")

    MSE_mean = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True))
    return MSE_mean, y_pred



# ******************** CROSS-VALIDATION *************************

def cross_validation(kfold, k, x, y, degree, case, lmb):
    """ Computing the cross-validation for a polynomial of degree 'degree'
        input: kfold =  splitting data into 'k' sets (from sklearn.model_selection)
               k (int) = number of folds
               x (array) = input data set
               y (array) = output data set
               degree (int) = degree of the polynomial 
               case (string) = any case between OLS, ridge and lasso
               lmb (float) = regularization parameter. For OLS put lmb = 0
        output: MSE (int) = average of the mean square errors from all k folds """

    MSE= np.zeros(k)
    j = 0

    for train_inds, test_inds in kfold.split(x):
        x_train = x[train_inds]
        y_train = y[train_inds]

        x_test = x[test_inds]
        y_test = y[test_inds]

        design_matrix_train = Design_Matrix_calc(degree, x_train)
        design_matrix_test = Design_Matrix_calc(degree, x_test)

        if case=='OLS':
            beta = beta_calc(design_matrix_train, y_train) 
            y_pred = design_matrix_test @ beta
        elif case=='ridge':
            beta = Ridge_beta_calc(design_matrix_train, y_train,lmb)
            y_pred = design_matrix_test @ beta
        elif case=='lasso':
            RegLasso = linear_model.Lasso(lmb, fit_intercept=True, max_iter = 1000000)
            RegLasso.fit(design_matrix_train, y_train)
            y_pred = RegLasso.predict(design_matrix_test)
        else:
            print('The possible cases are: OLS, ridge and lasso. For OLS choose lmb =0, for the other two cases choose the lamba of interest')
            sys.exit("The specified case is not among the possible cases. Choose between 'OLS', 'ridge' or 'lasso'. ")
        
        MSE[j] = np.mean((y_test - y_pred)**2)         #np.sum((y_pred - y_test)**2)/np.size(y_pred)
        j += 1

    return np.mean(MSE)



# ******************** CROSS-VALIDATION WITH SCALING *************************

def cross_validation_scaled(kfold, k, x, y, degree, case, lmb):
    """ Computing the cross-validation for a polynomial of degree 'degree'
        input: kfold =  splitting data into 'k' sets (from sklearn.model_selection)
               k (int) = number of folds
               x (array) = input data set
               y (array) = output data set
               degree (int) = degree of the polynomial 
               case (string) = any case between OLS, ridge and lasso
               lmb (float) = regularization parameter. For OLS put lmb = 0
        output: MSE_mean = average of the mean square errors from all k folds """

    # Define design matrix. We split into train and test and then we scalaed the data in the kfold loop
    design_matrix= Design_Matrix_calc(degree, x)

    MSE = np.zeros(k)
    j = 0

    for train_inds, test_inds in kfold.split(x):
        # Train
        X_train = design_matrix[train_inds]
        y_train = y[train_inds]
        # Test
        X_test = design_matrix[test_inds]
        y_test = y[test_inds]
        # Scaling the data 
        scaled_design_matrix_train, scaled_design_matrix_test = data_scaling(X_train, X_test)

        if case=='OLS':
            beta = beta_calc(scaled_design_matrix_train, y_train) 
            y_pred = scaled_design_matrix_test @ beta
        elif case=='ridge':
            beta = Ridge_beta_calc(scaled_design_matrix_train, y_train,lmb)
            y_pred = scaled_design_matrix_test @ beta
        elif case=='lasso':
            RegLasso = linear_model.Lasso(lmb, fit_intercept=False, max_iter = 1000000)
            RegLasso.fit(scaled_design_matrix_train, y_train)
            y_pred = RegLasso.predict(scaled_design_matrix_test)
        else:
            print('The possible cases are: OLS, ridge and lasso. For OLS choose lmb =0, for the other two cases choose the lamba of interest')
            sys.exit("The specified case is not among the possible cases. Choose between 'OLS', 'ridge' or 'lasso'. ")
        MSE[j] = np.mean((y_test - y_pred)**2)         #np.sum((y_pred - y_test)**2)/np.size(y_pred)  
        j += 1

    return np.mean(MSE)



# ******************** RIDGE OPTIMAL BETA *************************

def Ridge_beta_calc(X, y, lmb):
    """ Computing the optimal beta for Ridge 
        input: X (array) = design matrix obtain from data (x,y)
               y (array) = values of Franke function evaluated at (x,y), i.e. z = F(x,y)
               lmb (float) = regularization parameter 
        output: beta (array)  = optimal beta (X^TX + lambda I)^(-1) X^T y"""
    n = len(X[0])
    I = np.identity(n)
    inv_matrix = X.T@X + lmb * I
    beta = np.linalg.pinv(inv_matrix) @X.T @ y
    return beta



# ******************** GRADIENT DESCENT *************************

def GD(X, y, n, n_features, Niterations, eta, lmb, tol):
    """ Computing the gradient descent with varying (or not) learning rate.
        input: X (array) = design metrix
               y (array) = output data set
               n (int) = number of data points
               Niterations (int) = number of iterations
               eta_range (float or array) = learning rate. It can be either a scalar or a vectors of varying learning rate
               lmb (float) = hyperparameter for Ridge regression
               tol (float) = stop criterion (|| gradient ||_2 < tol)
        output: beta (float or array) = computed beta for given eta (can be either a scalar or a vector of varying hyperparameter) """


    if isinstance(eta,float) or isinstance(eta,int):
        np.random.seed(19)
        beta = np.random.randn(n_features,1)
        gradient_norm = tol + 10.0
        for iter in range(Niterations):
            if gradient_norm > tol:
                gradient = (2.0/n)*X.T @ (X @ beta-y) + 2*lmb*beta
                beta -= eta*gradient
                gradient_norm = norm(gradient)
            else:
                print('learning rate = {:.2e}'.format(eta) + ' --> iteration number {} with'.format(iter) + ' norm 2 of the gradient of the loss function = {:.3e}'.format(gradient_norm)) 
                break
    else:
        np.random.seed(19)
        beta = np.zeros((n_features, len(eta)))
        for j,etaj in enumerate(eta):
            betaj = np.random.randn(n_features,1)
            gradient_norm = tol + 10.0
            for iter in range(Niterations):
                if gradient_norm > tol:
                    gradient = (2.0/n)*X.T @ (X @ betaj-y) + 2*lmb*betaj
                    betaj = betaj - etaj*gradient
                    gradient_norm = norm(gradient)
                else:
                    print('learning rate = {:.2e}'.format(etaj) + ' --> iteration number {} with'.format(iter) + ' norm 2 of the gradient of the loss function = {:.3e}'.format(gradient_norm)) 
                    break
            beta[:,j] = betaj.ravel()

    return beta



# ******************** GRADIENT DESCENT WITH MOMENTUM *************************

def GDMomentum(X, y, n, n_features, Niterations, eta, delta, lmb, tol):
    """ Computing the gradient descent with momentum and with varying (or not) momentum hyperparameter.
        input: X (array) = design metrix
               y (array) = output data set
               n (int) = number of data points
               Niterations (int) = number of iterations
               eta (float) = learning rate. It has to be a scalar 
               delta (float or array) = momentum constant
               lmb (float) = hyperparameter for Ridge regression
               tol (fllat) = stop criterion (|| gradient ||_2 < tol)
        output: beta (array) = computed beta for given eta (can be either a scalar or a vector of varying hyperparameter) """

    if isinstance(delta,float) or isinstance(delta,int):
        np.random.seed(19)
        beta = np.random.randn(X.shape[1], 1)
        beta_old = beta
        gradient_norm = tol + 10.0
        for iter in range(Niterations):
            if gradient_norm > tol:
                gradient = (2.0/n)*X.T @ (X @ beta-y) + 2* lmb*beta
                change = (beta - beta_old)
                beta_old = beta
                beta =  beta - eta*gradient + delta*change
                gradient_norm = norm(gradient)
            else:
                print('momentum parameter = {} '.format(delta) + ' --> iteration number {} with'.format(iter) + ' norm 2 of the gradient of the loss function = {:.3e}'.format(gradient_norm))
                break
    else:
        beta = np.zeros((n_features, len(delta)))
        for j, deltaj in enumerate(delta):
            np.random.seed(19)
            betaj = np.random.randn(X.shape[1], 1)
            beta_oldj = betaj
            gradient_norm = tol + 10.0
            for iter in range(Niterations):
                if gradient_norm > tol:
                    gradient = (2.0/n)*X.T @ (X @ betaj-y) + 2* lmb*betaj
                    change = (betaj - beta_oldj)
                    beta_oldj = betaj
                    betaj =  betaj - eta*gradient + deltaj*change
                    gradient_norm = norm(gradient)
                else:
                    print('momentum parameter = {}'.format(deltaj) + ' --> iteration number {} with'.format(iter) + ' norm 2 of the gradient of the loss function = {:.3e}'.format(gradient_norm)) 
                    break
            beta[:,j] = betaj.ravel()

    return beta



# ******************** STOCHASTIC GRADIENT DESCENT *************************

def SGD(X, y, n_features, t0, t1, vector_size_minibatch, n_minibatch, vector_n_epochs, lmb):
    """ Computing the stochastic gradient descent with varying hyperparameters (number of epochs or size of minibatch).
        input: X (array) = design metrix
               y (array) = output data set
               t0, t1 (float or int) = parameters for the time decay rate (learning rate)
               vector_size_minibatch (int or array) = size of minibatch. It can be either a scalar or a vector
               n_minibatch (int or array) = number of minibatches.
               vector_n_epochs (int or array) = number of epochs. It can be either a scalar or a vector
               lmb (float) = hyperparameter for Ridge regression
        output: betas (array) = vector of computed betas for varying hyperparameter """

    if isinstance(vector_size_minibatch,int):
        betas = np.zeros((n_features,len(vector_n_epochs)))
        for j, n_epochs in enumerate(vector_n_epochs):
            np.random.seed(19)
            rng = np.random.default_rng()
            beta = rng.standard_normal(size=(n_features,1))  # np.random.randn(n_features,1)
            for epoch in range(n_epochs):
                for i in range(n_minibatch):
                    random_index = vector_size_minibatch*np.random.randint(n_minibatch)
                    xi = X[random_index:random_index+vector_size_minibatch]
                    yi = y[random_index:random_index+vector_size_minibatch]
                    gradients = (2.0/vector_size_minibatch)* xi.T @ ( xi @ beta -yi) + 2*lmb*beta
                    eta = t0/(( epoch * n_minibatch + i) + t1)
                    beta = beta - eta*gradients
            betas[:,j] = beta.reshape(n_features,)

    elif isinstance(vector_n_epochs,int):
        betas = np.zeros((n_features,len(vector_size_minibatch)))
        for j, size_minibatch in enumerate(vector_size_minibatch):
            np.random.seed(19)
            rng = np.random.default_rng()
            beta = rng.standard_normal(size=(n_features,1))  # np.random.randn(n_features,1)
            for epoch in range(vector_n_epochs):
                for i in range(n_minibatch[j]):
                    random_index = size_minibatch*np.random.randint(n_minibatch[j])
                    xi = X[random_index:random_index+size_minibatch]
                    yi = y[random_index:random_index+size_minibatch]
                    gradients = (2.0/size_minibatch)* xi.T @ (xi @ beta -yi) + 2*lmb*beta
                    eta = t0/((epoch *n_minibatch[j]+i)+t1)
                    beta = beta - eta*gradients
            betas[:,j] = beta.ravel()
    else:
        print('Pick which hyperparameter you want to vary: number of epochs or size of minibatch.')
        sys.exit()

    return betas


# ******************** ADAGRAD (as alternation of SGD) *************************

def AdaGrad(X, y, t0, t1, size_minibatch, n_minibatch, n_epochs, lmb, epsilon):
    """ Computing AdaGrad algorithm (tuning the learning rate at each step) with fixed hyperparameters.
        input: X (array) = design metrix
               y (array) = output data set
               t0, t1 (float or int) = parameters for the time decay rate (learning rate)
               size_minibatch (int) = size of minibatch. It has to a scalar
               n_minibatch (int) = number of minibatches. It has to a scalar
               n_epochs (int) = number of epochs. It has to a scalar
               epsilon (float) = small quantity to avoid division by zero
               lmb (float) = hyperparameter for Ridge regression
        output: beta (array) = computed beta for the given hyperparameters """

    betas = np.random.randn(X.shape[1], 1)
    for epoch in range(n_epochs):
        s_t = 0.0
        for i in range(n_minibatch):
            random_index = size_minibatch * np.random.randint(n_minibatch)
            xi = X[random_index:random_index + size_minibatch]
            yi = y[random_index:random_index + size_minibatch]
            gradients = (2.0 / size_minibatch) * xi.T @ ((xi @ betas) - yi) + 2 * lmb * betas
            eta = t0 / ((epoch * n_minibatch + i) + t1)
            s_t = s_t + np.square(gradients)
            eta_adjusted = eta/(np.sqrt(s_t) + epsilon)
            betas = betas - eta_adjusted * gradients
 
    return betas



# ******************** RMSProp (as alternation of SGD) *************************

def RMSProp(X, y, t0, t1, size_minibatch, n_minibatch, n_epochs, lmb, epsilon, gamma):
    """ Computing RMSProp algorithm (tuning the learning rate at each step) with fixed hyperparameters.
        input: X (array) = design metrix
               y (array) = output data set
               t0, t1 (int or float)= parameters for the time decay rate (learning rate)
               size_minibatch (int)= size of minibatch
               n_minibatch (int) = number of minibatches
               n_epochs (int) = number of epochs
               lmb (float) = hyperparameter for Ridge regression
               epsilon (float) = small quantity to avoid division by zero
               gamma (float) = constant that multiplies the old gradients (Weight of old gradients)
        output: beta (array) = computed beta for the given hyperparameters """

    beta = np.random.randn(X.shape[1], 1)
    for epoch in range(n_epochs):
        s_t = 0
        for i in range(n_minibatch):
            random_index = size_minibatch * np.random.randint(n_minibatch)
            xi = X[random_index:random_index + size_minibatch]
            yi = y[random_index:random_index + size_minibatch]
            gradients = (2.0 / size_minibatch) * xi.T @ ((xi @ beta) - yi) + 2 * lmb * beta
            eta = t0 / ((epoch * n_minibatch + i) + t1)
            s_t = gamma * s_t + (1-gamma) * np.square(gradients)
            eta_adjusted = eta/np.sqrt(s_t + epsilon)
            beta = beta - eta_adjusted * gradients

    return beta


# ******************** ADAM (as alternation of SGD) *************************

def ADAM(X, y, t0, t1, size_minibatch, n_minibatch, n_epochs, lmb, epsilon, beta1, beta2):
    """ Computing ADAM algorithm (tuning the learning rate at each step) with fixed hyperparameters..
        input: X (array) = design metrix
               y (array) = output data set
               t0, t1 (int) = parameters for the time decay rate (learning rate)
               size_minibatch (int) = size of minibatch
               n_minibatch (int) = number of minibatches
               n_epochs (int) = number of epochs
               lmb (float) = hyperparameter for Ridge regression
               epsilon (float) = small quantity to avoid division by zero
               beta1, beta2 (float) = constant for the decay rates of first moment (beta2) and second moment (beta1).  First moment = sum of gradients and Second moment = sum of squared past gradients
        output: beta (array) = computed beta for the given hyperparameters """

    beta = np.random.randn(X.shape[1], 1)
    for epoch in range(n_epochs):
        # Can you figure out a better way of setting up the contributions to each batch?
        s_t = 0         # First moment (gradients)
        v_t = 0         # Second moment (squared gradients)
        for i in range(n_minibatch):
            random_index = size_minibatch * np.random.randint(n_minibatch)
            xi = X[random_index:random_index + size_minibatch]
            yi = y[random_index:random_index + size_minibatch]
            gradients = (2.0 / size_minibatch) * xi.T @ ((xi @ beta) - yi) + 2 * lmb * beta
            eta = t0 / ((epoch * n_minibatch + i) + t1)
            s_t = beta2 * s_t + (1 - beta2) * np.square(gradients)
            s_t_hat = s_t/(1-beta2**(epoch+1))
            v_t = beta1 * v_t + (1 - beta1) * gradients
            v_t_hat = v_t / (1 - beta1 ** (epoch+1))
            eta_adjusted = (eta * v_t_hat) / (np.sqrt(s_t_hat) + epsilon)
            beta = beta - eta_adjusted
    
    return beta




# ******************** GRADIENT DESCENT FOR LOGISTIC REGRESSION *************************

def GD_LogReg(X, y, n, n_features, Niterations, eta, lmb, tol):
    """ Computing the gradient descent for Logistic Regression with varying (or not) learning rate.
        input: X (array) = design metrix
               y (array) = output data set
               n (int) = number of data points
               Niterations (int) = number of iterations
               eta_range (float or array) = learning rate. It can be either a scalar or a vectors of varying learning rate
               lmb (float) = hyperparameter for Ridge regression (default 0 )
               tol (float) = stop criterion (|| gradient ||_2 < tol)
        output: beta (float or array) = computed beta for given eta (can be either a scalar or a vector of varying hyperparameter) """

    if isinstance(eta,float) or isinstance(eta,int):
        np.random.seed(19)
        beta = np.random.randn(n_features,1)
        gradient_norm = tol + 10.0
        for iter in range(Niterations):
            if gradient_norm > tol:
                yi_tilde = 1.0 / (1.0 + (np.e)**( -X @ beta)) 
                gradient = (2.0/n)* ( X.T @ (yi_tilde -y) + lmb*beta )
                beta -= eta*gradient
                gradient_norm = norm(gradient)
            else:
                print('learning rate = {:.2e}'.format(eta) + ' --> iteration number {} with'.format(iter) + ' norm 2 of the gradient of the loss function = {:.3e}'.format(gradient_norm)) 
                break

    else:
        np.random.seed(19)
        beta = np.zeros((n_features, len(eta)))
        for j,etaj in enumerate(eta):
            betaj = np.random.randn(n_features,1)
            gradient_norm = tol + 10.0
            for iter in range(Niterations):
                if gradient_norm > tol:
                    yi_tilde = 1.0 / (1.0 + (np.e)**( -X @ betaj)) 
                    gradient = (2.0/n)* ( X.T @ (yi_tilde -y) + lmb*betaj )
                    betaj = betaj - etaj*gradient
                    gradient_norm = norm(gradient)
                else:
                    print('learning rate = {:.2e}'.format(etaj) + ' --> iteration number {} with'.format(iter) + ' norm 2 of the gradient of the loss function = {:.3e}'.format(gradient_norm)) 
                    break
            beta[:,j] = betaj.ravel()

    return beta



# ******************** STOCHASTIC GRADIENT DESCENT FOR LOGISTIC REGRESSION *************************

def SGD_LogReg(X, y, n_features, t0, t1, vector_size_minibatch, n_minibatch, n_epochs, lmb):
    """ Computing the stochastic gradient descent for Logistic Regression with varying size of minibatch or fixed hyperparameters.
        input: X (array) = design metrix
               y (array) = output data set
               t0, t1 (int) = parameters for the time decay rate (learning rate)
               vector_size_minibatch (int or array of int) = size of minibatch. It can be either a scalar or a vector
               n_minibatch (int or array of int) = number of minibatches. It can be either a scalar or a vector (Depends on vector_size_minibatch)
               n_epochs (int) = number of epochs. It has to be a scalar 
               lmb (float) = hyperparameter for Ridge regression
        output: betas (array) = vector of computed betas for varying hyperparameter """

    if isinstance(vector_size_minibatch,int):
        np.random.seed(11)
        beta = np.random.randn(n_features,1)
        for epoch in range(n_epochs):
            for i in range(n_minibatch):
                random_index = vector_size_minibatch*np.random.randint(n_minibatch)
                xi = X[random_index:random_index+vector_size_minibatch]
                yi = y[random_index:random_index+vector_size_minibatch]
                yi_tilde = 1.0 / (1.0 + (np.e)**( -xi @ beta)) 
                gradients = (2.0/vector_size_minibatch)* ( xi.T @ (yi_tilde -yi) + lmb*beta )
                eta = t0/((epoch *n_minibatch+i)+t1)
                beta = beta - eta*gradients
        betas = beta

    elif len(vector_size_minibatch)>=1 :
        betas = np.zeros((n_features,len(vector_size_minibatch)))
        for j, size_minibatch in enumerate(vector_size_minibatch):
            np.random.seed(11)
            beta = np.random.randn(n_features,1)
            for epoch in range(n_epochs):
                for i in range(n_minibatch[j]):
                    random_index = size_minibatch*np.random.randint(n_minibatch[j])
                    xi = X[random_index:random_index+size_minibatch]
                    yi = y[random_index:random_index+size_minibatch]
                    yi_tilde = 1.0 / (1.0 + (np.e)**( -xi @ beta))
                    gradients = (2.0/size_minibatch)* (xi.T @ (yi_tilde -yi) + lmb*beta)
                    eta = t0/((epoch *n_minibatch[j]+i)+t1)
                    beta = beta - eta*gradients
            betas[:,j] = beta.ravel()

    else:
        print('Only the hyperparameter "size of minibatch" can vary.')
        sys.exit()

    return betas


# ******************** LOGISTIC REGRESSION PREDICTION *************************

def Prediction_LogReg(X, beta):
    """ Computing the stochastic gradient descent with varying size of minibatch or fixed hyperparameters.
        input: X (array) = design metrix. Can be either test ot train
               beta (array) = optimal beta computed before with any method. Can be a vector o a scalar
        output: ypredict (array) = logist regression predictions (0 or 1) """

    ypredict = 1.0 /  (1.0 + (np.e)**( -X @ beta))

    if beta.shape[1] > 1:
        for j in range(beta.shape[1]): 
            for i in range(ypredict.shape[0]):
                if ypredict[i,j] > 0.5:
                    ypredict[i,j] = 1
                else:
                    ypredict[i,j] = 0
    else:
        for i in range(len(ypredict)):
            if ypredict[i] > 0.5:
                ypredict[i] = 1
            else:
                ypredict[i] = 0

    return ypredict