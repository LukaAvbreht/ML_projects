import numpy as np
import xlrd
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import svd
from categoric2numeric import categoric2numeric
from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show
import sklearn.linear_model as lm
import sklearn.model_selection as skmd
from toolbox.Toolbox_Python02450.Tools.toolbox_02450 import feature_selector_lr, bmplot, rlr_validate, train_neural_net, draw_neural_net
from matplotlib.pyplot import figure, plot, xlabel, ylabel, clim, semilogx, loglog, legend, title, subplot, show, grid
import pprint
import random
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# from regression_part_a import OPT_lambda_part_2, X2, YY, XX, X2_labesls

# Again import data
airbnb_data = "../data/AB_NYC_2019.csv"

attributes_datatype = {
    'id': np.float64,  # 0
    'name': str,  # 1
    'host_id': np.float64,  # 2
    'host_name': str,  # 3
    'neighbourhood_group': str,  # 4
    'neighbourhood': str,  # 5
    'latitude': np.float64,  # 6
    'longitude': np.float64,  # 7
    'room_type': str,  # 8
    'price': np.float64,  # 9
    'minimum_nights': np.float64,  # 10
    'number_of_reviews': np.float64,  # 11
    # 'last_review': str,  # 12
    'reviews_per_month': np.float64,  # 13
    'calculated_host_listings_count': np.float64,  # 14
    'availability_365': np.float64  # 15
}

attributes_dates = ["last_review"]

data_frame_original = pd.read_csv(airbnb_data, dtype=attributes_datatype, parse_dates=attributes_dates)
data_frame_original.fillna(0, inplace=True)

print("Size of original dataframe: ", data_frame_original.size)

# TODO TAKE CARE
# Get random part of data to get more sense of visualization:
data_frame = data_frame_original.sample(frac=0.1)

raw_data = data_frame.get_values()
attributes = list(data_frame.columns)

print("Atributes of dataframe: ", attributes)

print("Size of dataframe: ", data_frame.size)

prity_atributes = [
    'id',
    'name',
    'host id',
    'host name',
    'borough',
    'neighbourhood',
    'latitude',
    'longitude',
    'room type',
    'price',
    'minimum nights',
    'review number',
    'last review',
    'rev per month',
    'host listing count',
    'availability']

# Make a list of unique room types and neighbourhoods and unique boroughs
unique_boroughs = data_frame['neighbourhood_group'].unique()
unique_roomtypes = data_frame['room_type'].unique()
unique_neighbourhoods = data_frame['neighbourhood'].unique()

# # print(unique_neighbourhoods)
# print(unique_roomtypes)
# print(unique_boroughs)


# -- Regression PART B  --
# -- 1)     --
print("\n Part B \n 3) \n")

# Try to predict room type:

rtype_dict = dict()
num_to_atribute = dict()

for j,i in enumerate(unique_roomtypes):
    rtype_dict[i] = j
    num_to_atribute[j] = i

result_atributes = (8)
result_data = raw_data[:, result_atributes]
result_data = [rtype_dict[i] for i in result_data]
# print(result_data)
Y = np.array(result_data).T
yy_data = list()
for i in result_data:
    if i == 0:
        yy_data.append(np.array([1,0,0]))
    if i == 1:
        yy_data.append(np.array([0,1,0]))
    if i == 2:
        yy_data.append(np.array([0,0,1]))
YY = np.array(yy_data)
# Y = Y.reshape((Y.shape[0], 1))
print(Y.shape)
print(YY.shape)

# Standarize our data matrix
# One out K for nbh
nbh_data = raw_data[:, (4)]
x_nbh = np.array(nbh_data).T
X_K1, K1_labels = categoric2numeric(x_nbh)

# roomtype_data = raw_data[:, (8)]
# x_rty = np.array(roomtype_data).T
# X_K2, K2_labels = categoric2numeric(x_rty)

# Get other parameters and standardise them
other_params = (9, 15)
other_data = np.array(raw_data[:, other_params])

N, M = other_data.shape

# To get a shape of (n,1) to use in concatenate (only if we only use one additional parameter
if N == 1:
    other_data = other_data.reshape((other_data.shape[0], 1))

other_data = other_data - np.ones((N, 1)) * other_data.mean(axis=0)
other_data = other_data.astype(np.float64)
other_data = other_data * (1 / np.std(other_data, 0))

# Concatenate all of the data int one matrix

X = np.concatenate((X_K1, other_data), axis=1)
X_labesls = K1_labels + [attributes[i] for i in other_params]

def compare_ann_lin_reg():
    opt_lam =15.264
    h_lays = 15
    N, M = X.shape
    K = 10
    cvf = 10
    CV = skmd.KFold(K, shuffle=False)

    Error_test_lin = [0 for i in range(K)]

    Error_test_ann = [0 for i in range(K)]

    r_values = [0 for i in range(K)]

    outk = 0
    for train_index, test_index in CV.split(X, Y):
        X_train = X[train_index]
        y_train = Y[train_index]
        X_test = X[test_index]
        y_test = Y[test_index]

        X_train = X_train.astype(np.float64)
        y_train = y_train.astype(np.float64)
        X_test = X_test.astype(np.float64)
        y_test = y_test.astype(np.float64)

        CV = skmd.KFold(cvf, shuffle=True)

        Error_test_lin_inner = [0 for i in range(cvf)]

        Error_test_ann_inner = [0 for i in range(cvf)]

        ink = 0
        for inner_train_index, inner_test_index in CV.split(X_train, y_train):

            X_train_in = X[inner_train_index].astype(np.float64)
            y_train_in = Y[inner_train_index].astype(np.float64)
            X_test_in = X[inner_test_index].astype(np.float64)
            y_test_in = Y[inner_test_index].astype(np.float64)

            X_train_in_torch = torch.tensor(X_train_in, dtype=torch.float)
            y_train_in_torch = torch.tensor(y_train_in, dtype=torch.float)
            X_test_in_torch = torch.tensor(X_test_in, dtype=torch.float)

            y_train_in = y_train_in.reshape((y_train_in.shape[0],))
            y_test_in = y_test_in.reshape((y_test_in.shape[0],))

            # Linear regressoin

            mdl = LogisticRegression(penalty='l2', C=1 / opt_lam, solver="lbfgs", multi_class="auto",
                                     max_iter=10000)

            mdl.fit(X_train_in, y_train_in)

            y_train_est = mdl.predict(X_train_in).T
            y_test_est = mdl.predict(X_test_in).T

            Error_test_lin_e = np.sum(y_test_est != y_test_in) / len(y_test_in)

            Error_test_lin_inner[ink] = Error_test_lin_e

            # ANN

            model = lambda: torch.nn.Sequential(
                torch.nn.Linear(M, h_lays),  # M features to H hiden units
                torch.nn.ReLU(),  # 1st transfer function
                # Output layer:
                # H hidden units to C classes
                # the nodes and their activation before the transfer
                # function is often referred to as logits/logit output
                torch.nn.Linear(h_lays, C),  # C logits
                # To obtain normalised "probabilities" of each class
                # we use the softmax-funtion along the "class" dimension
                # (i.e. not the dimension describing observations)
                torch.nn.Softmax(dim=1)  # final tranfer function, normalisation of logit output
            )

            loss_fn = torch.nn.CrossEntropyLoss()

            max_iter = 10000
            print('Training model of type:\n{}\n'.format(str(model())))

            # Do cross-validation:
            errors = []  # make a list for storing generalizaition error in each loop
            # Loop over each cross-validation split. The CV.split-method returns the
            # indices to be used for training and testing in each split, and calling
            # the enumerate-method with this simply returns this indices along with
            # a counter k:
            # for k, (train_index, test_index) in enumerate(CV.split(X, YY)):
            #     print('\nCrossvalidation fold: {0}/{1}'.format(k + 1, K))
            #
            #     # Extract training and test set for current CV fold,
            #     # and convert them to PyTorch tensors
            #     X_train = torch.tensor(X[train_index, :], dtype=torch.float)
            #     y_train = torch.tensor(YY[train_index], dtype=torch.long)
            #     X_test = torch.tensor(X[test_index, :], dtype=torch.float)
            #     y_test = torch.tensor(YY[test_index], dtype=torch.long)

            net, final_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train_in_torch,
                                                               y=y_train_in_torch,
                                                               n_replicates=3,
                                                               max_iter=max_iter)

            print('\n\tBest loss: {}\n'.format(final_loss))

            softmax_logits = net(torch.tensor(X_test, dtype=torch.float))
            # Get the estimated class as the class with highest probability (argmax on softmax_logits)
            y_test_est = (torch.max(softmax_logits, dim=1)[1]).data.numpy()
            # Determine errors
            e = (y_test_est != y_test)

            eer = sum(e) / len(e)

            Error_test_ann_inner[ink] = eer

            # increment inner index
            ink += 1

        # save errors
        Error_test_lin[outk] = Error_test_lin_inner
        Error_test_ann[outk] = Error_test_ann_inner

        # Calculate error as in 11.4.1
        r_j = sum(i-j for i,j in zip(Error_test_lin_inner,Error_test_ann_inner))/len(Error_test_lin[outk])

        r_values[outk] = r_j

        # increment outter index
        outk += 1

    return Error_test_lin,Error_test_ann,r_values

def compare_baseline_lin_reg():
    opt_lam =100
    h_lays = 15
    N, M = X.shape
    K = 10
    cvf = 10
    CV = skmd.KFold(K, shuffle=False)

    Error_test_lin = [0 for i in range(K)]

    Error_test_baseline = [0 for i in range(K)]

    r_values = [0 for i in range(K)]

    outk = 0
    for train_index, test_index in CV.split(X, Y):
        X_train = X[train_index]
        y_train = Y[train_index]
        X_test = X[test_index]
        y_test = Y[test_index]

        X_train = X_train.astype(np.float64)
        y_train = y_train.astype(np.float64)
        X_test = X_test.astype(np.float64)
        y_test = y_test.astype(np.float64)

        CV = skmd.KFold(cvf, shuffle=True)

        Error_test_lin_inner = [0 for i in range(cvf)]

        Error_test_basline_inner = [0 for i in range(cvf)]

        ink = 0
        for inner_train_index, inner_test_index in CV.split(X_train, y_train):

            X_train_in = X[inner_train_index].astype(np.float64)
            y_train_in = Y[inner_train_index].astype(np.float64)
            X_test_in = X[inner_test_index].astype(np.float64)
            y_test_in = Y[inner_test_index].astype(np.float64)

            X_train_in_torch = torch.tensor(X_train_in, dtype=torch.float)
            y_train_in_torch = torch.tensor(y_train_in, dtype=torch.float)
            X_test_in_torch = torch.tensor(X_test_in, dtype=torch.float)

            y_train_in = y_train_in.reshape((y_train_in.shape[0],))
            y_test_in = y_test_in.reshape((y_test_in.shape[0],))

            # Linear regressoin

            mu = np.mean(X_train_in[:, 1:], 0)
            sigma = np.std(X_train_in[:, 1:], 0)

            X_train_in[:, 1:] = (X_train_in[:, 1:] - mu) / sigma
            X_test_in[:, 1:] = (X_test_in[:, 1:] - mu) / sigma

            Xty = X_train_in.T @ y_train_in
            XtX = X_train_in.T @ X_train_in

            # Compute mean squared error without using the input data at all
            Error_train_nofeatures = np.square(y_train_in - y_train_in.mean()).sum(axis=0) / y_train_in.shape[0]
            Error_test_nofeatures = np.square(y_test_in - y_test_in.mean()).sum(axis=0) / y_test_in.shape[0]

            # Estimate weights for the optimal value of lambda, on entire training set
            lambdaI = opt_lam * np.eye(M)
            lambdaI[0, 0] = 0  # Do no regularize the bias term
            w_rlr = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
            # Compute mean squared error with regularization with optimal lambda
            Error_train_rlr = np.square(y_train_in - X_train_in @ w_rlr).sum(axis=0) / y_train_in.shape[0]

            Error_test_rlr = np.square(y_test_in - X_test_in @ w_rlr).sum(axis=0) / y_test_in.shape[0]

            # Estimate weights for unregularized linear regression, on entire training set
            w_noreg = np.linalg.solve(XtX, Xty).squeeze()
            # Compute mean squared error without regularization
            Error_train_lin = np.square(y_train_in - X_train_in @ w_noreg).sum(axis=0) / y_train_in.shape[0]

            # The importatn thing
            Error_test_lin_e = np.square(y_test_in - X_test_in @ w_noreg).sum(axis=0) / y_test_in.shape[0]

            Error_test_lin_inner[ink] = Error_test_lin_e

            # baseline

            y_pred = np.mean(y_train_in)

            eval_error = np.square(y_test_in - y_pred).sum(axis=0) / y_test.shape[0]

            Error_test_basline_inner[ink] = eval_error

            # increment inner index
            ink += 1

        # save errors
        Error_test_lin[outk] = Error_test_lin_inner
        Error_test_baseline[outk] = Error_test_basline_inner

        # Calculate error as in 11.4.1
        r_j = sum(i-j for i,j in zip(Error_test_lin_inner,Error_test_basline_inner))/len(Error_test_lin[outk])

        r_values[outk] = r_j

        # increment outter index
        outk += 1

    return Error_test_lin,Error_test_baseline,r_values

def compare_ann_baseline():
    opt_lam =100
    h_lays = 15
    N, M = X.shape
    K = 10
    cvf = 10
    CV = skmd.KFold(K, shuffle=False)

    Error_test_baseline = [0 for i in range(K)]

    Error_test_ann = [0 for i in range(K)]

    r_values = [0 for i in range(K)]

    outk = 0
    for train_index, test_index in CV.split(X, Y):
        X_train = X[train_index]
        y_train = Y[train_index]
        X_test = X[test_index]
        y_test = Y[test_index]

        X_train = X_train.astype(np.float64)
        y_train = y_train.astype(np.float64)
        X_test = X_test.astype(np.float64)
        y_test = y_test.astype(np.float64)

        CV = skmd.KFold(cvf, shuffle=True)

        Error_test_baseline_inner = [0 for i in range(cvf)]

        Error_test_ann_inner = [0 for i in range(cvf)]

        ink = 0
        for inner_train_index, inner_test_index in CV.split(X_train, y_train):

            X_train_in = X[inner_train_index].astype(np.float64)
            y_train_in = Y[inner_train_index].astype(np.float64)
            X_test_in = X[inner_test_index].astype(np.float64)
            y_test_in = Y[inner_test_index].astype(np.float64)

            X_train_in_torch = torch.tensor(X_train_in, dtype=torch.float)
            y_train_in_torch = torch.tensor(y_train_in, dtype=torch.float)
            X_test_in_torch = torch.tensor(X_test_in, dtype=torch.float)

            y_train_in = y_train_in.reshape((y_train_in.shape[0],))
            y_test_in = y_test_in.reshape((y_test_in.shape[0],))

            # Baseline

            y_pred = np.mean(y_train_in)

            eval_error = np.square(y_test_in - y_pred).sum(axis=0) / y_test.shape[0]

            Error_test_baseline_inner[ink] = eval_error

            # ANN

            model = lambda: torch.nn.Sequential(
                torch.nn.Linear(M, h_lays),  # M features to H hiden units
                # 1st transfer function, either Tanh or ReLU:
                torch.nn.ReLU(),
                # torch.nn.Tanh(),
                torch.nn.Linear(h_lays, 1),  # H hidden units to 1 output neuron
                # torch.nn.Sigmoid()  # final tranfer function
            )

            loss_fn = torch.nn.MSELoss()

            # Train for a maximum of 10000 steps, or until convergence (see help for the
            # function train_neural_net() for more on the tolerance/convergence))
            max_iter = 10000

            # Go to the file 'toolbox_02450.py' in the Tools sub-folder of the toolbox
            # and see how the network is trained (search for 'def train_neural_net',
            # which is the place the function below is defined)
            net, final_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train_in_torch,
                                                               y=y_train_in_torch,
                                                               n_replicates=3,
                                                               max_iter=max_iter)

            y_res = net(X_test_in_torch)

            y_res = y_res.data.numpy()
            # y_test = y_test.data.numpy()

            eval_error = np.square(y_test_in - y_res).sum(axis=0) / y_test_in.shape[0]

            Error_test_ann_inner[ink] = eval_error

            # increment inner index
            ink += 1

        # save errors
        Error_test_baseline[outk] = Error_test_baseline_inner
        Error_test_ann[outk] = Error_test_ann_inner

        # Calculate error as in 11.4.1
        r_j = sum(i-j for i,j in zip(Error_test_ann_inner,Error_test_baseline_inner))/len(Error_test_ann_inner[outk])

        r_values[outk] = r_j

        # increment outter index
        outk += 1

    return Error_test_baseline,Error_test_ann,r_values

def t_test_analais(r_vals,alpha = 0.05):
    J = len(r_vals)
    npr_vals = np.array(r_vals)
    r_st = np.mean(npr_vals)
    r_std = np.std(npr_vals)

    




print("\n Comparison 1  \n")
# # ANN and lin reg
# Error_test_lin,Error_test_ann,r_values = compare_ann_lin_reg()
#
# print("Compare ANN and lin reg")
# print("ANN results")
# print("Errors: ")
# pprint.pprint(Error_test_ann)
#
# print("Lin reg results")
# print("Errors: ")
# pprint.pprint(Error_test_lin)

print("\n Comparison 2  \n")
# baseline and lin reg
Error_test_lin,Error_test_baseline,r_values = compare_baseline_lin_reg()

print("Compare Baseline and lin reg")
print("Baseline results")
print("Errors: ")
pprint.pprint(Error_test_baseline)

print("Lin reg results")
print("Errors: ")
pprint.pprint(Error_test_lin)

print("11.4.1 analasis")

print("\n Comparison 3  \n")
# baseline and ann
# Error_test_baseline,Error_test_ann,r_values = compare_ann_baseline()
#
# print("Compare ANN and Baseline")
# print("Baseline results")
# print("Errors: ")
# pprint.pprint(Error_test_baseline)
#
# print("ANN results")
# print("Errors: ")
# pprint.pprint(Error_test_ann)


# Latex table
# for index,res in enumerate(zip(Opt_h_ann, Error_test_ann, Opt_lambdas_lin, Error_test_lin, Error_test_baseline)):
#     print(str(index)+" & {0:.3f} & {1:.3f} & {2:.3f} & {3:.3f} & {4:.3f}".format(*[i[0] for i in res])+r" \\")

# Draw neural net and learning curve for last layer

