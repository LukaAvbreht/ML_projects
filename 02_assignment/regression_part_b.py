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

from regression_part_a import OPT_lambda_part_2, X2, YY, XX, X2_labesls

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
print("\n Part B \n 1) \n")

lambdas = np.power(10., range(-5, 9))
h_vals = np.array(list(range(1, 40, 4)))

print("Lambdas Used:")
pprint.pprint(lambdas)

print("Hidden layers used: ")
pprint.pprint(h_vals)

best_net = None
learning_curve_best = None


def two_level_cross_validation():
    # Model 1:
    # In our case linear model we got from part a

    # Imported from regression part a
    print("Optimal lambda from part 1: ", OPT_lambda_part_2)
    print("Also import data from part 1 of size:", X2.shape, YY.shape)

    N, M = X2.shape
    _, MM = XX.shape
    K = 10
    CV = skmd.KFold(K, shuffle=False)

    Error_train_lin = np.empty((K, 1))
    Error_test_lin = np.empty((K, 1))
    Opt_lambdas_lin = np.empty((K, 1))

    Error_test_baseline = np.empty((K, 1))

    Error_test_ann = np.empty((K, 1))
    Opt_h_ann = np.empty((K, 1))

    Error_train_rlr = np.empty((K, 1))
    Error_test_rlr = np.empty((K, 1))
    Error_train_nofeatures = np.empty((K, 1))
    Error_test_nofeatures = np.empty((K, 1))
    w_rlr = np.empty((M, K))
    mu = np.empty((K, M - 1))
    sigma = np.empty((K, M - 1))
    w_noreg = np.empty((M, K))

    k = 0

    for train_index, test_index in CV.split(X2, YY):
        # Linear regression
        X_train = X2[train_index]
        y_train = YY[train_index]
        X_test = X2[test_index]
        y_test = YY[test_index]
        internal_cross_validation = 10

        y_train = y_train.reshape((y_train.shape[0],))
        y_test = y_test.reshape((y_test.shape[0],))

        X_train = X_train.astype(np.float64)
        y_train = y_train.astype(np.float64)
        X_test = X_test.astype(np.float64)
        y_test = y_test.astype(np.float64)

        opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train,
                                                                                                          y_train,
                                                                                                          lambdas,
                                                                                                          internal_cross_validation)

        Opt_lambdas_lin[k] = opt_lambda

        mu[k, :] = np.mean(X_train[:, 1:], 0)
        sigma[k, :] = np.std(X_train[:, 1:], 0)

        X_train[:, 1:] = (X_train[:, 1:] - mu[k, :]) / sigma[k, :]
        X_test[:, 1:] = (X_test[:, 1:] - mu[k, :]) / sigma[k, :]

        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train

        # Compute mean squared error without using the input data at all
        Error_train_nofeatures[k] = np.square(y_train - y_train.mean()).sum(axis=0) / y_train.shape[0]
        Error_test_nofeatures[k] = np.square(y_test - y_test.mean()).sum(axis=0) / y_test.shape[0]

        # Estimate weights for the optimal value of lambda, on entire training set
        lambdaI = opt_lambda * np.eye(M)
        lambdaI[0, 0] = 0  # Do no regularize the bias term
        w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
        # Compute mean squared error with regularization with optimal lambda
        Error_train_rlr[k] = np.square(y_train - X_train @ w_rlr[:, k]).sum(axis=0) / y_train.shape[0]

        Error_test_rlr[k] = np.square(y_test - X_test @ w_rlr[:, k]).sum(axis=0) / y_test.shape[0]

        # Estimate weights for unregularized linear regression, on entire training set
        w_noreg[:, k] = np.linalg.solve(XtX, Xty).squeeze()
        # Compute mean squared error without regularization
        Error_train_lin[k] = np.square(y_train - X_train @ w_noreg[:, k]).sum(axis=0) / y_train.shape[0]

        # The importatn thing
        Error_test_lin[k] = np.square(y_test - X_test @ w_noreg[:, k]).sum(axis=0) / y_test.shape[0]

        # Baseline

        y_pred = np.mean(y_train)

        Error_test_baseline[k] = np.square(y_test - y_pred).sum(axis=0) / y_test.shape[0]

        # ANN
        # Cast to torch tensors

        # Just to work with all data

        X_train = XX[train_index]
        X_test = XX[test_index]

        X_train = X_train.astype(np.float64)
        X_test = X_test.astype(np.float64)

        y_train = y_train.reshape((y_train.shape[0], 1))
        y_test = y_test.reshape((y_test.shape[0], 1))

        X_train = torch.tensor(X_train, dtype=torch.float)
        y_train = torch.tensor(y_train, dtype=torch.float)
        X_test = torch.tensor(X_test, dtype=torch.float)
        # y_test = torch.tensor(y_test, dtype=torch.uint8)

        best_val = 10 ** 30
        hopt = 0

        for n_hidden_units in h_vals:

            model = lambda: torch.nn.Sequential(
                torch.nn.Linear(MM, n_hidden_units),  # M features to H hiden units
                # 1st transfer function, either Tanh or ReLU:
                torch.nn.ReLU(),
                # torch.nn.Tanh(),
                torch.nn.Linear(n_hidden_units, 1),  # H hidden units to 1 output neuron
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
                                                               X=X_train,
                                                               y=y_train,
                                                               n_replicates=3,
                                                               max_iter=max_iter)

            y_res = net(X_test)

            y_res = y_res.data.numpy()
            # y_test = y_test.data.numpy()

            eval_error = np.square(y_test - y_res).sum(axis=0) / y_test.shape[0]

            if eval_error < best_val:
                hopt = n_hidden_units
                best_val = eval_error

                learning_curve_best = learning_curve
                best_net = net

        Error_test_ann[k] = best_val
        Opt_h_ann[k] = hopt

        k += 1

    return Opt_h_ann, Error_test_ann, Opt_lambdas_lin, Error_test_lin, Error_test_baseline


Opt_h_ann, Error_test_ann, Opt_lambdas_lin, Error_test_lin, Error_test_baseline = two_level_cross_validation()

print("ANN results")
print("H: ")
print(Opt_h_ann)
print("Errors: ")
print(Error_test_ann)

print("Lin reg results")
print("lambdas: ")
print(Opt_lambdas_lin)
print("Errors: ")
print(Error_test_lin)

print("Baseline test")
print("Errors: ")
print(Error_test_baseline)


# Latex table
for index,res in enumerate(zip(Opt_h_ann, Error_test_ann, Opt_lambdas_lin, Error_test_lin, Error_test_baseline)):
    print(str(index)+" & {0:.3f} & {1:.3f} & {2:.3f} & {3:.3f} & {4:.3f}".format(*[i[0] for i in res])+r" \\")

# Draw neural net and learning curve for last layer
weights = [best_net[i].weight.data.numpy().T for i in [0,2]]
biases = [best_net[i].bias.data.numpy() for i in [0,2]]
tf = [str(best_net[i]) for i in [1,3]]
draw_neural_net(weights, biases, tf, attribute_names=X2_labesls)


# Plot learning curves
plt.plot(learning_curve_best)
plt.ylabel("Error")
plt.xlabel('Iterations')
plt.xlim((0, 10000))
plt.ylabel('Loss')
plt.title('Learning curves')
plt.show()

