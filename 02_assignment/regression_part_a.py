import numpy as np
import xlrd
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import svd
from categoric2numeric import categoric2numeric
from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show
import sklearn.linear_model as lm
import sklearn.model_selection as skmd
from toolbox.Toolbox_Python02450.Tools.toolbox_02450 import feature_selector_lr, bmplot, rlr_validate
from matplotlib.pyplot import figure, plot, xlabel, ylabel, clim, semilogx, loglog, legend, title, subplot, show, grid
import pprint
import random

random.seed = 1337

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

# print(unique_neighbourhoods)
print(unique_roomtypes)
print(unique_boroughs)

# We can see that there are many neighbourhoods within nyc, where airbnb offers accommodation
# We can use this to predict stuff within boroughs

# -- Regression PART A  --
# -- 1)     --
print("\n Part A \n 1) \n")
# Predict price based on borough, room type, minimum nights and availability

# lets first of all do one-out-of-K encoding for borough and room type, and extract data we wanna use

# Get prices of rooms and create np array out of it
result_atributes = (9)
result_data = raw_data[:, result_atributes]
Y = np.array(result_data).T
Y = Y.reshape((Y.shape[0], 1))
print(Y.shape)

# Standarize our data matrix
# One out K for nbh
nbh_data = raw_data[:, (4)]
x_nbh = np.array(nbh_data).T
X_K1, K1_labels = categoric2numeric(x_nbh)

roomtype_data = raw_data[:, (8)]
x_rty = np.array(roomtype_data).T
X_K2, K2_labels = categoric2numeric(x_rty)

# Get other parameters and standardise them
other_params = (10, 15)
other_data = np.array(raw_data[:, other_params])

# Shape of
N, M = other_data.shape

# To get a shape of (n,1) to use in concatenate (only if we only use one additional parameter
if N == 1:
    other_data = other_data.reshape((other_data.shape[0], 1))

other_data = other_data - np.ones((N, 1)) * other_data.mean(axis=0)
other_data = other_data.astype(np.float64)
other_data = other_data * (1 / np.std(other_data, 0))

# Concatenate all of the data int one matrix

X = np.concatenate((X_K1, X_K2, other_data), axis=1)
X_labesls = K1_labels + K2_labels + [attributes[i] for i in other_params]

print(X.shape)
print(X_labesls)

# DO linear regression on it

# Fit ordinary least squares regression model (data is not centered)
model = lm.LinearRegression(fit_intercept=True)
model = model.fit(X, Y)
print("Parameters of network: ", model.get_params())
print("Coeficients of linear moddel: ", model.coef_)

# Compute model output:
y_est = model.predict(X)

# # Plot original data and the model output
# f = figure()
# plot(X,Y,'.')
# # plot(X,y_est,'-')
# xlabel('X'); ylabel('y')
# legend(['Training data', 'Regression fit (model)'])
#
# show()

# We can see that plotting linear regerssion model for this case makes almost 0 sense.
# Since we use 10 parameters and therefore cant plot this into plane

# Additional thing:

print("\n Part1 ++ \n")

raw_dataplus = data_frame_original.get_values()
all_but_price = (10, 11, 13, 14, 15)

# Get prices of rooms and create np array out of it
result_atributes = (9)
result_data = raw_dataplus[:, result_atributes]
YY = np.array(result_data).T
YY = YY.reshape((YY.shape[0], 1))

# Standarize our data matrix
# One out K for nbh
nbh_data = raw_dataplus[:, (4)]
x_nbh = np.array(nbh_data).T
X_K1, K1_labels = categoric2numeric(x_nbh)

roomtype_data = raw_dataplus[:, (8)]
x_rty = np.array(roomtype_data).T
X_K2, K2_labels = categoric2numeric(x_rty)

# Get other parameters and standardise them
other_data = np.array(raw_dataplus[:, all_but_price])

XX = np.concatenate((X_K1, X_K2, other_data), axis=1)
XX_labesls = K1_labels + K2_labels + [prity_atributes[i] for i in all_but_price]

print(XX_labesls)
print(XX.shape)

N, M = XX.shape
K = 8
Kfold = skmd.KFold(n_splits=K, shuffle=True)

Features = np.zeros((M, K))
Error_train = np.empty((K, 1))
Error_test = np.empty((K, 1))
Error_train_fs = np.empty((K, 1))
Error_test_fs = np.empty((K, 1))
Error_train_nofeatures = np.empty((K, 1))
Error_test_nofeatures = np.empty((K, 1))

k = 0
for train_index, test_index in Kfold.split(X):

    # extract training and test set for current CV fold
    X_train = XX[train_index, :]
    y_train = YY[train_index]
    X_test = XX[test_index, :]
    y_test = YY[test_index]
    internal_cross_validation = 10

    # Compute squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train - y_train.mean()).sum() / y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test - y_test.mean()).sum() / y_test.shape[0]

    # Compute squared error with all features selected (no feature selection)
    m = lm.LinearRegression(fit_intercept=True).fit(X_train, y_train)
    Error_train[k] = np.square(y_train - m.predict(X_train)).sum() / y_train.shape[0]
    Error_test[k] = np.square(y_test - m.predict(X_test)).sum() / y_test.shape[0]


    # Compute squared error with feature subset selection
    # textout = 'verbose';
    textout = '';
    selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train, internal_cross_validation,
                                                                          display=textout)

    Features[selected_features, k] = 1
    # .. alternatively you could use module sklearn.feature_selection
    if len(selected_features) is 0:
        print('No features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).')
    else:
        m = lm.LinearRegression(fit_intercept=True).fit(X_train[:, selected_features], y_train)
        Error_train_fs[k] = np.square(y_train - m.predict(X_train[:, selected_features])).sum() / y_train.shape[0]
        Error_test_fs[k] = np.square(y_test - m.predict(X_test[:, selected_features])).sum() / y_test.shape[0]

        # figure(k)
        # subplot(1, 2, 1)
        # plot(range(1, len(loss_record)), loss_record[1:])
        # xlabel('Iteration')
        # ylabel('Squared error (crossvalidation)')
        #
        # subplot(1, 3, 3)
        # bmplot(XX_labesls, range(1, features_record.shape[1]), -features_record[:, 1:])
        # clim(-1.5, 0)
        # xlabel('Iteration')

    print('Cross validation fold {0}/{1}'.format(k + 1, K))
    # print('Train indices: {0}'.format(train_index))
    # print('Test indices: {0}'.format(test_index))
    print('Features no: {0}\n'.format(selected_features.size))

    k += 1

# Display results
print('\n')
print('Linear regression without feature selection:\n')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum() - Error_train.sum()) / Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum() - Error_test.sum()) / Error_test_nofeatures.sum()))
print('\nLinear regression with feature selection:\n')
print('- Training error: {0}'.format(Error_train_fs.mean()))
print('- Test error:     {0}'.format(Error_test_fs.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum() - Error_train_fs.sum()) / Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum() - Error_test_fs.sum()) / Error_test_nofeatures.sum()))

figure(k)
subplot(1, 3, 2)
bmplot(XX_labesls, range(1, Features.shape[1] + 1), -Features)
clim(-1.5, 0)
xlabel('Crossvalidation fold')
ylabel('Attribute')

# Inspect selected feature coefficients effect on the entire dataset and
# plot the fitted model residual error as function of each attribute to
# inspect for systematic structure in the residual

f = 2  # cross-validation fold to inspect
ff = Features[:, f - 1].nonzero()[0]
# if len(ff) is 0:
#     print('\nNo features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).')
# else:

m_fit = lm.LinearRegression(fit_intercept=True).fit(XX[:, ff], YY)

y_est = m_fit.predict(XX[:, ff])
residual = YY - y_est

figure(k + 1, figsize=(12, 6))
title('Residual error vs. Attributes for features selected in cross-validation fold {0}'.format(f))
for i in range(0, len(ff)):
    subplot(2, np.ceil(len(ff) / 2.0), i + 1)
    plot(XX[:, ff[i]], residual, '.')
    xlabel(XX_labesls[ff[i]])
    ylabel('residual error')

show()

print("The most fit model according to feature selection (")
print("It uses:")
pprint.pprint([XX_labesls[i] for i in selected_features])
print("In linear regression")

# -- Regression PART A  --
# -- 2)     --
print("\n Part A \n 2) \n")

# Introduce regularization parameter lambda
# lambs = [0.001*i for i in range(25000)]
# lambs = [0., 0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]

# Ratio between train and test data as test/(test+train) data
# My testing
# test_train_ratio = 0.1
#
# nums = [False for _ in range(int(X.shape[0]*test_train_ratio))] + [True for _ in range(int(X.shape[0]*test_train_ratio),X.shape[0])]
# random.shuffle(nums)
#
# nnums = [1-i for i in nums]
#
# X_train = X[nums,:]
# X_test = X[nnums,:]
# Y_train = Y[nums,:]
# Y_test = Y[nnums,:]
#
# for lam in lambs:
#     model = lm.Ridge(alpha=lam, fit_intercept=True)
#     model = model.fit(X_train, Y_train)

# THIS THING
# general_errors = list()
# kf = skmd.KFold(n_splits=10)
#
# # grnorm = 1/10000
#
# for lam in lambs:
#     grn_error = 0
#     for train_index, test_index in kf.split(X):
#         # print("TRAIN:", train_index, "TEST:", test_index)
#         X_train, X_test = X[train_index], X[test_index]
#         Y_train, Y_test = Y[train_index], Y[test_index]
#         model = lm.Ridge(alpha=lam, fit_intercept=True)
#         model = model.fit(X_train, Y_train)
#
#         y_predicts = model.predict(X_test)
#
#         grn_error += (sum(abs(i[0]-j[0]) for i,j in zip(y_predicts, Y_test))/y_predicts.shape[0])
#     general_errors.append(grn_error)
#
#
# print(lambs)
# print(general_errors)
# plot(lambs,general_errors)
# xlabel('lambdas'); ylabel('generalized errors')
#
# show()

# Only take fetures we got from part a
print("Selected features: ",selected_features)
# We take only the data of that features
XXX = XX[:,selected_features]

X2 = np.concatenate((np.ones((XXX.shape[0],1)),XXX),1)
X2_labesls = [u'Offset']+[i for j,i in enumerate(XX_labesls) if j in selected_features]

## Crossvalidation
# Create crossvalidation partition for evaluation
N, M = X2.shape
K = 5
# CV = skmd.KFold(K, shuffle=True)
CV = skmd.KFold(K, shuffle=False)

# Values of lambda
lambdas = np.power(10.,range(-5,9))

# Initialize variables
#T = len(lambdas)
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))

# TO export to part 2
OPT_lambda_part_2 = None


k = 0
for train_index, test_index in CV.split(X2, YY):

    # extract training and test set for current CV fold
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

    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train,
                                                                                                      lambdas,
                                                                                                      internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
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
    Error_train[k] = np.square(y_train - X_train @ w_noreg[:, k]).sum(axis=0) / y_train.shape[0]
    Error_test[k] = np.square(y_test - X_test @ w_noreg[:, k]).sum(axis=0) / y_test.shape[0]
    # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
    # m = lm.LinearRegression().fit(X_train, y_train)
    # Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    # Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Display the results for the last cross-validation fold
    if k == K - 1:
        figure(k, figsize=(12, 8))
        subplot(1, 2, 1)
        semilogx(lambdas, mean_w_vs_lambda.T[:, 1:], '.-')  # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()
        # You can choose to display the legend, but it's omitted for a cleaner
        # plot, since there are many attributes
        legend(X2_labesls[1:], loc='best')

        OPT_lambda_part_2 = opt_lambda

        subplot(1, 2, 2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(lambdas, train_err_vs_lambda.T, 'b.-', lambdas, test_err_vs_lambda.T, 'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error', 'Validation error'])
        grid()

        print("Coefficinet values represent values of each w (for each parameter we use in regression)")

    # To inspect the used indices, use these print statements
    # print('Cross validation fold {0}/{1}:'.format(k+1,K))
    # print('Train indices: {0}'.format(train_index))
    # print('Test indices: {0}\n'.format(test_index))

    k += 1

show()
# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum() - Error_train.sum()) / Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum() - Error_test.sum()) / Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format(
    (Error_train_nofeatures.sum() - Error_train_rlr.sum()) / Error_train_nofeatures.sum()))
print(
    '- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum() - Error_test_rlr.sum()) / Error_test_nofeatures.sum()))

print('Weights in last fold:')
for m in range(M):
    print('{:>15} {:>15}'.format(X2_labesls[m], np.round(w_rlr[m,-1],2)))

print("Optimal lambda: ",OPT_lambda_part_2)

# -- Regression PART A --
# -- 3)     --
print("\n Part A \n 3) \n")

# Explain how new data is predicted

# It is calculated based on the linear function wiht coefficients W_i i = 1,...,10 in our case (since we use 10
# columns for linear regression)
# This coefficients are:

# Used in last moddel (The optimal one)

# Weights in last fold:
#          Offset          148.66
#           Bronx           -6.19
#        Brooklyn           -8.55
#       Manhattan           17.11
#          Queens          -10.17
#   Staten Island           -3.01
# Entire home/apt           28.23
#    Private room          -25.28
#     Shared room          -10.65
#  minimum nights           -2.09
#   review number            -7.5
#   rev per month           -8.68
# host listing count           -2.25
#    availability           21.75

print("Take a look at code")

