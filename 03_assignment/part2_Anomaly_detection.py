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
from matplotlib.pyplot import figure, plot, xlabel, ylabel, clim, semilogx, loglog, legend, title, subplot, show, grid, bar, xticks, yticks, cm, imshow
import pprint

from toolbox.Toolbox_Python02450.Tools.toolbox_02450 import gausKernelDensity
from sklearn.neighbors import NearestNeighbors

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
data_frame = data_frame_original.sample(frac=0.3)

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
ogiginal_vals = np.array(raw_data[:, other_params])

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

# Limit to roomtype that is

print(X.shape)

# X(YY == np.array([1,0,0]))
rome_type_1_index = list(map(all,YY == np.array([1,0,0])))

X = X[rome_type_1_index,:]

print(X.shape)

## part 1 (a)
##
############################

N, M = np.shape(X)

### Gausian Kernel density estimator
# cross-validate kernel width by leave-one-out-cross-validation
# (efficient implementation in gausKernelDensity function)
# evaluate for range of kernel widths
widths = X.var(axis=0).max() * (2.0 ** np.arange(-10, 3))
logP = np.zeros(np.size(widths))
for i, w in enumerate(widths):
    print('Fold {:2d}, w={:f}'.format(i, w))
    density, log_density = gausKernelDensity(X, w)
    logP[i] = log_density.sum()

val = logP.max()
ind = logP.argmax()

width = widths[ind]
print('Optimal estimated width is: {0}'.format(width))

# evaluate density for estimated width
density, log_density = gausKernelDensity(X, width)

# Sort the densities
i = (density.argsort(axis=0)).ravel()
density = density[i].reshape(-1, )

gauss_i = i[:]

# Plot density estimate of outlier score
figure(1)
bar(range(20), density[:20])
title('Gaussian Kernel Density')
show()
# Plot possible outliers
# figure(2)
# for k in range(1, 21):
#     subplot(4, 5, k)
#     imshow(np.reshape(X[i[k], :], (16, 16)).T, cmap=cm.binary)
#     xticks([]);
#     yticks([])
#     if k == 3: title('Gaussian Kernel Density: Possible outliers')

# Get possible outliers!
print("Gaussian Kernel Density: Possible outliers")
pprint.pprint([X[i[k], :] for k in range(1,21)])


### K-neighbors density estimator
# Neighbor to use:
K = 5

# Find the k nearest neighbors
knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)

density = 1. / (D.sum(axis=1) / K)

# Sort the scores
i = density.argsort()
density = density[i]

KNN_i = i[:]

# Plot k-neighbor estimate of outlier score (distances)
figure(2)
bar(range(20), density[:20])
title('KNN density: Outlier score')
show()
# Plot possible outliers
# figure(4)
# for k in range(1, 21):
#     subplot(4, 5, k)
#     imshow(np.reshape(X[i[k], :], (16, 16)).T, cmap=cm.binary)
#     xticks([]);
#     yticks([])
#     if k == 3: title('KNN density: Possible outliers')
print("KNN density: Possible outliers")
pprint.pprint([X[i[k], :] for k in range(1,21)])
### K-nearest neigbor average relative density
# Compute the average relative density

knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)
print(K)
print(D.sum(axis=1))

density = 1. / ((D.sum(axis=1)+abs(np.random.normal(0,0.00001,D.shape[0]))) / K)
avg_rel_density = density / (density[i[:, 1:]].sum(axis=1) / K)

# Sort the avg.rel.densities
i_avg_rel = avg_rel_density.argsort()
avg_rel_density = avg_rel_density[i_avg_rel]

# Plot k-neighbor estimate of outlier score (distances)
figure(3)
bar(range(20), avg_rel_density[:20])
title('KNN average relative density: Outlier score')
show()
# Plot possible outliers
# figure(6)
# for k in range(1, 21):
#     subplot(4, 5, k)
#     imshow(np.reshape(X[i_avg_rel[k], :], (16, 16)).T, cmap=cm.binary)
#     xticks([]);
#     yticks([])
#     if k == 3: title('KNN average relative density: Possible outliers')

print("KNN average relative density: Possible outliers")
pprint.pprint([X[i_avg_rel[k], :] for k in range(1,21)])


### Distance to 5'th nearest neighbor outlier score
K = 5

# Find the k nearest neighbors
knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)

# Outlier score
score = D[:, K - 1]
# Sort the scores
i = score.argsort()
score = score[i[::-1]]

# Plot k-neighbor estimate of outlier score (distances)
figure(4)
bar(range(20), score[:20])
title('5th neighbor distance: Outlier score')
show()
# Plot possible outliers
# figure(8)
# for k in range(1, 21):
#     subplot(4, 5, k)
#     imshow(np.reshape(X[i[k], :], (16, 16)).T, cmap=cm.binary);
#     xticks([]);
#     yticks([])
#     if k == 3: title('5th neighbor distance: Possible outliers')
# show()

print("5th neighbor distance: Possible outliers")
pprint.pprint([X[i[k], :] for k in range(1,21)])

# get table for price outliers

for ind in range(10):
    print("{0} & {1:.0f} & {2:.0f} & {3:.0f} & {4:.0f}".format(ind+1,
                                                               ogiginal_vals[gauss_i[ind]][0],
                                                               ogiginal_vals[KNN_i[ind]][0],
                                                               ogiginal_vals[i_avg_rel[ind]][0],
                                                               ogiginal_vals[i[ind]][0]
                                                               ),r"\\ \hline")

# manhatan data conut

print("Scoring method  & "," & ".join(K1_labels),r"\\ \hline")

score = [[0 for _ in range(len(K1_labels))] for _ in range(4)]
for ind in range(200):
    for eni,method in enumerate([gauss_i,KNN_i,i_avg_rel,i]):
        val = X[method[ind],:]
        if val[0] == 1.:
            score[eni][0] += 1
        elif val[1] == 1.:
            score[eni][1] += 1
        elif val[2] == 1.:
            score[eni][2] += 1
        elif val[3] == 1.:
            score[eni][3] += 1
        elif val[4] == 1.:
            score[eni][4] += 1

pprty_met_name = [
    "Gaussian Kernel Density",
    "KNN Density",
    "KNN Relative Density",
    "5'th Neighbourhood"
]
for eni,line in enumerate(score):
    print(pprty_met_name[eni]+" & ", " & ".join(map(str,line)), r"\\ \hline")

