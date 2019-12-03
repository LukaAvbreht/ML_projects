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

# result_atributes = (8)
# result_data = raw_data[:, result_atributes]
# result_data = [rtype_dict[i] for i in result_data]
# # print(result_data)
# Y = np.array(result_data).T
# yy_data = list()
# for i in result_data:
#     if i == 0:
#         yy_data.append(np.array([1,0,0]))
#     if i == 1:
#         yy_data.append(np.array([0,1,0]))
#     if i == 2:
#         yy_data.append(np.array([0,0,1]))
# YY = np.array(yy_data)
# Y = Y.reshape((Y.shape[0], 1))

# Standarize our data matrix
# One out K for nbh
nbh_data = raw_data[:, (4)]
x_nbh = np.array(nbh_data).T
X_K1, K1_labels = categoric2numeric(x_nbh)

# one ot of K for room_type
roomtype_data = raw_data[:, (8)]
x_rty = np.array(roomtype_data).T
X_K2, K2_labels = categoric2numeric(x_rty)


# Get other parameters and standardise them
other_params = (9,10,11,12,13,14, 15)
other_data = np.array(raw_data[:, other_params])
ogiginal_vals = np.array(raw_data[:, other_params])

N, M = other_data.shape

# To get a shape of (n,1) to use in concatenate (only if we only use one additional parameter
if N == 1:
    other_data = other_data.reshape((other_data.shape[0], 1))

# other_data = other_data - np.ones((N, 1)) * other_data.mean(axis=0)
# other_data = other_data.astype(np.float64)
# other_data = other_data * (1 / np.std(other_data, 0))

# Concatenate all of the data int one matrix

X = np.concatenate((X_K1, other_data), axis=1)
X_labesls = K1_labels + [attributes[i] for i in other_params]

# Limit to roomtype that is

# X(YY == np.array([1,0,0]))
rome_type_1_index = list(map(all,YY == np.array([1,0,0])))

X = X[rome_type_1_index,:]
