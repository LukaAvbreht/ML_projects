import numpy as np
import xlrd
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import svd
from categoric2numeric import categoric2numeric

airbnb_data = "../data/AB_NYC_2019.csv"

attributes_datatype = {
    'id': np.float64,   # 0
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

data_frame = pd.read_csv(airbnb_data, dtype=attributes_datatype, parse_dates=attributes_dates)
data_frame.fillna(0, inplace=True)

raw_data = data_frame.get_values()
attributes = list(data_frame.columns)

print(attributes)

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

# -- PART 1 --
# -- A)     --

# Predict price based on borough, room type, minimum nights and availability

# lets first of all do one-out-of-K encoding for borough and room type, and extract data we wanna use

# Get prices of rooms and create np array out of it
result_atributes = (9)
result_data = raw_data[:,result_atributes]
Y = np.array(result_data).T
Y = Y.reshape((Y.shape[0], 1))
print(Y.shape)

# Standarize our data matrix
# One out K for nbh
nbh_data = raw_data[:,(4)]
x_nbh = np.array(nbh_data).T
X_K1,K1_labels = categoric2numeric(x_nbh)

roomtype_data = raw_data[:,(8)]
x_rty = np.array(roomtype_data).T
X_K2,K2_labels = categoric2numeric(x_rty)


# Get other parameters and standardise them
other_params = (10,15)
other_data = np.array(raw_data[:,other_params])

# Shape of
N,M = other_data.shape

# To get a shape of (n,1) to use in concatenate (only if we only use one additional parameter
if N == 1:
    other_data = other_data.reshape((other_data.shape[0], 1))

other_data = other_data - np.ones((N,1)) * other_data.mean(axis=0)
other_data = other_data.astype(np.float64)
other_data = other_data*(1/np.std(other_data,0))


# Concatenate all of the data int one matrix

X = np.concatenate((X_K1,X_K2,other_data),axis=1)
X_labesls = K1_labels + K2_labels + [attributes[i] for i in other_params]

print(X.shape)
print(X_labesls)

print(X[:5])


# -- PART 1 --
# -- B)     --


