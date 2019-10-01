import numpy as np
import xlrd
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import svd

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

data_frame = pd.read_csv(airbnb_data,dtype=attributes_datatype,parse_dates=attributes_dates)
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


# # filter out all of the data that is of type string (text fields) and are not just id's
# also filter out longitude and latitude
# non_text_fields = (9, 10, 11, 13, 14, 15)
non_text_fields = (9, 10, 14, 15)
selected_atribute_names = [i for j,i in enumerate(prity_atributes) if (j in non_text_fields)]
filtered_data = raw_data[:,non_text_fields]
filtered_attributes = [attributes[i] for i in non_text_fields]

N, M = filtered_data.shape

# plt.plot(filtered_data[:,0],filtered_data[:,1],"o")
# plt.show()

X = np.array(filtered_data)
XX = X.mean(axis=0)
print(XX)
Y = X - np.ones((N,1))*XX
Y = Y.astype(np.float64)
Y2 = Y*(1/np.std(Y,0))


# PCA by computing SVD of Y (without std)
U,S,Vh = svd(Y,full_matrices=False,compute_uv=True)

# PCA with std
U2,S2,Vh2 = svd(Y2,full_matrices=False,compute_uv=True)

pcasum = np.sum(S*S)
# print(pcasum)
for i in S:
    print((i**2/pcasum) * 100)

print()
pcasum2 = np.sum(S2*S2)
# print(pcasum2)
for i in S2:
    print((i**2/pcasum2) * 100)

rho = (S*S) / (S*S).sum()
rho2 = (S2*S2) / (S2*S2).sum()
threshold = 0.90

# Plots of pcs effect on data

# Plot variance explained withotut std
# plt.figure()
# plt.plot(range(1,len(rho)+1),rho,'x-')
# plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
# plt.plot([1,len(rho)],[threshold, threshold],'k--')
# plt.title('Variance explained by principal components');
# plt.xlabel('Principal component');
# plt.ylabel('Variance explained');
# plt.legend(['Individual','Cumulative','Threshold'])
# plt.grid()
# plt.show()

# Plot variance explained with std
# plt.figure()
# plt.plot(range(1,len(rho2)+1),rho2,'x-')
# plt.plot(range(1,len(rho2)+1),np.cumsum(rho2),'o-')
# plt.plot([1,len(rho2)],[threshold, threshold],'k--')
# plt.title('Variance explained by principal components');
# plt.xlabel('Principal component');
# plt.ylabel('Variance explained');
# plt.legend(['Individual','Cumulative','Threshold'])
# plt.grid()
# plt.show()


V = Vh.T
Z = Y @ V

V2 = Vh2.T
Z2 = Y2 @ V2

# Get all unique neighbourhoods
unique_neighbourhoods = data_frame['neighbourhood_group'].unique()

unique_roomtypes = data_frame['room_type'].unique()
print(unique_roomtypes)



y_nbh = list()
y_rty = list()
for listing in raw_data:
    for i,j in enumerate(unique_neighbourhoods):
        if listing[4] == j:
            y_nbh.append(i)
    for i,j in enumerate(unique_roomtypes):
        if listing[8] == j:
            y_rty.append(i)

y_nbh = np.array(y_nbh)
y_rty = np.array(y_rty)

# Indices of the principal components to be plotted
cp1 = 0
cp2 = 1

# data ploted in pc1 vs pc2 projection

# for c in range(len(unique_neighbourhoods)):
#     # select indices belonging to class c:
#     class_mask = [(y_nbh[i]==c) for i in range(y_nbh.size)]
#     plt.plot(Z[class_mask,cp1], Z[class_mask,cp2], 'o', alpha=.5)
# plt.legend(unique_neighbourhoods)
# plt.title("Neighbourhoods in PC space")
# plt.xlabel('PC{0}'.format(cp1+1))
# plt.ylabel('PC{0}'.format(cp2+1))
# plt.show()

# for c in range(len(unique_roomtypes)):
#     # select indices belonging to class c:
#     class_mask = [(y_rty[i]==c) for i in range(y_rty.size)]
#     plt.plot(Z[class_mask,cp1], Z[class_mask,cp2], 'o', alpha=.5)
# plt.legend(unique_roomtypes)
# plt.title("Room types in PC space")
# plt.xlabel('PC{0}'.format(cp1+1))
# plt.ylabel('PC{0}'.format(cp2+1))
# plt.show()


# for c in range(len(unique_roomtypes)):
#     # select indices belonging to class c:
#     class_mask = [(y_rty[i]==c) for i in range(y_rty.size)]
#     plt.plot(Z2[class_mask,cp1], Z2[class_mask,cp2], 'o', alpha=.5)
# plt.legend(unique_roomtypes)
# plt.title("Room types in PC space")
# plt.xlabel('PC{0}'.format(cp1+1))
# plt.ylabel('PC{0}'.format(cp2+1))
# plt.show()

# for c in range(len(unique_neighbourhoods)):
#     # select indices belonging to class c:
#     class_mask = [(y_nbh[i]==c) for i in range(y_nbh.size)]
#     plt.plot(Z2[class_mask,cp1], Z2[class_mask,cp2], 'o', alpha=.5)
# plt.legend(unique_neighbourhoods)
# plt.title("Neighbourhoods in PC space")
# plt.xlabel('PC{0}'.format(cp1+1))
# plt.ylabel('PC{0}'.format(cp2+1))
# plt.show()


# effects of parameters to top n pca's

# Pcs without std taken into account
# pcs = [0,1,2]
# legendStrs = ['PC'+str(e+1) for e in pcs]
# # c = ['r','g','b']
# bw = .2
# r = np.arange(1,M+1)
# for i in pcs:
#     plt.bar(r+i*bw, V[:,i], width=bw)
#     # print(V[:,i],np.sum(V[:,i]))
# plt.xticks(r+bw, selected_atribute_names)
# plt.xlabel('Attributes')
# plt.ylabel('Component coefficients')
# plt.legend(legendStrs)
# plt.grid()
# plt.title('NanoNose: PCA Component Coefficients')
# plt.show()


pcs = [0,1,2]
legendStrs = ['PC'+str(e+1) for e in pcs]
# c = ['r','g','b']
bw = .2
r = np.arange(1,M+1)
for i in pcs:
    plt.bar(r+i*bw, V2[:,i], width=bw)
    # print(V2[:,i],np.sum(V2[:,i]))
plt.xticks(r+bw, selected_atribute_names)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('NanoNose: PCA Component Coefficients')
plt.show()

print(np.sum(np.abs(V2[:,1])))
