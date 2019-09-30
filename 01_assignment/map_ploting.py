import numpy as np
import xlrd
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import svd

airbnb_data = "../data/AB_NYC_2019.csv"
nyc_image = "../data/New_York_City_.png"

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

# latitude
Y = np.array(data_frame["latitude"])
# longitude
X = np.array(data_frame["longitude"])

print(min(X))


img = plt.imread(nyc_image)
fig, ax = plt.subplots()
ax.imshow(img, extent=[-74.26, -73.7, 40.4, 41])


ax.scatter(X,Y,marker=",")

fig.show()
# ax.savefig("something.svg",dpi=1200)


