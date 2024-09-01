import pandas as pd

data = pd.read_csv('../data/ml/hotel_reservations_train.csv')
X = data.drop('booking_status', axis=1)
y = data['booking_status']

