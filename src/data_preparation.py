### Description
# This script performs data preparation on the hotel reservations dataset.
# The following steps are performed:
# 1. Load the dataset
# 2. Drop the 'Booking_ID' column
# 3. Perform one-hot encoding on the 'type_of_meal_plan' column
# 4. Perform one-hot encoding on the 'room_type_reserved' column
# 5. Perform one-hot encoding on the 'market_segment_type' column
# 6. Encode the 'booking_status' column
# 7. Remove rows where 'type_of_meal_plan' is 'Meal Plan 3'
# 8. Remove rows where 'room_type_reserved' is 'Room_Type 3'
# 9. Remove rows where 'market_segment_type' is 'Aviation' or 'Complementary' and join them into a new category called 'Aviation_Funded'
# 10. Remove rows where 'no_of_children' is 9 or 10
# 11. Swap the values of 'no_of_adults' and 'no_of_children' columns where 'no_of_adults' is 0 and 'no_of_children' is not 0
# 12. Convert the 'no_of_children' column to a binary column 'with_children' where 0 means no children and 1 means at least 1 child
# 13. Drop the records where the 'no_of_weekend_nights' is 0 and the 'no_of_week_nights' is also 0
# 14. Create categories for the 'no_of_weekend_nights' column
# 15. Encode the 'no_of_weekend_nights' column
# 16. Convert the 'no_of_week_nights' column to category data type
# 17. Apply categories to the 'no_of_week_nights' column
# 18. Encode the 'no_of_week_nights' column
# 19. Drop the 'required_car_parking_space' column
# 20. Drop the 'repeated_guest' column
# 21. Drop the 'no_of_previous_cancellations' column
# 22. Drop the 'no_of_previous_bookings_not_canceled' column
# 23. Filter the data to select rows where the average price per room is zero
# 24. Drop the rows where the average price per room is zero and the market segment type is not Online, Offline, or Corporate
# 25. Drop records with average price over 500
# 27. Group the 'no_of_special_requests' column into categories
# 28. Convert the 'no_of_special_requests' column to category data type
# 29. Encode the 'no_of_special_requests' column
# 30. Remove outliers using Isolation Forest
# 31. Get indices where the arrival_date is 29 and arrival_month is 2
# 32. Drop the rows where the arrival_date is 29 and the arrival_month is 2
# 33. Save the processed data
# The processed data is saved in the 'data/processed' directory as 'hotel_reservations.csv'
# The script can be executed from the command line using the following command:
# python src/data_preparation.py
# Copyright (c) 2024
# Uvod u nauku o podacima, Prirodno-matematicki fakultet, Univerzitet u Kragujevcu
# Authors: Radovan Draskovic, Marija Jolovic
# Project: Predstavljanje i tumacenje skupa podataka Hotel Reservations

import pandas as pd
import numpy as np

RAW_FILE_PATH = '../data/raw/hotel_reservations.csv'
PROCESSED_FILE_PATH = '../data/processed/hotel_reservations.csv'
LINE_SEPARATOR = '-' * 80
NEW_LINE = '\n'

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def save_data(data: pd.DataFrame, file_path: str) -> None:
    data.to_csv(file_path, index=False)
    
def perform_one_hot_encoding(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    return pd.get_dummies(data, columns=columns, drop_first=True, dtype=np.int64)

def encoding_categorical_features(data: pd.DataFrame) -> pd.DataFrame:
    # Delete rows with type_of_meal_plan = 'Meal Plan 3'
    indices_meal_plan_3 = data[data['type_of_meal_plan'] == 'Meal Plan 3'].index
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    print("\033[1;32mDeleting rows with 'type_of_meal_plan' = 'Meal Plan 3'...\033[0m")
    print("\033[1;32mData shape before deleting rows with 'type_of_meal_plan' = 'Meal Plan 3': {}\033[0m".format(data.shape))
    data = data.drop(indices_meal_plan_3)
    print("\033[1;32mRows with 'type_of_meal_plan' = 'Meal Plan 3' deleted successfully!\033[0m")
    print("\033[1;32mData shape after deleting rows with 'type_of_meal_plan' = 'Meal Plan 3': {}\033[0m".format(data.shape))
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    
    # Encode 'type_of_meal_plan' column
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    print("\033[1;32mPerforming one-hot encoding on 'type_of_meal_plan' column...\033[0m")
    print("\033[1;32mData shape before one-hot encoding: {}\033[0m".format(data.shape))
    data['type_of_meal_plan'] = data['type_of_meal_plan'].astype('category')
    data = perform_one_hot_encoding(data, ['type_of_meal_plan'])
    print("\033[1;32mOne-hot encoding on 'type_of_meal_plan' column performed successfully!\033[0m")
    print("\033[1;32mData shape after one-hot encoding: {}\033[0m".format(data.shape))
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    
    # Delete rows where 'room_type_reserved' = 'Room_Type 3'
    indices_room_type_3 = data[data['room_type_reserved'] == 'Room_Type 3'].index
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    print("\033[1;32mDeleting rows with 'room_type_reserved' = 'Room_Type 3'...\033[0m")
    print("\033[1;32mData shape before deleting rows with 'room_type_reserved' = 'Room_Type 3': {}\033[0m".format(data.shape))
    data = data.drop(indices_room_type_3)
    print("\033[1;32mRows with 'room_type_reserved' = 'Room_Type 3' deleted successfully!\033[0m")
    print("\033[1;32mData shape after deleting rows with 'room_type_reserved' = 'Room_Type 3': {}\033[0m".format(data.shape))
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    
    # Encode 'room_type_reserved' column
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    print("\033[1;32mPerforming one-hot encoding on 'room_type_reserved' column...\033[0m")
    print("\033[1;32mData shape before one-hot encoding: {}\033[0m".format(data.shape))
    data['room_type_reserved'] = data['room_type_reserved'].astype('category')
    data = perform_one_hot_encoding(data, ['room_type_reserved'])
    print("\033[1;32mOne-hot encoding on 'room_type_reserved' column performed successfully!\033[0m")
    print("\033[1;32mData shape after one-hot encoding: {}\033[0m".format(data.shape))
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    
    print("Number of rows where market_segment_type is Aviation: ", len(data[data['market_segment_type'] == 'Aviation']))
    print("Number of rows where market_segment_type is Complementary: ", len(data[data['market_segment_type'] == 'Complementary']))

    # Join Aviation and Complementary into new category called Aviation_Funded
    data['market_segment_type'] = data['market_segment_type'].replace(['Aviation', 'Complementary'], 'Aviation_Funded')

    print("Number of rows where market_segment_type is Aviation_Funded: ", len(data[data['market_segment_type'] == 'Aviation_Funded']))

    # Encode 'market_segment_type' column
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    print("\033[1;32mPerforming one-hot encoding on 'market_segment_type' column...\033[0m")
    print("\033[1;32mData shape before one-hot encoding: {}\033[0m".format(data.shape))
    data['market_segment_type'] = data['market_segment_type'].astype('category')
    data = perform_one_hot_encoding(data, ['market_segment_type'])
    print("\033[1;32mOne-hot encoding on 'market_segment_type' column performed successfully!\033[0m")
    print("\033[1;32mData shape after one-hot encoding: {}\033[0m".format(data.shape))
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    
    # Encode 'booking_status' column
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    print("\033[1;32mEncoding 'booking_status' column...\033[0m")
    data['booking_status'] = data['booking_status'].apply(lambda x: 1 if x == 'Canceled' else 0)
    print("\033[1;32m'booking_status' column encoded successfully!\033[0m")
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    
    return data

def remove_outliers_iso_forest(data: pd.DataFrame) -> pd.DataFrame:
    from sklearn.ensemble import IsolationForest
    
    # Remove outliers using Isolation Forest
    iso_forest = IsolationForest(contamination=0.075, random_state=0)
    outliers = iso_forest.fit_predict(data.drop(columns=['booking_status'], axis=1))
    
    outlier_indices = data.index[outliers == -1]
    data = data.drop(outlier_indices)
    
    return data

def annomalies_detection_and_removal(data: pd.DataFrame) -> pd.DataFrame:
    # Removing rows where number of children is 9 or 10
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    print("\033[1;32mRemoving rows where number of children is 9 or 10...\033[0m")
    index_children_9_10 = data[(data['no_of_children'] == 9) | (data['no_of_children'] == 10)].index
    print("\033[1;32mData shape before removing rows where number of children is 9 or 10: {}\033[0m".format(data.shape))
    data = data.drop(index_children_9_10)
    print("\033[1;32mRows where number of children is 9 or 10 removed successfully!\033[0m")
    print("\033[1;32mData shape after removing rows where number of children is 9 or 10: {}\033[0m".format(data.shape))
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    
    # Swap the values of 'no_of_adults' and 'no_of_children' columns where 'no_of_adults' is 0 and 'no_of_children' is not 0
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    print("\033[1;32mSwapping the values of 'no_of_adults' and 'no_of_children' columns where 'no_of_adults' is 0 and 'no_of_children' is not 0...\033[0m")
    index_adults_0_children_not_0 = data[(data['no_of_adults'] == 0) & (data['no_of_children'] != 0)].index
    print("\033[1;32mData shape before swapping the values of 'no_of_adults' and 'no_of_children' columns: {}\033[0m".format(data.shape))
    data.loc[index_adults_0_children_not_0, ['no_of_adults', 'no_of_children']] = data.loc[index_adults_0_children_not_0, ['no_of_children', 'no_of_adults']].values
    print("After check: {}".format(data[(data['no_of_adults'] == 0) & (data['no_of_children'] != 0)].shape))
    print("\033[1;32mValues of 'no_of_adults' and 'no_of_children' columns swapped successfully!\033[0m")
    print("\033[1;32mData shape after swapping the values of 'no_of_adults' and 'no_of_children' columns: {}\033[0m".format(data.shape))
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    
    # Convert the 'no_of_children' column to a binary column 'with_children' where 0 means no children and 1 means at least 1 child
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    print("\033[1;32mConverting the 'no_of_children' column to a binary column 'with_children' where 0 means no children and 1 means at least 1 child...\033[0m")
    print("\033[1;32mData shape before converting the 'no_of_children' column to 'with_children' column: {}\033[0m".format(data.shape))
    data['with_children'] = data['no_of_children'].apply(lambda x: 1 if x > 0 else 0)
    data = data.drop('no_of_children', axis=1)
    print("\033[1;32m'no_of_children' column converted to 'with_children' column successfully!\033[0m")
    print("\033[1;32mData shape after converting the 'no_of_children' column to 'with_children' column: {}\033[0m".format(data.shape))
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    
    # Drop the records where the 'no_of_weekend_nights' is 0 and the 'no_of_week_nights' is also 0
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    print("\033[1;32mDropping the records where the 'no_of_weekend_nights' is 0 and the 'no_of_week_nights' is also 0...\033[0m")
    index_weekend_nights_0_week_nights_0 = data[(data['no_of_weekend_nights'] == 0) & (data['no_of_week_nights'] == 0)].index
    print("\033[1;32mData shape before dropping the records where the 'no_of_weekend_nights' is 0 and the 'no_of_week_nights' is also 0: {}\033[0m".format(data.shape))
    data = data.drop(index_weekend_nights_0_week_nights_0)
    print("\033[1;32mRecords where the 'no_of_weekend_nights' is 0 and the 'no_of_week_nights' is also 0 dropped successfully!\033[0m")
    print("\033[1;32mData shape after dropping the records where the 'no_of_weekend_nights' is 0 and the 'no_of_week_nights' is also 0: {}\033[0m".format(data.shape))
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    
    # Create categories for the 'no_of_weekend_nights' column as follows
    # 0: 0 nights
    # 1: 1 night
    # 2: 2 nights
    # 3+: 3 or more nights
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    print("\033[1;32mCreating categories for the 'no_of_weekend_nights' column...\033[0m")
    print("\033[1;32mData shape before creating categories for the 'no_of_weekend_nights' column: {}\033[0m".format(data.shape))
    data['no_of_weekend_nights'] = data['no_of_weekend_nights'].apply(lambda x: '0' if x == 0 else '1' if x == 1 else '2' if x == 2 else '3+')
    print("\033[1;32mCategories for the 'no_of_weekend_nights' column created successfully!\033[0m")
    print("\033[1;32mData shape after creating categories for the 'no_of_weekend_nights' column: {}\033[0m".format(data.shape))
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    
    # Encode 'no_of_weekend_nights' column
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    print("\033[1;32mPerforming one-hot encoding on 'no_of_weekend_nights' column...\033[0m")
    print("\033[1;32mData shape before one-hot encoding: {}\033[0m".format(data.shape))
    data['no_of_weekend_nights'] = data['no_of_weekend_nights'].astype('category')
    data = perform_one_hot_encoding(data, ['no_of_weekend_nights'])
    print("\033[1;32mOne-hot encoding on 'no_of_weekend_nights' column performed successfully!\033[0m")
    print("\033[1;32mData shape after one-hot encoding: {}\033[0m".format(data.shape))
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    
    # Concvert the 'no_of_week_nights' column to category data type
    data['no_of_week_nights'] = data['no_of_week_nights'].astype('category')
    
    # Apply the following categories to the 'no_of_week_nights' column
    # 0: 0 nights
    # 1: 1 night
    # 2: 2 nights
    # 3: 3 nights
    # 4: 4 nights
    # 5: 5 nights
    # 6+: 6 or more nights
    data['no_of_week_nights'] = data['no_of_week_nights'].apply(lambda x: '0' if x == 0 else '1' if x == 1 else '2' if x == 2 else '3' if x == 3 else '4' if x == 4 else '5' if x == 5 else '6+')
    
    # Encode 'no_of_week_nights' column
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    print("\033[1;32mPerforming one-hot encoding on 'no_of_week_nights' column...\033[0m")
    print("\033[1;32mData shape before one-hot encoding: {}\033[0m".format(data.shape))
    data = perform_one_hot_encoding(data, ['no_of_week_nights'])
    print("\033[1;32mOne-hot encoding on 'no_of_week_nights' column performed successfully!\033[0m")
    print("\033[1;32mData shape after one-hot encoding: {}\033[0m".format(data.shape))
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    
    # Drop the 'required_car_parking_space' column
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    print("\033[1;32mDropping the 'required_car_parking_space' column...\033[0m")
    print("\033[1;32mData shape before dropping the 'required_car_parking_space' column: {}\033[0m".format(data.shape))
    data = data.drop('required_car_parking_space', axis=1)
    print("\033[1;32m'required_car_parking_space' column dropped successfully!\033[0m")
    print("\033[1;32mData shape after dropping the 'required_car_parking_space' column: {}\033[0m".format(data.shape))
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    
    # Drop the 'repeated_guest' column
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    print("\033[1;32mDropping the 'repeated_guest' column...\033[0m")
    print("\033[1;32mData shape before dropping the 'repeated_guest' column: {}\033[0m".format(data.shape))
    data = data.drop('repeated_guest', axis=1)
    print("\033[1;32m'repeated_guest' column dropped successfully!\033[0m")
    print("\033[1;32mData shape after dropping the 'repeated_guest' column: {}\033[0m".format(data.shape))
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    
    # Drop the 'no_of_previous_cancellations' column
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    print("\033[1;32mDropping the 'no_of_previous_cancellations' column...\033[0m")
    print("\033[1;32mData shape before dropping the 'no_of_previous_cancellations' column: {}\033[0m".format(data.shape))
    data = data.drop('no_of_previous_cancellations', axis=1)
    print("\033[1;32m'no_of_previous_cancellations' column dropped successfully!\033[0m")
    print("\033[1;32mData shape after dropping the 'no_of_previous_cancellations' column: {}\033[0m".format(data.shape))
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    
    # Drop the 'no_of_previous_bookings_not_canceled' column
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    print("\033[1;32mDropping the 'no_of_previous_bookings_not_canceled' column...\033[0m")
    print("\033[1;32mData shape before dropping the 'no_of_previous_bookings_not_canceled' column: {}\033[0m".format(data.shape))
    data = data.drop('no_of_previous_bookings_not_canceled', axis=1)
    print("\033[1;32m'no_of_previous_bookings_not_canceled' column dropped successfully!\033[0m")
    print("\033[1;32mData shape after dropping the 'no_of_previous_bookings_not_canceled' column: {}\033[0m".format(data.shape))
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    
    # Filter the data to select rows where the average price per room is zero
    zero_avg_price_per_room = data[data['avg_price_per_room'] == 0]
    index_zero_avg_price_per_room = zero_avg_price_per_room[~(zero_avg_price_per_room['market_segment_type_Online'] == 0) & (zero_avg_price_per_room['market_segment_type_Offline'] == 0) & (zero_avg_price_per_room['market_segment_type_Corporate'] == 0)].index

    # Drop the rows where the average price per room is zero and the market segment type is not Online, Offline, or Corporate
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    print("\033[1;32mDropping the rows where the average price per room is zero and the market segment type is not Online, Offline, or Corporate...\033[0m")
    print("\033[1;32mData shape before dropping the rows where the average price per room is zero and the market segment type is not Online, Offline, or Corporate: {}\033[0m".format(data.shape))
    data = data.drop(index_zero_avg_price_per_room)
    print("\033[1;32mRows where the average price per room is zero and the market segment type is not Online, Offline, or Corporate dropped successfully!\033[0m")
    print("\033[1;32mData shape after dropping the rows where the average price per room is zero and the market segment type is not Online, Offline, or Corporate: {}\033[0m".format(data.shape))
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    
    # Drop records with avg price over 500
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    print("\033[1;32mDropping records with average price per room over 500...\033[0m")
    print("\033[1;32mData shape before dropping records with average price per room over 500: {}\033[0m".format(data.shape))
    data = data[data['avg_price_per_room'] <= 500]
    print("\033[1;32mRecords with average price per room over 500 dropped successfully!\033[0m")
    print("\033[1;32mData shape after dropping records with average price per room over 500: {}\033[0m".format(data.shape))
    
    # Detect and remove outliers using the IQR method
    Q1 = data['avg_price_per_room'].quantile(0.25)
    Q3 = data['avg_price_per_room'].quantile(0.75)
    IQR = Q3 - Q1
    
    # Calculate the lower and upper bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter the data to remove outliers
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    print("\033[1;32mRemoving outliers using the IQR method...\033[0m")
    print("\033[1;32mData shape before removing outliers: {}\033[0m".format(data.shape))
    data = data[(data['avg_price_per_room'] >= lower_bound) & (data['avg_price_per_room'] <= upper_bound)]
    print("\033[1;32mOutliers removed successfully!\033[0m")
    print("\033[1;32mData shape after removing outliers: {}\033[0m".format(data.shape))
    
    # Group the 'no_of_special_requests' column into categories
    data['no_of_special_requests'] = data['no_of_special_requests'].apply(lambda x: '0' if x == 0 else '1' if x == 1 else '2+' if x >= 2 else 'Unknown')

    # Convert the 'no_of_special_requests' column to category data type
    data['no_of_special_requests'] = data['no_of_special_requests'].astype('category')
    
    # Encode 'no_of_special_requests' column
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    print("\033[1;32mPerforming one-hot encoding on 'no_of_special_requests' column...\033[0m")
    print("\033[1;32mData shape before one-hot encoding: {}\033[0m".format(data.shape))
    data = perform_one_hot_encoding(data, ['no_of_special_requests'])
    print("\033[1;32mOne-hot encoding on 'no_of_special_requests' column performed successfully!\033[0m")
    print("\033[1;32mData shape after one-hot encoding: {}\033[0m".format(data.shape))
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    
    # Remove outliers using Isolation Forest
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    print("\033[1;32mRemoving outliers using Isolation Forest...\033[0m")
    print("\033[1;32mData shape before removing outliers: {}\033[0m".format(data.shape))
    data = remove_outliers_iso_forest(data)
    print("\033[1;32mOutliers removed successfully!\033[0m")
    print("\033[1;32mData shape after removing outliers: {}\033[0m".format(data.shape))
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    
    # Get indices where the arrival_date is 29 and arrival_month is 2
    index_arrival_date_29_arrival_month_2 = data[(data['arrival_date'] == 29) & (data['arrival_month'] == 2)].index
    
    # Drop the rows where the arrival_date is 29 and the arrival_month is 2
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    print("\033[1;32mDropping the rows where the arrival_date is 29 and the arrival_month is 2...\033[0m")
    print("\033[1;32mData shape before dropping the rows where the arrival_date is 29 and the arrival_month is 2: {}\033[0m".format(data.shape))
    data = data.drop(index_arrival_date_29_arrival_month_2)
    print("\033[1;32mRows where the arrival_date is 29 and the arrival_month is 2 dropped successfully!\033[0m")
    print("\033[1;32mData shape after dropping the rows where the arrival_date is 29 and the arrival_month is 2: {}\033[0m".format(data.shape))
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    
    return data

def main() -> None:
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    print("\033[1;32mLoading data...\033[0m")
    data = load_data(RAW_FILE_PATH)
    print("\033[1;32mData loaded successfully!\033[0m")
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    print("\033[1;32mDrop the 'Booking_ID' column...\033[0m")
    print("\033[1;32mData shape before droping the 'Booking_ID' column: {}\033[0m".format(data.shape))
    data = data.drop('Booking_ID', axis=1)
    print("\033[1;32m'Booking_ID' column dropped successfully!\033[0m")
    print("\033[1;32mData shape after droping the 'Booking_ID' column: {}\033[0m".format(data.shape))
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    
    data = encoding_categorical_features(data)
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    print("\033[1;32mData encoding completed!\033[0m")
    print(data.info())
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    
    data = annomalies_detection_and_removal(data)
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    print("\033[1;32mData cleaning completed!\033[0m")
    print(data.info())
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)
    print("\033[1;32mSaving processed data...\033[0m")
    save_data(data, PROCESSED_FILE_PATH)
    print("\033[1;32mProcessed data saved successfully!\033[0m")
    print(NEW_LINE + LINE_SEPARATOR + NEW_LINE)

if __name__ == '__main__':
    main()