import pandas as pd
import numpy as np

RAW_FILE_PATH = '../data/raw/hotel_reservations.csv'
PROCESSED_FILE_PATH = '../data/processed/hotel_reservations.csv'

categorical_columns_to_encode_one_hot = [
    'type_of_meal_plan',
    'room_type_reserved',
    'market_segment_type'
]

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def save_data(data: pd.DataFrame, file_path: str) -> None:
    data.to_csv(file_path, index=False)
    
def encode_one_hot(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    return pd.get_dummies(data, columns=[column_name])

def main() -> None:
    data = load_data(RAW_FILE_PATH)
    
    data = data.drop('Booking_ID', axis=1) # Drop the 'Booking_ID' column
    data = data.dropna() # Drop rows with missing values
    

if __name__ == '__main__':
    main()