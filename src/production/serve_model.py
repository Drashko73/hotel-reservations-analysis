import bentoml
import bentoml.types
import pandas as pd
from pathlib import Path
from typing import Annotated
from bentoml.validators import DataframeSchema
from bentoml.validators import ContentType

@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class ProductionService():
    def __init__(self):
        try:
            model = bentoml.models.get("hotel_booking_model:latest")
        except bentoml.exceptions.NotFound:
            print("Model not found. Loading from file.")
            model = bentoml.models.import_model(path="../models/hotel_booking_model.bentomodel")
        except Exception as e:
            raise e
        
        self.model = model.load_model()
    
    @bentoml.api(route="/predict_from_record")
    def predict_record(
        self, 
        input: Annotated[
            pd.DataFrame, 
            DataframeSchema(
                orient="records",
                columns=[
                    'no_of_adults', 'lead_time', 'arrival_year', 'arrival_month',
                    'arrival_date', 'avg_price_per_room', 'type_of_meal_plan_Meal Plan 1',
                    'type_of_meal_plan_Meal Plan 2', 'type_of_meal_plan_Meal Plan 3',
                    'room_type_reserved_Room_Type 1', 'room_type_reserved_Room_Type 2',
                    'room_type_reserved_Room_Type 3', 'room_type_reserved_Room_Type 4',
                    'room_type_reserved_Room_Type 5', 'room_type_reserved_Room_Type 6',
                    'market_segment_type_Aviation', 'market_segment_type_Complementary',
                    'market_segment_type_Corporate', 'market_segment_type_Offline',
                    'with_children', 'no_of_weekend_nights_0', 'no_of_weekend_nights_1',
                    'no_of_weekend_nights_2', 'no_of_week_nights_0', 'no_of_week_nights_1',
                    'no_of_week_nights_2', 'no_of_week_nights_3', 'no_of_week_nights_4',
                    'no_of_week_nights_5', 'no_of_special_requests_0',
                    'no_of_special_requests_1'
                ]
            )]
        ) -> int:
        return self.model.predict(input)
    
    @bentoml.api(route="/predict_from_file")
    def predict_file(self, file: Annotated[Path, ContentType("text/csv")]):
        input_df = pd.read_csv(file)
        
        # Drop the target column if it exists
        if "booking_status" in input_df.columns:
            input_df = input_df.drop("booking_status", axis=1)
        
        prediction = self.model.predict(input_df)
        serialized_prediction = prediction.tolist()  # Serialize ndarray to nested list
        
        return {"prediction": serialized_prediction}