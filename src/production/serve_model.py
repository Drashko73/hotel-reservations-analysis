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
            self.model = bentoml.sklearn.load_model("hotel_booking_model_1")
        except Exception as e:
            self.model = bentoml.models.import_model("./hotel_booking_model_1.bentomodel").to_runner()
    
    @bentoml.api(route="/predict_from_record")
    def predict_record(
        self, 
        input: Annotated[
            pd.DataFrame, 
            DataframeSchema(
                orient="records",
                columns=[
                    'lead_time',
                    'avg_price_per_room',
                    'arrival_date',
                    'arrival_month',
                    'no_of_special_requests_1',
                    'no_of_special_requests_2+',
                    'market_segment_type_Online',
                    'no_of_weekend_nights_1',
                    'no_of_weekend_nights_2',
                    'type_of_meal_plan_Not Selected',
                    'room_type_reserved_Room_Type 5',
                    'no_of_week_nights_3',
                    'arrival_year',
                    'type_of_meal_plan_Meal Plan 2',
                    'no_of_adults',
                    'room_type_reserved_Room_Type 6',
                    'room_type_reserved_Room_Type 4'
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
        
        # Drop columns that are not used
        try:
            input_df = self._drop_unused_columns(input_df)
            prediction = self.model.predict(input_df)
            serialized_prediction = prediction.tolist()  # Serialize ndarray to nested list
            return {"prediction": serialized_prediction}
        except Exception as e:
            return {"error": str(e)}
    
    def _drop_unused_columns(self, input_df):
        columns_to_preserve = [
            'lead_time',
            'avg_price_per_room',
            'arrival_date',
            'arrival_month',
            'no_of_special_requests_1',
            'no_of_special_requests_2+',
            'market_segment_type_Online',
            'no_of_weekend_nights_1',
            'no_of_weekend_nights_2',
            'type_of_meal_plan_Not Selected',
            'room_type_reserved_Room_Type 5',
            'no_of_week_nights_3',
            'arrival_year',
            'type_of_meal_plan_Meal Plan 2',
            'no_of_adults',
            'room_type_reserved_Room_Type 6',
            'room_type_reserved_Room_Type 4'
        ]
        return input_df[columns_to_preserve]
        