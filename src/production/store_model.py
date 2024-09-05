import pandas as pd

data = pd.read_csv('train.csv')
X = data.drop('booking_status', axis=1)
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
X = X[columns_to_preserve]
y = data['booking_status']

X.shape, y.shape

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,                   # Number of trees in the forest
    criterion='gini',                   # Function to measure the quality of a split
    max_depth=None,                     # Maximum depth of the tree
    min_samples_split=2,                # Minimum number of samples required to split an internal node
    min_samples_leaf=1,                 # Minimum number of samples required to be at a leaf node
    min_weight_fraction_leaf=0.0,       # Minimum weighted fraction of the sum total of weights
    max_features='sqrt',                # Number of features to consider when looking for the best split
    max_leaf_nodes=None,                # Grow trees with max_leaf_nodes in best-first fashion
    min_impurity_decrease=0.0,          # A node will be split if this split induces a decrease of the impurity greater than or equal to this value
    bootstrap=True,                     # Whether bootstrap samples are used when building trees
    oob_score=False,                    # Whether to use out-of-bag samples to estimate the generalization accuracy
    n_jobs=None,                        # The number of jobs to run in parallel
    random_state=42,                    # Seed of the pseudo random number generator
    verbose=1,                          # Controls the verbosity when fitting and predicting
    warm_start=False,                   # When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble
    class_weight=None,                  # Weights associated with classes in the form {class_label: weight}
    ccp_alpha=0.0,                      # Complexity parameter used for Minimal Cost-Complexity Pruning
    max_samples=None                    # If bootstrap is True, the number of samples to draw from X to train each base estimator
)

model.fit(X, y)

import bentoml

try:
    bentoml.sklearn.save_model(
        name="hotel_booking_model_1",
        model=model
    )
except Exception as e:
    print(e)
    
try:
    bentoml.models.export_model(
        tag="hotel_booking_model_1:latest",
        path="hotel_booking_model_1"
    )
except Exception as e:
    print(e)