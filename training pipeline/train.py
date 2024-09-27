'''
Automatic training with flows
'''
import pathlib
import pickle
from datetime import datetime

import dagshub
import mlflow
import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from prefect import flow, task
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error  # type: ignore


@task(name = 'Read Data')
def read_data(file_path: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    df = pd.read_parquet(file_path)

    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df

@task(name = 'Add Feature')
def add_features(df_train: pd.DataFrame, df_val: pd.DataFrame):
    """Add features to the model"""
    df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
    df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]

    categorical = ["PU_DO"]  #'PULocationID', 'DOLocationID']
    numerical = ["trip_distance"]

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    x_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    x_val = dv.transform(val_dicts) # type: ignore

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values
    return x_train, x_val, y_train, y_val, dv

@task(name = 'Hyper-Parameter Tunning')
def hyper_parameter_tunning(x_train, x_val, y_train, y_val, dv):
    '''Hyper parameter tune

    Args:
        x_train (_type_): _description_
        x_val (_type_): _description_
        y_train (_type_): _description_
        y_val (_type_): _description_
        dv (_type_): _description_

    Returns:
        _type_: _description_
    '''

    mlflow.xgboost.autolog()

    training_dataset = mlflow.data.from_numpy(x_train.data, targets=y_train, name="green_tripdata_2024-01")

    validation_dataset = mlflow.data.from_numpy(x_val.data, targets=y_val, name="green_tripdata_2024-02")

    train = xgb.DMatrix(x_train, label=y_train)

    valid = xgb.DMatrix(x_val, label=y_val)

    def objective(params):
        with mlflow.start_run(nested=True):

            # Tag model
            mlflow.set_tag("model_family", "xgboost")

            # Train model
            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=100,
                evals=[(valid, 'validation')],
                early_stopping_rounds=10
            )

            # Predict in the val dataset
            y_pred = booster.predict(valid)

            # Calculate metric
            rmse = root_mean_squared_error(y_val, y_pred)

            # Log performance metric
            mlflow.log_metric("rmse", rmse)

        return {'loss': rmse, 'status': STATUS_OK}

    with mlflow.start_run(run_name="Xgboost Hyper-parameter Optimization", nested=True):
        search_space = {
            'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
            'learning_rate': hp.loguniform('learning_rate', -3, 0),
            'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
            'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
            'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
            'objective': 'reg:squarederror',
            'seed': 42
        }

        best_params = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=10,
            trials=Trials()
        )
        best_params["max_depth"] = int(best_params["max_depth"])
        best_params["seed"] = 42
        best_params["objective"] = "reg:squarederror"

        mlflow.log_params(best_params)

    return best_params

@task(name = 'Train Best Model')
def train_best_model(x_train, x_val, y_train, y_val, dv, best_params) -> None:
    """train a model with best hyperparams and write everything out"""

    with mlflow.start_run(run_name = "Best model ever"):
        train = xgb.DMatrix(x_train, label=y_train)
        valid = xgb.DMatrix(x_val, label=y_val)

        mlflow.log_params(best_params)

        # Log a fit model instance
        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, 'validation')],
            early_stopping_rounds=10
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        pathlib.Path("models").mkdir(exist_ok=True)
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

@task(name = 'Register Model')
def register_best_model(client):
    experiment_name = 'nyc-taxi-experiment-prefect_auto_registry'
    current_experiment=dict(mlflow.get_experiment_by_name(experiment_name))
    experiment_id=current_experiment['experiment_id']
    best_run = client.search_runs(
        experiment_id, order_by=["metrics.rmse ASC"], max_results=1
    )[0]
    info = best_run.info
    run_id = info.run_id
    run_uri = f"runs:/{run_id}/model"

    result = mlflow.register_model(
        model_uri=run_uri,
        name="nyc-taxi-model-prefect"
    )

@task(name = 'Set Champion')
def set_champion(client):
    
    client.update_registered_model(
        name="nyc-taxi-model-prefect",
        description="Model registry for the NYC Taxi Time Prediction Project",
    )

    new_alias = "champion"
    date = datetime.today()
    model_version = "1"

    # create "champion" alias for version 1 of model "nyc-taxi-model"
    client.set_registered_model_alias(
        name="nyc-taxi-model-prefect",
        alias=new_alias,
        version=model_version
    )

    client.update_model_version(
        name="nyc-taxi-model-prefect",
        version=model_version,
        description=f"The model version {model_version} was transitioned to {new_alias} on {date}",
    )

@flow(name = 'Register Champion')
def register_champion(ml_tr):
    client = mlflow.MlflowClient(tracking_uri=ml_tr)
    register_best_model(client)
    set_champion(client)

@flow(name = 'Main Flow')
def main_flow(year: str, month_train: str, month_val: str) -> None:
    """The main training pipeline"""

    train_path = f"./data/green_tripdata_{year}-{month_train}.parquet"
    val_path = f"./data/green_tripdata_{year}-{month_val}.parquet"

    # MLflow settings
    dagshub.init(url="https://dagshub.com/G4ll4rd0/nyc-taxi-time-prediction", mlflow=True)

    mlflow_tracking_uri = mlflow.get_tracking_uri()

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name="nyc-taxi-experiment-prefect_auto_registry")

    # Load
    df_train = read_data(train_path)
    df_val = read_data(val_path)

    # Transform
    x_train, x_val, y_train, y_val, dv = add_features(df_train, df_val)

    # Hyper-parameter Tunning
    best_params = hyper_parameter_tunning(x_train, x_val, y_train, y_val, dv)

    # Train
    train_best_model(x_train, x_val, y_train, y_val, dv, best_params)

    # Registry
    register_champion(mlflow_tracking_uri)

main_flow(year = '2024', month_train = '01', month_val = '02')