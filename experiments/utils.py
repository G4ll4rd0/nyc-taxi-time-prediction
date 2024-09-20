'''
Utils
'''
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

def read_dataframe(filename: str) -> pd.DataFrame:
    '''_summary_

    Args:
        filename (str): _description_

    Returns:
        pd.DataFrame: _description_
    '''

    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df

def create_datasets():
    '''_summary_

    Returns:
        _type_: _description_
    '''
    df_train = read_dataframe('../data/green_tripdata_2024-01.parquet')
    df_val = read_dataframe('../data/green_tripdata_2024-02.parquet')

    # Feature engineering
    df_train['PU_DO'] = df_train['PULocationID'] + '_' + df_train['DOLocationID']
    df_val['PU_DO'] = df_val['PULocationID'] + '_' + df_val['DOLocationID']

    # One Hot Encoding
    categorical = ['PU_DO']  #'PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    dv = DictVectorizer()
    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    x_train = dv.fit_transform(train_dicts)
    val_dicts = df_val[categorical + numerical].to_dict(orient='records')
    x_val = dv.transform(val_dicts) # type: ignore

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    return x_train, x_val, y_train, y_val, dv
