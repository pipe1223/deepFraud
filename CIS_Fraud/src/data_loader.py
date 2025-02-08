import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import gc

import warnings
# Suppress warnings
warnings.filterwarnings("ignore")



def reduce_memory_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type not in ['object', 'category']:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # Avoid converting to float16 if the column is used in one-hot encoding
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Mem. usage decreased to {end_mem:.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

def load_data_with_scaling_and_encoding():
    train_transaction = pd.read_csv('dataset/train_transaction.csv',)
    train_identity = pd.read_csv('dataset/train_identity.csv')
    test_transaction = pd.read_csv('dataset/test_transaction.csv')
    test_identity = pd.read_csv('dataset/test_identity.csv')

    # Standardize column names
    test_identity.columns = test_identity.columns.str.replace('-', '_')
    test_transaction.columns = test_transaction.columns.str.replace('-', '_')
    
    train = train_transaction.merge(train_identity, how='left', on='TransactionID')
    test = test_transaction.merge(test_identity, how='left', on='TransactionID')

    # Free up memory
    del train_transaction, train_identity, test_transaction, test_identity
    gc.collect()

    # Apply memory optimization
    train = reduce_memory_usage(train)
    test = reduce_memory_usage(test)
    
    # Handle missing values
    train.fillna(-999, inplace=True)
    test.fillna(-999, inplace=True)
    
    # Define categorical features
    categorical_features = [
        'ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain',
        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
        'id_33', 'id_34', 'DeviceType', 'DeviceInfo'
    ]
    categorical_features += [f'id_{i}' for i in range(12, 39)]

    # Encode categorical features
    for col in categorical_features:
        if col in train.columns:
            # Convert all values to strings to handle mixed data types
            train[col] = train[col].astype(str)
            test[col] = test[col].astype(str)

            le = LabelEncoder()
            combined_data = pd.concat([train[col], test[col]], axis=0)
            le.fit(combined_data)
            train[col] = le.transform(train[col])
            test[col] = le.transform(test[col])

    return train, test