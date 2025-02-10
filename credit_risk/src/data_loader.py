
import pandas as pd
import numpy as np
import re
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


data_dir = 'dataset'



def readfile(filevar): 
    df = pd.read_csv(filevar)
    return df

def load_application_data():
    # LOAD THE TRAINING SET
    df_train = readfile(data_dir+'/application_train.csv')
    
    # LOAD THE TEST SET
    df_test = readfile(data_dir+'/application_test.csv')
    
    #missing data
    missing_values_train = df_train.isnull().sum()
    missing_values_test = df_test.isnull().sum()

    return df_train, df_test, missing_values_train, missing_values_test

def feature_engineering_application(df):
    # Create an error flag column
    df['DAYS_EMPLOYED_ERROR'] = df["DAYS_EMPLOYED"] == 365243
    # Replace the error values with nan
    df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
    
    # Flag to represent when Total income is greater than Credit
    df['INCOME_GT_CREDIT_FLAG'] = df['AMT_INCOME_TOTAL'] > df['AMT_CREDIT']
    # Column to represent Credit Income Percent
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    # Column to represent Annuity Income percent
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    # Column to represent Credit Term
    df['CREDIT_TERM'] = df['AMT_CREDIT'] / df['AMT_ANNUITY'] 
    # Column to represent Days Employed percent in his life
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    
    # Shape of Application data
    print('The shape of application data:', df.shape)
    return df

def load_bureau_data(df):
    bureau = readfile(data_dir+'/bureau.csv')

    # Combining numerical features
    bureau_numeric = bureau.drop(['SK_ID_BUREAU'], axis=1).select_dtypes(include=[np.number])
    grp = bureau_numeric.groupby('SK_ID_CURR').mean().reset_index()
    # grp = bureau.drop(['SK_ID_BUREAU'], axis = 1).groupby(by=['SK_ID_CURR']).mean().reset_index()
    grp.columns = ['BUREAU_'+column if column !='SK_ID_CURR' else column for column in grp.columns]

    #merge numerical features
    application_bureau = df.merge(grp, on='SK_ID_CURR', how='left')
    application_bureau.update(application_bureau[grp.columns].fillna(0))
    
    # Combining categorical features
    bureau_categorical = pd.get_dummies(bureau.select_dtypes('object'))
    bureau_categorical['SK_ID_CURR'] = bureau['SK_ID_CURR']
    grp = bureau_categorical.groupby(by = ['SK_ID_CURR']).mean().reset_index()
    grp.columns = ['BUREAU_'+column if column !='SK_ID_CURR' else column for column in grp.columns]

    #merge the rest
    application_bureau = application_bureau.merge(grp, on='SK_ID_CURR', how='left')
    application_bureau.update(application_bureau[grp.columns].fillna(0))
    
    # Shape of application and bureau data combined
    print('The shape application and bureau data combined:',application_bureau.shape)

    return application_bureau, bureau

def feature_engineering_bureau(df, b_df):
    # Number of past loans per customer
    grp = b_df.groupby(by = ['SK_ID_CURR'])['SK_ID_BUREAU'].count().reset_index().rename(columns = {'SK_ID_BUREAU': 'BUREAU_LOAN_COUNT'})
    df = df.merge(grp, on='SK_ID_CURR', how='left')
    df['BUREAU_LOAN_COUNT'] = df['BUREAU_LOAN_COUNT'].fillna(0)
    grp = b_df[['SK_ID_CURR', 'CREDIT_TYPE']].groupby(by = ['SK_ID_CURR'])['CREDIT_TYPE'].nunique().reset_index().rename(columns={'CREDIT_TYPE': 'BUREAU_LOAN_TYPES'})
    # Number of types of past loans per customer 
    df = df.merge(grp, on='SK_ID_CURR', how='left')
    df['BUREAU_LOAN_TYPES'] = df['BUREAU_LOAN_TYPES'].fillna(0)
    
    # Debt over credit ratio 
    b_df['AMT_CREDIT_SUM'] = b_df['AMT_CREDIT_SUM'].fillna(0)
    b_df['AMT_CREDIT_SUM_DEBT'] = b_df['AMT_CREDIT_SUM_DEBT'].fillna(0)
    
    grp1 = b_df.groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM'].sum().reset_index()
    grp1 = grp1.rename(columns={'AMT_CREDIT_SUM': 'TOTAL_CREDIT_SUM'})
    # grp1 = b_df[['SK_ID_CURR','AMT_CREDIT_SUM']].groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM'].sum().reset_index().rename(columns={'AMT_CREDIT_SUM': 'TOTAL_CREDIT_SUM'})
    
    grp2 = b_df.groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT'].sum().reset_index()
    grp2 = grp2.rename(columns={'AMT_CREDIT_SUM_DEBT': 'TOTAL_CREDIT_SUM_DEBT'})
    # grp2 = b_df[['SK_ID_CURR','AMT_CREDIT_SUM_DEBT']].groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename(columns={'AMT_CREDIT_SUM_DEBT':'TOTAL_CREDIT_SUM_DEBT'})
    
    grp1 = grp1.merge(grp2, on='SK_ID_CURR', how='left')
    grp1['DEBT_CREDIT_RATIO'] = grp1['TOTAL_CREDIT_SUM_DEBT'] / grp1['TOTAL_CREDIT_SUM']
    
    # grp1['DEBT_CREDIT_RATIO'] = grp2['TOTAL_CREDIT_SUM_DEBT']/grp1['TOTAL_CREDIT_SUM']
    grp1.drop(columns=['TOTAL_CREDIT_SUM'], inplace=True)
    # del grp1['TOTAL_CREDIT_SUM']
    df = df.merge(grp1, on='SK_ID_CURR', how='left')
    # df['DEBT_CREDIT_RATIO'] = df['DEBT_CREDIT_RATIO'].fillna(0)
    # df['DEBT_CREDIT_RATIO'] = df.replace([np.inf, -np.inf], 0)
    df['DEBT_CREDIT_RATIO'] = df['DEBT_CREDIT_RATIO'].fillna(0)
    df['DEBT_CREDIT_RATIO'] = df['DEBT_CREDIT_RATIO'].replace([np.inf, -np.inf], 0)
    
    df['DEBT_CREDIT_RATIO'] = pd.to_numeric(df['DEBT_CREDIT_RATIO'], downcast='float')
    # Overdue over debt ratio
    b_df['AMT_CREDIT_SUM_OVERDUE'] = b_df['AMT_CREDIT_SUM_OVERDUE'].fillna(0)
    b_df['AMT_CREDIT_SUM_DEBT'] = b_df['AMT_CREDIT_SUM_DEBT'].fillna(0)
    grp1 = b_df[['SK_ID_CURR','AMT_CREDIT_SUM_OVERDUE']].groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM_OVERDUE'].sum().reset_index().rename(columns={'AMT_CREDIT_SUM_OVERDUE': 'TOTAL_CUSTOMER_OVERDUE'})
    grp2 = b_df[['SK_ID_CURR','AMT_CREDIT_SUM_DEBT']].groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename(columns={'AMT_CREDIT_SUM_DEBT':'TOTAL_CUSTOMER_DEBT'})
    grp1['OVERDUE_DEBT_RATIO'] = grp1['TOTAL_CUSTOMER_OVERDUE']/grp2['TOTAL_CUSTOMER_DEBT']
    grp1.drop(columns=['TOTAL_CUSTOMER_OVERDUE'], inplace=True)
    
    df = df.merge(grp1, on='SK_ID_CURR', how='left')
    # del grp1['TOTAL_CUSTOMER_OVERDUE']
    
    df['OVERDUE_DEBT_RATIO'] = df['OVERDUE_DEBT_RATIO'].fillna(0)
    df['OVERDUE_DEBT_RATIO'] = df['OVERDUE_DEBT_RATIO'].replace([np.inf, -np.inf], 0)
    df['OVERDUE_DEBT_RATIO'] = pd.to_numeric(df['OVERDUE_DEBT_RATIO'], downcast='float')

    return df

def load_previous_applicaton(df):

    previous_applicaton = readfile(data_dir+'/previous_application.csv')
    
    # Number of previous applications per customer
    grp = previous_applicaton[['SK_ID_CURR','SK_ID_PREV']].groupby(by=['SK_ID_CURR'])['SK_ID_PREV'].count().reset_index().rename(columns={'SK_ID_PREV':'PREV_APP_COUNT'})
    application_bureau_prev = df.merge(grp, on =['SK_ID_CURR'], how = 'left')
    application_bureau_prev['PREV_APP_COUNT'] = application_bureau_prev['PREV_APP_COUNT'].fillna(0)
    # Combining numerical features
    
    
    grp = previous_applicaton.drop('SK_ID_PREV', axis=1).groupby('SK_ID_CURR').mean(numeric_only=True).reset_index()
    # grp = previous_applicaton.drop('SK_ID_PREV', axis =1).groupby(by=['SK_ID_CURR']).mean().reset_index()
    prev_columns = ['PREV_'+column if column != 'SK_ID_CURR' else column for column in grp.columns ]
    grp.columns = prev_columns
    application_bureau_prev = application_bureau_prev.merge(grp, on =['SK_ID_CURR'], how = 'left')
    application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))
    # Combining categorical features
    prev_categorical = pd.get_dummies(previous_applicaton.select_dtypes('object'))
    prev_categorical['SK_ID_CURR'] = previous_applicaton['SK_ID_CURR']
    prev_categorical.head()
    grp = prev_categorical.groupby('SK_ID_CURR').mean().reset_index()
    grp.columns = ['PREV_'+column if column != 'SK_ID_CURR' else column for column in grp.columns]
    application_bureau_prev = application_bureau_prev.merge(grp, on=['SK_ID_CURR'], how='left')
    application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))

    return application_bureau_prev

def load_pos_cash_balance(df):
    
    pos_cash = readfile(data_dir+'/POS_CASH_balance.csv')
    
    # Combining numerical features
    grp = pos_cash.drop('SK_ID_PREV', axis =1).groupby(by=['SK_ID_CURR']).mean(numeric_only=True).reset_index()
    prev_columns = ['POS_'+column if column != 'SK_ID_CURR' else column for column in grp.columns ]
    grp.columns = prev_columns
    df = df.merge(grp, on =['SK_ID_CURR'], how = 'left')
    df.update(df[grp.columns].fillna(0))
    # Combining categorical features
    pos_cash_categorical = pd.get_dummies(pos_cash.select_dtypes('object'))
    pos_cash_categorical['SK_ID_CURR'] = pos_cash['SK_ID_CURR']
    grp = pos_cash_categorical.groupby('SK_ID_CURR').mean().reset_index()
    grp.columns = ['POS_'+column if column != 'SK_ID_CURR' else column for column in grp.columns]
    df = df.merge(grp, on=['SK_ID_CURR'], how='left')
    df.update(df[grp.columns].fillna(0))
    
    return df





def load_installments_payments(df):
    
    insta_payments = readfile(data_dir+'/installments_payments.csv')
    
    # Combining numerical features and there are no categorical features in this dataset
    grp = insta_payments.drop('SK_ID_PREV', axis =1).groupby(by=['SK_ID_CURR']).mean().reset_index()
    prev_columns = ['INSTA_'+column if column != 'SK_ID_CURR' else column for column in grp.columns ]
    grp.columns = prev_columns
    df = df.merge(grp, on =['SK_ID_CURR'], how = 'left')
    df.update(df[grp.columns].fillna(0))
    
    return df


def load_credit_card_balance(df):
    
    credit_card = readfile(data_dir+'/credit_card_balance.csv')
    
    # Combining numerical features
    grp = credit_card.drop('SK_ID_PREV', axis =1).groupby(by=['SK_ID_CURR']).mean(numeric_only=True).reset_index()
    prev_columns = ['CREDIT_'+column if column != 'SK_ID_CURR' else column for column in grp.columns ]
    grp.columns = prev_columns
    df = df.merge(grp, on =['SK_ID_CURR'], how = 'left')
    df.update(df[grp.columns].fillna(0))
    # Combining categorical features
    credit_categorical = pd.get_dummies(credit_card.select_dtypes('object'))
    credit_categorical['SK_ID_CURR'] = credit_card['SK_ID_CURR']
    grp = credit_categorical.groupby('SK_ID_CURR').mean().reset_index()
    grp.columns = ['CREDIT_'+column if column != 'SK_ID_CURR' else column for column in grp.columns]
    df = df.merge(grp, on=['SK_ID_CURR'], how='left')
    df.update(df[grp.columns].fillna(0))
    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        
    return df


def preload_credit_risk_data():
    df_train, _, _, _ = load_application_data()
    df_train = feature_engineering_application(df_train)

    df_train, df_bureau = load_bureau_data(df_train)

    df_train = feature_engineering_bureau(df_train, df_bureau)
    df_train = load_previous_applicaton(df_train)

    df_train = load_pos_cash_balance(df_train)
    df_train = load_installments_payments(df_train)
    df_train = load_credit_card_balance(df_train)

    X_train = df_train
    X_train = X_train.drop(['SK_ID_CURR'],axis=1)
    y = X_train.pop('TARGET').values

    # Seperation of columns into numeric and categorical columns
    types = np.array([dt for dt in X_train.dtypes])
    all_columns = X_train.columns.values
    is_num = types != 'object'
    num_cols = all_columns[is_num]
    cat_cols = all_columns[~is_num]
    # Featurization of numeric data
    imputer_num = SimpleImputer(strategy='median')
    X_train_num = imputer_num.fit_transform(X_train[num_cols])
    scaler_num = StandardScaler()
    X_train_num1 = scaler_num.fit_transform(X_train_num)
    X_train_num_final = pd.DataFrame(X_train_num1, columns=num_cols)
    # Featurization of categorical data
    imputer_cat = SimpleImputer(strategy='constant', fill_value='MISSING')
    X_train_cat = imputer_cat.fit_transform(X_train[cat_cols])
    X_train_cat1= pd.DataFrame(X_train_cat, columns=cat_cols)
    ohe = OneHotEncoder(sparse=False,handle_unknown='ignore')
    X_train_cat2 = ohe.fit_transform(X_train_cat1)
    
    cat_cols_ohe = list(ohe.get_feature_names_out(input_features=cat_cols))
    X_train_cat_final = pd.DataFrame(X_train_cat2, columns = cat_cols_ohe)
    # Final complete data
    X_train_final = pd.concat([X_train_num_final,X_train_cat_final], axis = 1)
    print(X_train_final.shape)

    return X_train_final, y








