from __future__ import annotations

# [START tutorial]
# [START import_module]
import json
import pendulum

from airflow.decorators import dag, task
import pandas as pd
import snowflake.connector
import boto3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pickle
from io import StringIO
from smart_open import smart_open
from sklearn.ensemble import RandomForestClassifier

# [END import_module]

con = snowflake.connector.connect(
    user='xxx', # Change it with actual value of your snowflake.
    password='xxxx', # Change it with actual value of your snowflake.
    account='fd00396.ap-south-1.aws', # Change it with actual value of your account.
    warehouse="COMPUTE_WH", # Change it with actual value of your snowflake.
    database="DEMO_AI101", # Change it with actual value of your snowflake.
    schema="PUBLIC" # Change it with actual value of your snowflake.
)


AWS_ACCESS_KEY_ID = 'xxxx' # Change it with actual value of your AWS account.
AWS_SECRET_ACCESS_KEY = 'xxxx' # Change it with actual value of your AWS account.

# [START instantiate_dag]
@dag(
    schedule=None,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    tags=["example"],
)
def ai101_pipeline():

    # [START extract]
    @task()
    def extract():
        """
        #### Extract task
        A simple Extract task to get data ready for the rest of the data
        pipeline.
        """
        bucket_name = 'ai101'
        object_key = 'piyush/diabetic_data.csv'

        path = 's3://{}:{}@{}/{}'.format(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, bucket_name, object_key)

        sf_table = pd.read_csv(smart_open(path))

        return sf_table

    # [END extract]

    # [START transform]
    @task()
    def transform_model_build(sf_table):
        
        df = sf_table
        df = df.drop(['weight','payer_code','medical_specialty'], axis = 1)
        drop_Idx = set(df[(df['diag_1'] == '?') & (df['diag_2'] == '?') & (df['diag_3'] == '?')].index)

        drop_Idx = drop_Idx.union(set(df['diag_1'][df['diag_1'] == '?'].index))
        drop_Idx = drop_Idx.union(set(df['diag_2'][df['diag_2'] == '?'].index))
        drop_Idx = drop_Idx.union(set(df['diag_3'][df['diag_3'] == '?'].index))
        drop_Idx = drop_Idx.union(set(df['race'][df['race'] == '?'].index))
        drop_Idx = drop_Idx.union(set(df[df['discharge_disposition_id'] == 11].index))
        drop_Idx = drop_Idx.union(set(df['gender'][df['gender'] == 'Unknown/Invalid'].index))
        new_Idx = list(set(df.index) - set(drop_Idx))
        df = df.iloc[new_Idx]
        df = df.drop(['citoglipton', 'examide'], axis = 1)
        df['service_utilization'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
        keys = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'insulin', 'glyburide-metformin', 'tolazamide', 'metformin-pioglitazone','metformin-rosiglitazone', 'glimepiride-pioglitazone', 'glipizide-metformin', 'troglitazone', 'tolbutamide', 'acetohexamide']
        for col in keys:
            colname = str(col) + 'temp'
            df[colname] = df[col].apply(lambda x: 0 if (x == 'No' or x == 'Steady') else 1)
        df['numchange'] = 0
        for col in keys:
            colname = str(col) + 'temp'
            df['numchange'] = df['numchange'] + df[colname]
            del df[colname]

        # re-encoding admission type, discharge type and admission source into fewer categories

        df['admission_type_id'] = df['admission_type_id'].replace(2,1)
        df['admission_type_id'] = df['admission_type_id'].replace(7,1)
        df['admission_type_id'] = df['admission_type_id'].replace(6,5)
        df['admission_type_id'] = df['admission_type_id'].replace(8,5)

        df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(6,1)
        df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(8,1)
        df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(9,1)
        df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(13,1)
        df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(3,2)
        df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(4,2)
        df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(5,2)
        df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(14,2)
        df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(22,2)
        df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(23,2)
        df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(24,2)
        df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(12,10)
        df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(15,10)
        df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(16,10)
        df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(17,10)
        df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(25,18)
        df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(26,18)

        df['admission_source_id'] = df['admission_source_id'].replace(2,1)
        df['admission_source_id'] = df['admission_source_id'].replace(3,1)
        df['admission_source_id'] = df['admission_source_id'].replace(5,4)
        df['admission_source_id'] = df['admission_source_id'].replace(6,4)
        df['admission_source_id'] = df['admission_source_id'].replace(10,4)
        df['admission_source_id'] = df['admission_source_id'].replace(22,4)
        df['admission_source_id'] = df['admission_source_id'].replace(25,4)
        df['admission_source_id'] = df['admission_source_id'].replace(15,9)
        df['admission_source_id'] = df['admission_source_id'].replace(17,9)
        df['admission_source_id'] = df['admission_source_id'].replace(20,9)
        df['admission_source_id'] = df['admission_source_id'].replace(21,9)
        df['admission_source_id'] = df['admission_source_id'].replace(13,11)
        df['admission_source_id'] = df['admission_source_id'].replace(14,11)

        df['change'] = df['change'].replace('Ch', 1)
        df['change'] = df['change'].replace('No', 0)
        df['gender'] = df['gender'].replace('Male', 1)
        df['gender'] = df['gender'].replace('Female', 0)
        df['diabetesMed'] = df['diabetesMed'].replace('Yes', 1)
        df['diabetesMed'] = df['diabetesMed'].replace('No', 0)
        # keys is the same as before
        for col in keys:
            df[col] = df[col].replace('No', 0)
            df[col] = df[col].replace('Steady', 1)
            df[col] = df[col].replace('Up', 1)
            df[col] = df[col].replace('Down', 1)

        df['A1Cresult'] = df['A1Cresult'].replace('>7', 1)
        df['A1Cresult'] = df['A1Cresult'].replace('>8', 1)
        df['A1Cresult'] = df['A1Cresult'].replace('Norm', 0)
        df['A1Cresult'] = df['A1Cresult'].replace('None', -99)
        df['max_glu_serum'] = df['max_glu_serum'].replace('>200', 1)
        df['max_glu_serum'] = df['max_glu_serum'].replace('>300', 1)
        df['max_glu_serum'] = df['max_glu_serum'].replace('Norm', 0)
        df['max_glu_serum'] = df['max_glu_serum'].replace('None', -99)

        # code age intervals [0-10) - [90-100) from 1-10
        for i in range(0,10):
            df['age'] = df['age'].replace('['+str(10*i)+'-'+str(10*(i+1))+')', i+1)
        df['age'].value_counts()

        df2 = df.drop_duplicates(subset= ['patient_nbr'], keep = 'first')
        df['readmitted'] = df['readmitted'].replace('>30', 0)
        df['readmitted'] = df['readmitted'].replace('<30', 1)
        df['readmitted'] = df['readmitted'].replace('NO', 0)

        # Creating additional columns for diagnosis# Creati
        df['level1_diag1'] = df['diag_1']
        df['level2_diag1'] = df['diag_1']
        df['level1_diag2'] = df['diag_2']
        df['level2_diag2'] = df['diag_2']
        df['level1_diag3'] = df['diag_3']
        df['level2_diag3'] = df['diag_3']

        df.loc[df['diag_1'].str.contains('V'), ['level1_diag1', 'level2_diag1']] = 0
        df.loc[df['diag_1'].str.contains('E'), ['level1_diag1', 'level2_diag1']] = 0
        df.loc[df['diag_2'].str.contains('V'), ['level1_diag2', 'level2_diag2']] = 0
        df.loc[df['diag_2'].str.contains('E'), ['level1_diag2', 'level2_diag2']] = 0
        df.loc[df['diag_3'].str.contains('V'), ['level1_diag3', 'level2_diag3']] = 0
        df.loc[df['diag_3'].str.contains('E'), ['level1_diag3', 'level2_diag3']] = 0
        df['level1_diag1'] = df['level1_diag1'].replace('?', -1)
        df['level2_diag1'] = df['level2_diag1'].replace('?', -1)
        df['level1_diag2'] = df['level1_diag2'].replace('?', -1)
        df['level2_diag2'] = df['level2_diag2'].replace('?', -1)
        df['level1_diag3'] = df['level1_diag3'].replace('?', -1)
        df['level2_diag3'] = df['level2_diag3'].replace('?', -1)

        df['level1_diag1'] = df['level1_diag1'].astype(float)
        df['level2_diag1'] = df['level2_diag1'].astype(float)
        df['level1_diag2'] = df['level1_diag2'].astype(float)
        df['level2_diag2'] = df['level2_diag2'].astype(float)
        df['level1_diag3'] = df['level1_diag3'].astype(float)
        df['level2_diag3'] = df['level2_diag3'].astype(float)

        for index, row in df.iterrows():
            if (row['level1_diag1'] >= 390 and row['level1_diag1'] < 460) or (np.floor(row['level1_diag1']) == 785):
                df.loc[index, 'level1_diag1'] = 1
            elif (row['level1_diag1'] >= 460 and row['level1_diag1'] < 520) or (np.floor(row['level1_diag1']) == 786):
                df.loc[index, 'level1_diag1'] = 2
            elif (row['level1_diag1'] >= 520 and row['level1_diag1'] < 580) or (np.floor(row['level1_diag1']) == 787):
                df.loc[index, 'level1_diag1'] = 3
            elif (np.floor(row['level1_diag1']) == 250):
                df.loc[index, 'level1_diag1'] = 4
            elif (row['level1_diag1'] >= 800 and row['level1_diag1'] < 1000):
                df.loc[index, 'level1_diag1'] = 5
            elif (row['level1_diag1'] >= 710 and row['level1_diag1'] < 740):
                df.loc[index, 'level1_diag1'] = 6
            elif (row['level1_diag1'] >= 580 and row['level1_diag1'] < 630) or (np.floor(row['level1_diag1']) == 788):
                df.loc[index, 'level1_diag1'] = 7
            elif (row['level1_diag1'] >= 140 and row['level1_diag1'] < 240):
                df.loc[index, 'level1_diag1'] = 8
            else:
                df.loc[index, 'level1_diag1'] = 0

            if (row['level1_diag2'] >= 390 and row['level1_diag2'] < 460) or (np.floor(row['level1_diag2']) == 785):
                df.loc[index, 'level1_diag2'] = 1
            elif (row['level1_diag2'] >= 460 and row['level1_diag2'] < 520) or (np.floor(row['level1_diag2']) == 786):
                df.loc[index, 'level1_diag2'] = 2
            elif (row['level1_diag2'] >= 520 and row['level1_diag2'] < 580) or (np.floor(row['level1_diag2']) == 787):
                df.loc[index, 'level1_diag2'] = 3
            elif (np.floor(row['level1_diag2']) == 250):
                df.loc[index, 'level1_diag2'] = 4
            elif (row['level1_diag2'] >= 800 and row['level1_diag2'] < 1000):
                df.loc[index, 'level1_diag2'] = 5
            elif (row['level1_diag2'] >= 710 and row['level1_diag2'] < 740):
                df.loc[index, 'level1_diag2'] = 6
            elif (row['level1_diag2'] >= 580 and row['level1_diag2'] < 630) or (np.floor(row['level1_diag2']) == 788):
                df.loc[index, 'level1_diag2'] = 7
            elif (row['level1_diag2'] >= 140 and row['level1_diag2'] < 240):
                df.loc[index, 'level1_diag2'] = 8
            else:
                df.loc[index, 'level1_diag2'] = 0

            if (row['level1_diag3'] >= 390 and row['level1_diag3'] < 460) or (np.floor(row['level1_diag3']) == 785):
                df.loc[index, 'level1_diag3'] = 1
            elif (row['level1_diag3'] >= 460 and row['level1_diag3'] < 520) or (np.floor(row['level1_diag3']) == 786):
                df.loc[index, 'level1_diag3'] = 2
            elif (row['level1_diag3'] >= 520 and row['level1_diag3'] < 580) or (np.floor(row['level1_diag3']) == 787):
                df.loc[index, 'level1_diag3'] = 3
            elif (np.floor(row['level1_diag3']) == 250):
                df.loc[index, 'level1_diag3'] = 4
            elif (row['level1_diag3'] >= 800 and row['level1_diag3'] < 1000):
                df.loc[index, 'level1_diag3'] = 5
            elif (row['level1_diag3'] >= 710 and row['level1_diag3'] < 740):
                df.loc[index, 'level1_diag3'] = 6
            elif (row['level1_diag3'] >= 580 and row['level1_diag3'] < 630) or (np.floor(row['level1_diag3']) == 788):
                df.loc[index, 'level1_diag3'] = 7
            elif (row['level1_diag3'] >= 140 and row['level1_diag3'] < 240):
                df.loc[index, 'level1_diag3'] = 8
            else:
                df.loc[index, 'level1_diag3'] = 0

        for index, row in df.iterrows():
            if (row['level2_diag1'] >= 390 and row['level2_diag1'] < 399):
                df.loc[index, 'level2_diag1'] = 1
            elif (row['level2_diag1'] >= 401 and row['level2_diag1'] < 415):
                df.loc[index, 'level2_diag1'] = 2
            elif (row['level2_diag1'] >= 415 and row['level2_diag1'] < 460):
                df.loc[index, 'level2_diag1'] = 3
            elif (np.floor(row['level2_diag1']) == 785):
                df.loc[index, 'level2_diag1'] = 4
            elif (row['level2_diag1'] >= 460 and row['level2_diag1'] < 489):
                df.loc[index, 'level2_diag1'] = 5
            elif (row['level2_diag1'] >= 490 and row['level2_diag1'] < 497):
                df.loc[index, 'level2_diag1'] = 6
            elif (row['level2_diag1'] >= 500 and row['level2_diag1'] < 520):
                df.loc[index, 'level2_diag1'] = 7
            elif (np.floor(row['level2_diag1']) == 786):
                df.loc[index, 'level2_diag1'] = 8
            elif (row['level2_diag1'] >= 520 and row['level2_diag1'] < 530):
                df.loc[index, 'level2_diag1'] = 9
            elif (row['level2_diag1'] >= 530 and row['level2_diag1'] < 544):
                df.loc[index, 'level2_diag1'] = 10
            elif (row['level2_diag1'] >= 550 and row['level2_diag1'] < 554):
                df.loc[index, 'level2_diag1'] = 11
            elif (row['level2_diag1'] >= 555 and row['level2_diag1'] < 580):
                df.loc[index, 'level2_diag1'] = 12
            elif (np.floor(row['level2_diag1']) == 787):
                df.loc[index, 'level2_diag1'] = 13
            elif (np.floor(row['level2_diag1']) == 250):
                df.loc[index, 'level2_diag1'] = 14
            elif (row['level2_diag1'] >= 800 and row['level2_diag1'] < 1000):
                df.loc[index, 'level2_diag1'] = 15
            elif (row['level2_diag1'] >= 710 and row['level2_diag1'] < 740):
                df.loc[index, 'level2_diag1'] = 16
            elif (row['level2_diag1'] >= 580 and row['level2_diag1'] < 630):
                df.loc[index, 'level2_diag1'] = 17
            elif (np.floor(row['level2_diag1']) == 788):
                df.loc[index, 'level2_diag1'] = 18
            elif (row['level2_diag1'] >= 140 and row['level2_diag1'] < 240):
                df.loc[index, 'level2_diag1'] = 19
            elif row['level2_diag1'] >= 240 and row['level2_diag1'] < 280 and (np.floor(row['level2_diag1']) != 250):
                df.loc[index, 'level2_diag1'] = 20
            elif (row['level2_diag1'] >= 680 and row['level2_diag1'] < 710) or (np.floor(row['level2_diag1']) == 782):
                df.loc[index, 'level2_diag1'] = 21
            elif (row['level2_diag1'] >= 290 and row['level2_diag1'] < 320):
                df.loc[index, 'level2_diag1'] = 22
            else:
                df.loc[index, 'level2_diag1'] = 0

            if (row['level2_diag2'] >= 390 and row['level2_diag2'] < 399):
                df.loc[index, 'level2_diag2'] = 1
            elif (row['level2_diag2'] >= 401 and row['level2_diag2'] < 415):
                df.loc[index, 'level2_diag2'] = 2
            elif (row['level2_diag2'] >= 415 and row['level2_diag2'] < 460):
                df.loc[index, 'level2_diag2'] = 3
            elif (np.floor(row['level2_diag2']) == 785):
                df.loc[index, 'level2_diag2'] = 4
            elif (row['level2_diag2'] >= 460 and row['level2_diag2'] < 489):
                df.loc[index, 'level2_diag2'] = 5
            elif (row['level2_diag2'] >= 490 and row['level2_diag2'] < 497):
                df.loc[index, 'level2_diag2'] = 6
            elif (row['level2_diag2'] >= 500 and row['level2_diag2'] < 520):
                df.loc[index, 'level2_diag2'] = 7
            elif (np.floor(row['level2_diag2']) == 786):
                df.loc[index, 'level2_diag2'] = 8
            elif (row['level2_diag2'] >= 520 and row['level2_diag2'] < 530):
                df.loc[index, 'level2_diag2'] = 9
            elif (row['level2_diag2'] >= 530 and row['level2_diag2'] < 544):
                df.loc[index, 'level2_diag2'] = 10
            elif (row['level2_diag2'] >= 550 and row['level2_diag2'] < 554):
                df.loc[index, 'level2_diag2'] = 11
            elif (row['level2_diag2'] >= 555 and row['level2_diag2'] < 580):
                df.loc[index, 'level2_diag2'] = 12
            elif (np.floor(row['level2_diag2']) == 787):
                df.loc[index, 'level2_diag2'] = 13
            elif (np.floor(row['level2_diag2']) == 250):
                df.loc[index, 'level2_diag2'] = 14
            elif (row['level2_diag2'] >= 800 and row['level2_diag2'] < 1000):
                df.loc[index, 'level2_diag2'] = 15
            elif (row['level2_diag2'] >= 710 and row['level2_diag2'] < 740):
                df.loc[index, 'level2_diag2'] = 16
            elif (row['level2_diag2'] >= 580 and row['level2_diag2'] < 630):
                df.loc[index, 'level2_diag2'] = 17
            elif (np.floor(row['level2_diag2']) == 788):
                df.loc[index, 'level2_diag2'] = 18
            elif (row['level2_diag2'] >= 140 and row['level2_diag2'] < 240):
                df.loc[index, 'level2_diag2'] = 19
            elif row['level2_diag2'] >= 240 and row['level2_diag2'] < 280 and (np.floor(row['level2_diag2']) != 250):
                df.loc[index, 'level2_diag2'] = 20
            elif (row['level2_diag2'] >= 680 and row['level2_diag2'] < 710) or (np.floor(row['level2_diag2']) == 782):
                df.loc[index, 'level2_diag2'] = 21
            elif (row['level2_diag2'] >= 290 and row['level2_diag2'] < 320):
                df.loc[index, 'level2_diag2'] = 22
            else:
                df.loc[index, 'level2_diag2'] = 0


            if (row['level2_diag3'] >= 390 and row['level2_diag3'] < 399):
                df.loc[index, 'level2_diag3'] = 1
            elif (row['level2_diag3'] >= 401 and row['level2_diag3'] < 415):
                df.loc[index, 'level2_diag3'] = 2
            elif (row['level2_diag3'] >= 415 and row['level2_diag3'] < 460):
                df.loc[index, 'level2_diag3'] = 3
            elif (np.floor(row['level2_diag3']) == 785):
                df.loc[index, 'level2_diag3'] = 4
            elif (row['level2_diag3'] >= 460 and row['level2_diag3'] < 489):
                df.loc[index, 'level2_diag3'] = 5
            elif (row['level2_diag3'] >= 490 and row['level2_diag3'] < 497):
                df.loc[index, 'level2_diag3'] = 6
            elif (row['level2_diag3'] >= 500 and row['level2_diag3'] < 520):
                df.loc[index, 'level2_diag3'] = 7
            elif (np.floor(row['level2_diag3']) == 786):
                df.loc[index, 'level2_diag3'] = 8
            elif (row['level2_diag3'] >= 520 and row['level2_diag3'] < 530):
                df.loc[index, 'level2_diag3'] = 9
            elif (row['level2_diag3'] >= 530 and row['level2_diag3'] < 544):
                df.loc[index, 'level2_diag3'] = 10
            elif (row['level2_diag3'] >= 550 and row['level2_diag3'] < 554):
                df.loc[index, 'level2_diag3'] = 11
            elif (row['level2_diag3'] >= 555 and row['level2_diag3'] < 580):
                df.loc[index, 'level2_diag3'] = 12
            elif (np.floor(row['level2_diag3']) == 787):
                df.loc[index, 'level2_diag3'] = 13
            elif (np.floor(row['level2_diag3']) == 250):
                df.loc[index, 'level2_diag3'] = 14
            elif (row['level2_diag3'] >= 800 and row['level2_diag3'] < 1000):
                df.loc[index, 'level2_diag3'] = 15
            elif (row['level2_diag3'] >= 710 and row['level2_diag3'] < 740):
                df.loc[index, 'level2_diag3'] = 16
            elif (row['level2_diag3'] >= 580 and row['level2_diag3'] < 630):
                df.loc[index, 'level2_diag3'] = 17
            elif (np.floor(row['level2_diag3']) == 788):
                df.loc[index, 'level2_diag3'] = 18
            elif (row['level2_diag3'] >= 140 and row['level2_diag3'] < 240):
                df.loc[index, 'level2_diag3'] = 19
            elif row['level2_diag3'] >= 240 and row['level2_diag3'] < 280 and (np.floor(row['level2_diag3']) != 250):
                df.loc[index, 'level2_diag3'] = 20
            elif (row['level2_diag3'] >= 680 and row['level2_diag3'] < 710) or (np.floor(row['level2_diag3']) == 782):
                df.loc[index, 'level2_diag3'] = 21
            elif (row['level2_diag3'] >= 290 and row['level2_diag3'] < 320):
                df.loc[index, 'level2_diag3'] = 22
            else:
                df.loc[index, 'level2_diag3'] = 0

        df['age'] = df['age'].astype('int64')
        # convert age categories to mid-point values
        age_dict = {1:5, 2:15, 3:25, 4:35, 5:45, 6:55, 7:65, 8:75, 9:85, 10:95}
        df['age'] = df.age.map(age_dict)
        # print(df.age.value_counts())

        # convert data type of nominal features in dataframe to 'object' type
        i = ['encounter_id', 'patient_nbr', 'gender', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id',\
                  'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', \
                  'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose','miglitol', \
                  'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin', \
                  'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed', \
                  'age', 'max_glu_serum', 'level1_diag1', 'level1_diag2', 'level1_diag3', 'level2_diag1', 
             'level2_diag2', 'level2_diag3']

        df[i] = df[i].astype('object')

        df['nummed'] = 0

        for col in keys:
            df['nummed'] = df['nummed'] + df[col]
        df['nummed'].value_counts()

        # get list of only numeric features
        num_col = list(set(list(df._get_numeric_data().columns))- {'readmitted','age'})
        num_col

        # Removing skewnewss and kurtosis using log transformation if it is above a threshold value -  2

        statdataframe = pd.DataFrame()
        statdataframe['numeric_column'] = num_col
        skew_before = []
        skew_after = []

        kurt_before = []
        kurt_after = []

        standard_deviation_before = []
        standard_deviation_after = []

        log_transform_needed = []

        log_type = []

        for i in num_col:
            skewval = df[i].skew()
            skew_before.append(skewval)

            kurtval = df[i].kurtosis()
            kurt_before.append(kurtval)

            sdval = df[i].std()
            standard_deviation_before.append(sdval)

            if (abs(skewval) >2) & (abs(kurtval) >2):
                log_transform_needed.append('Yes')

                if len(df[df[i] == 0])/len(df) <=0.02:
                    log_type.append('log')
                    skewvalnew = np.log(pd.DataFrame(df[train_data[i] > 0])[i]).skew()
                    skew_after.append(skewvalnew)

                    kurtvalnew = np.log(pd.DataFrame(df[train_data[i] > 0])[i]).kurtosis()
                    kurt_after.append(kurtvalnew)

                    sdvalnew = np.log(pd.DataFrame(df[train_data[i] > 0])[i]).std()
                    standard_deviation_after.append(sdvalnew)

                else:
                    log_type.append('log1p')
                    skewvalnew = np.log1p(pd.DataFrame(df[df[i] >= 0])[i]).skew()
                    skew_after.append(skewvalnew)

                    kurtvalnew = np.log1p(pd.DataFrame(df[df[i] >= 0])[i]).kurtosis()
                    kurt_after.append(kurtvalnew)

                    sdvalnew = np.log1p(pd.DataFrame(df[df[i] >= 0])[i]).std()
                    standard_deviation_after.append(sdvalnew)

            else:
                log_type.append('NA')
                log_transform_needed.append('No')

                skew_after.append(skewval)
                kurt_after.append(kurtval)
                standard_deviation_after.append(sdval)

        statdataframe['skew_before'] = skew_before
        statdataframe['kurtosis_before'] = kurt_before
        statdataframe['standard_deviation_before'] = standard_deviation_before
        statdataframe['log_transform_needed'] = log_transform_needed
        statdataframe['log_type'] = log_type
        statdataframe['skew_after'] = skew_after
        statdataframe['kurtosis_after'] = kurt_after
        statdataframe['standard_deviation_after'] = standard_deviation_after

        # performing the log transformation for the columns determined to be needing it above.

        for i in range(len(statdataframe)):
            if statdataframe['log_transform_needed'][i] == 'Yes':
                colname = str(statdataframe['numeric_column'][i])

                if statdataframe['log_type'][i] == 'log':
                    df = df[df[colname] > 0]
                    df[colname + "_log"] = np.log(df[colname])

                elif statdataframe['log_type'][i] == 'log1p':
                    df = df[df[colname] >= 0]
                    df[colname + "_log1p"] = np.log1p(df[colname])


        df = df.drop(['number_outpatient', 'number_inpatient', 'number_emergency','service_utilization'], axis = 1)
        # get list of only numeric features
        numerics = list(set(list(df._get_numeric_data().columns))- {'readmitted'})
        # show list of features that are categorical
        df.encounter_id = df.encounter_id.astype('int64')
        df.patient_nbr = df.patient_nbr.astype('int64')
        df.diabetesMed = df.diabetesMed.astype('int64')
        df.change = df.change.astype('int64')

        # convert data type of nominal features in dataframe to 'object' type for aggregating
        i = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', \
                  'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose','miglitol', \
                  'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin', \
                  'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone','A1Cresult']

        dfcopy = df.copy(deep = True)
        df['readmitted'] = df['readmitted'].apply(lambda x: 0 if x == 2 else x)
        df.drop(['diag_1', 'diag_2', 'diag_3', 'level2_diag1', 'level1_diag2', 'level2_diag2', 'level1_diag3',
         'level2_diag3'], axis=1, inplace=True)
        interactionterms = [('num_medications','time_in_hospital'),
            ('num_medications','num_procedures'),
            ('time_in_hospital','num_lab_procedures'),
            ('num_medications','num_lab_procedures'),
            ('num_medications','number_diagnoses'),
            ('age','number_diagnoses'),
            ('change','num_medications'),
            ('number_diagnoses','time_in_hospital'),
            ('num_medications','numchange')]

        for inter in interactionterms:
            name = inter[0] + '|' + inter[1]
            df[name] = df[inter[0]] * df[inter[1]]

        df[['num_medications','time_in_hospital', 'num_medications|time_in_hospital']].head()
        # Feature Scaling
        datf = pd.DataFrame()
        datf['features'] = numerics
        datf['std_dev'] = datf['features'].apply(lambda x: df[x].std())
        datf['mean'] = datf['features'].apply(lambda x: df[x].mean())

        # dropping multiple encounters while keeping either first or last encounter of these patients
        df2 = df.drop_duplicates(subset= ['patient_nbr'], keep = 'first')
        df2.shape

        # standardize function
        def standardize(raw_data):
            return ((raw_data - np.mean(raw_data, axis = 0)) / np.std(raw_data, axis = 0))

        df2[numerics] = standardize(df2[numerics])
        import scipy as sp
        df2 = df2[(np.abs(sp.stats.zscore(df2[numerics])) < 3).all(axis=1)]

        df2['level1_diag1'] = df2['level1_diag1'].astype('object')
        df_pd = pd.get_dummies(df2, columns=['gender', 'admission_type_id', 'discharge_disposition_id',
                                              'admission_source_id', 'max_glu_serum', 'A1Cresult', 'level1_diag1'], drop_first = True)
        just_dummies = pd.get_dummies(df_pd['race'])
        df_pd = pd.concat([df_pd, just_dummies], axis=1)
        df_pd.drop(['race'], inplace=True, axis=1)

        non_num_cols = ['race', 'gender', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
                'max_glu_serum', 'A1Cresult', 'level1_diag1' ]

        num_cols = list(set(list(df._get_numeric_data().columns))- {'readmitted', 'change'})

        new_non_num_cols = []
        for i in non_num_cols:
            for j in df_pd.columns:
                if i in j:
                    new_non_num_cols.append(j)

        l = []
        for feature in list(df_pd.columns):
            if '|' in feature:
                l.append(feature)

        feature_set_no_int = ['age', 'time_in_hospital', 'num_procedures', 'num_medications', 'number_outpatient_log1p',
                 'number_emergency_log1p', 'number_inpatient_log1p', 'number_diagnoses', 'metformin',
                 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'glipizide',
                 'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                 'tolazamide', 'insulin', 'glyburide-metformin',
                 'AfricanAmerican', 'Asian', 'Caucasian',
                 'Hispanic', 'Other', 'gender_1',
                 'admission_type_id_3', 'admission_type_id_5',
                 'discharge_disposition_id_2', 'discharge_disposition_id_7',
                 'discharge_disposition_id_10', 'discharge_disposition_id_18',
                 'admission_source_id_4', 'admission_source_id_7',
                 'admission_source_id_9',
                 'level1_diag1_1.0',
                 'level1_diag1_2.0',
                 'level1_diag1_3.0',
                 'level1_diag1_4.0',
                 'level1_diag1_5.0',
                 'level1_diag1_6.0',
                 'level1_diag1_7.0',
                 'level1_diag1_8.0']
        X = df_pd[feature_set_no_int]
        y = df_pd['readmitted']

        print('Original dataset shape {}'.format(Counter(y)))
        smt = SMOTE(random_state=20)
        train_input_new, train_output_new = smt.fit_resample(X, y)
        print('New dataset shape {}'.format(Counter(train_output_new)))
        train_input_new = pd.DataFrame(train_input_new, columns = list(X.columns))
        X_train, X_test, y_train, y_test = train_test_split(train_input_new, train_output_new, test_size=0.20, random_state=0)

        rm = RandomForestClassifier(n_estimators = 10, max_depth=25, criterion = "gini", min_samples_split=10)
        rm.fit(X_train, y_train)
        return rm

    # [END transform]

    # [START load]
    @task()
    def load(model):
        """
        #### Load task
        Load the output pickle file to S3 bucket.
        """
        # Save the model to a pickle file
        with open('trained_model_RForest2.pkl', 'wb') as file:
            pickle.dump(model, file)

        # print("Model saved to trained_model_RF.pkl")

        session = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
        s3 = session.resource('s3')

        BUCKET = "ai101"

        s3.Bucket(BUCKET).upload_file("./trained_model_RForest2.pkl", "piyush/trained_model_RForest2.pkl")

    # [END load]

    # [START main_flow]
    sf_table = extract()
    model = transform_model_build(sf_table)
    load(model)
    # [END main_flow]


# [START dag_invocation]
ai101_pipeline()
# [END dag_invocation]

# [END tutorial]
