import pandas as pd
import numpy as np
from dateutil import parser
import time
import datetime
import sys

#from src.config import TIME_COLS
sys.path.insert(0, 'E:/GitHub/hse_workshop_classification/src')
import config as cfg
# import config as cfg
# from src.config import TARGET_COLS


def set_idx(df: pd.DataFrame, idx_col: str) -> pd.DataFrame:
    df = df.set_index(idx_col)
    return df


def drop_unnecesary_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop('ID_y', axis=1)
    return df


#def add_ord_edu(df: pd.DataFrame) -> pd.DataFrame:
#    df[f'{cfg.EDU_COL}_ord'] = df[cfg.EDU_COL].str.slice(0, 1).astype(np.int8).values
#   return df


def fill_sex(df: pd.DataFrame) -> pd.DataFrame:
    most_freq = df[cfg.SEX_COL].value_counts().index[0]
    df[cfg.SEX_COL] = df[cfg.SEX_COL].fillna(most_freq)
    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    df[cfg.CAT_COLS] = df[cfg.CAT_COLS].astype('category')
    oneh_int_cols = df[cfg.ONEH_COLS].select_dtypes('number').columns
    df[oneh_int_cols] = df[oneh_int_cols].astype(np.int8)
    df[cfg.REAL_COLS] = df[cfg.REAL_COLS].astype(np.float32)
    return df


def preprocess_target(df: pd.DataFrame) -> pd.DataFrame:
    df[cfg.TARGET_COLS] = df[cfg.TARGET_COLS].astype(np.int8)
    return df


def extract_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df, target = df.drop(cfg.TARGET_COLS, axis=1), df[cfg.TARGET_COLS]
    return df, target


def drop_miss_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Заполненность менее половины
    df.drop(columns=['Возраст курения', 'Сигарет в день',
                     'Частота пасс кур'], inplace=True)
    return df

def date_encoding(df: pd.DataFrame) -> pd.DataFrame:
    # Кодировка времени
    df['Время засыпания'].replace('12:00:00', '00:00:00', inplace=True)
    df['Время засыпания'].replace('09:00:00', '21:00:00', inplace=True)
    df['Время засыпания'].replace('00:00:30', '00:30:00', inplace=True)
    df['Время пробуждения'].replace('00:06:00', '06:00:00', inplace=True)
    return df


def fill_date_na(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        mode = df[col].mode()[0]
        df[col] = df[col].fillna(mode)
    return df


def cast_time(df: pd.DataFrame) -> pd.DataFrame:
    def str_time_sec(df, time_cols = cfg.TIME_COLS):
        for col in time_cols:
            obs_num = df[col].shape[0]
            for i in range(obs_num):
                idx = df.index[i]
                #print('ВЫВОДИТСЯ: ', df.loc[idx, col], type(df.loc[idx, col]))
                df.loc[idx, col] = parser.parse(df.loc[idx, col])
                #x = time.strptime(df.loc[idx, col],'%H:%M:%S')
                #df.loc[idx, col] = 5
                #df.loc[idx, col] = datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()

        zero_time = parser.parse('00:00:00')
        #zero_time = time.strptime('00:00:00','%H:%M:%S')
        #zero_time = datetime.timedelta(hours=zero_time.tm_hour,minutes=zero_time.tm_min,seconds=zero_time.tm_sec).total_seconds()
        obs_num = df.shape[0]

        for col in time_cols:
            for i in range(obs_num):
                idx = df.index[i]
                df.loc[idx, col] = (df.loc[idx, col] - zero_time).seconds

        return df
    df.loc[:, cfg.TIME_COLS] = str_time_sec(df.loc[:, cfg.TIME_COLS]).astype(np.int8)
    return df


def transfrom_edu(df: pd.DataFrame) -> pd.DataFrame:
    df['Образование'] = df['Образование'].str[0].astype(np.int8)
    return df


def transform_smoking(df: pd.DataFrame) -> pd.DataFrame:
    df["Статус Курения"] = df["Статус Курения"].replace({'Никогда не курил(а)': 0,
                                                         'Бросил(а)': 1,
                                                         'Никогда не курил': 0,
                                                         'Бросил': 1,
                                                         'Курит': 2})
    return df


def transform_alco(df: pd.DataFrame) -> pd.DataFrame:
    df["Алкоголь"] = df["Алкоголь"].replace({'никогда не употреблял': 0,
                                             'ранее употреблял': 1,
                                             'употребляю в настоящее время': 2})
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = set_idx(df, cfg.ID_COL)
    # df = drop_unnecesary_id(df)
    df = fill_sex(df)
    df = cast_types(df)
    df = drop_miss_cols(df)
    df = date_encoding(df)
    df = fill_date_na(df)
    df = cast_time(df)
    df = transfrom_edu(df)
    df = transform_smoking(df)
    df = transform_alco(df)
    return df
