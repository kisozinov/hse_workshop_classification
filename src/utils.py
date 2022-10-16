from pandas import DataFrame

def save_as_pickle(df: DataFrame, path: str) -> None:
    df.to_pickle(path)
