"""This module holds utils for id3 algorithm"""
import pandas as pd


def discretize_stepwise(dataframe: pd.DataFrame, intervals: int) -> pd.DataFrame:
    """
    Discretizes all continuous attributes in dataframe based on intervals value
    :param dataframe: target_df
    :param intervals: num of intervals
    :returns: dataframe with discretized values
    """
    df_copy = dataframe.copy()
    numeric_cols = df_copy.select_dtypes(include=['number']).columns.tolist()
    for column in numeric_cols:
        df_copy[column] = pd.cut(dataframe[column], intervals)
    return df_copy
