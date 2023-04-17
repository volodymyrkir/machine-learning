"""This module holds utils for id3 algorithm"""
import pandas as pd

from utils import entropy


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


def discretize_info_gain(dataframe: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Discretizes all continuous attributes in dataframe based on info gain for each column
    :param dataframe: target_df
    :param target_col: target column of dataframe
    :returns: dataframe with discretized values
    """
    dataframe_copy = dataframe.copy()
    numeric_cols = dataframe_copy.select_dtypes(include=['number']).columns.tolist()
    num_bins = len(dataframe[target_col].unique()) - 1
    total_instances = dataframe_copy.shape[0]

    for col in numeric_cols:
        df = dataframe_copy[[col, target_col]].copy()
        df = df.sort_values([col, target_col], ascending=True)
        thresholds = set(df.loc[df[target_col].ne(df[target_col].shift())][col][1:].tolist())
        thresholds_mapping = {}
        if len(thresholds) == num_bins:
            pass
        for threshold in thresholds:
            left, right = df.loc[df[col] <= threshold], df.loc[df[col] > threshold]
            inf_gain = entropy(df, target_col) - ((len(left) / total_instances) * entropy(left, target_col)
                                                  + (len(right) / total_instances) * entropy(right, target_col))
            thresholds_mapping[threshold] = inf_gain
        best_thresholds = sorted(thresholds_mapping.items(),
                                 key=lambda x: x[1],
                                 reverse=True)[:num_bins]
        prev_threshold = min(df[col])
        max_threshold = max(df[col])
        for threshold, _ in best_thresholds:
            dataframe_copy.loc[(dataframe[col] < threshold) &
                               (dataframe[col] >= prev_threshold), col] = f'{prev_threshold}-{threshold}'
            prev_threshold = threshold
        dataframe_copy.loc[dataframe[col] >= prev_threshold, col] = f'{prev_threshold}-{max_threshold}'
    return dataframe_copy
