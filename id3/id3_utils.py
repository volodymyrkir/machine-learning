"""This module holds utils for id3 algorithm"""
from math import log


import pandas as pd


def get_iris_data(iris_cols) -> pd.DataFrame:
    return pd.read_csv('../input_data/iris.data', names=iris_cols)


def get_congress_data(congress_cols) -> pd.DataFrame:
    return pd.read_csv('../input_data/congress.data', names=congress_cols)


def entropy(data: pd.DataFrame, target_col: str) -> float:
    """
    Calculates the entropy of a dataset based on the distribution of a target variable.

    Args:
        data (pd.DataFrame): The input dataset.
        target_col (str): The name of the target column.

    Returns:
        float: The entropy of the dataset.
    """
    counts = data[target_col].value_counts()
    total = counts.sum()
    entropy_value = 0
    for count in counts:
        prob = count / total
        entropy_value += -prob * log(prob, 2)
    return entropy_value


def get_column_max_gain(data_frame: pd.DataFrame, target_col: str, attributes: list[str]) -> str:
    """ Given data_frame, selects the column with the max info gain"""
    entropy_general = entropy(data_frame, target_col)
    instances = data_frame.shape[0]
    attributes_gains = {}
    for attribute in attributes:
        attr_value = entropy_general
        for value in data_frame[attribute].unique():
            new_frame = data_frame.loc[data_frame[attribute] == value]
            attr_value -= (new_frame.shape[0] / instances) * entropy(new_frame, target_col)
        attributes_gains[attribute] = attr_value
    return sorted(attributes_gains.items(),
                  key=lambda x: x[1],
                  reverse=True)[0][0]


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


def replace_nulls(data_frame: pd.DataFrame, target_col: str, default_null='?'):
    """Replaces nulls in dataset using most common value in class """
    non_class_columns = [col for col in data_frame.columns if col != target_col]
    for column in non_class_columns:
        for value in data_frame[target_col].unique():
            most_common_value = data_frame[data_frame[target_col] == value][column].mode()[0]

            data_frame.loc[
                (data_frame[column] == default_null) &
                (data_frame[target_col] == value), column
            ] = most_common_value
    return data_frame
