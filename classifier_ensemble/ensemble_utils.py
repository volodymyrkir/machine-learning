import pandas as pd


def bagging(data: pd.DataFrame, num_classifiers) -> list[pd.DataFrame]:
    return [data.sample(n=len(data), replace=True)
            for _ in range(num_classifiers)]


def cross_validation(data: pd.DataFrame, num_classifiers: int) -> list[pd.DataFrame]:
    indexes = list(range(0, len(data)+1, int(len(data)/num_classifiers)))

    data_slices = [pd.concat([data.iloc[:index_lower], data.iloc[index_upper:]])
                   for index_lower, index_upper in list(zip(indexes, indexes[1:]))]

    return data_slices

