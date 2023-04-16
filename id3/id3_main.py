"""Entrypoint for ID3 algorithm package"""
from id3_algorithm import TreeManager

from utils import (get_data, replace_nulls,
                   IRIS_COLS, IRIS_ATTRIBUTES, IRIS_TARGET,
                   CONGRESS_COLS, CONGRESS_ATTRIBUTES, CONGRESS_TARGET)


if __name__ == '__main__':
    iris_data = get_data('iris.data', IRIS_COLS)
    congress_data = replace_nulls(get_data('congress.data', CONGRESS_COLS), CONGRESS_TARGET)
    tree_manager = TreeManager()
    tree = tree_manager.create_tree(iris_data, IRIS_TARGET, IRIS_ATTRIBUTES, discrete='info_gain')
    tree.predict()
    tree1 = tree_manager.create_tree(iris_data, IRIS_TARGET, IRIS_ATTRIBUTES, discrete='stepwise')
    tree1.predict()
    tree2 = tree_manager.create_tree(congress_data, CONGRESS_TARGET, CONGRESS_ATTRIBUTES, discrete='info_gain')
    tree2.predict()
    tree3 = tree_manager.create_tree(congress_data, CONGRESS_TARGET, CONGRESS_ATTRIBUTES, discrete='stepwise')
    tree3.predict()
