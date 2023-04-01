"""Entrypoint for ID3 algorithm package"""
from id3_algorithm import TreeManager
from id3_utils import get_iris_data, get_congress_data, replace_nulls

IRIS_COLS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
IRIS_ATTRIBUTES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
IRIS_TARGET = 'species'

CONGRESS_COLS = ['class', *[f'c{n}' for n in range(1, 17)]]
CONGRESS_TARGET = 'class'
CONGRESS_ATTRIBUTES = [f'c{n}' for n in range(1, 17)]
if __name__ == '__main__':
    iris_data = get_iris_data(IRIS_COLS)
    congress_data = replace_nulls(get_congress_data(CONGRESS_COLS), CONGRESS_TARGET)
    tree_manager = TreeManager()
    tree = tree_manager.create_tree(iris_data, IRIS_TARGET, IRIS_ATTRIBUTES, discrete='info_gain')
    tree.predict()
    tree1 = tree_manager.create_tree(iris_data, IRIS_TARGET, IRIS_ATTRIBUTES, discrete='stepwise')
    tree1.predict()
    tree2 = tree_manager.create_tree(congress_data, CONGRESS_TARGET, CONGRESS_ATTRIBUTES, discrete='info_gain')
    tree2.predict()
    tree3 = tree_manager.create_tree(congress_data, CONGRESS_TARGET, CONGRESS_ATTRIBUTES, discrete='stepwise')
    tree3.predict()
