import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from skimpy import skim

__version__ = 1.1

def tips():
    print("- To get a column , use <dataframe>[<column>]")
    print("- To convert a dataframe to an array , we do : <dataframe>.values")
    print("- To get more than one type by using one data_science.get_elements_with_type() , you can use a list : ds.get_elements_with_type(pop, [type1, type2, ...]")
    print("- To use the function data_science.get_values_less_or_more_than() , we can put 'more_than' to get more than a value , and 'less_than' to get less than a value")
    print("- To create a graph with Dasc, we begin by using graph_theory.create_graph() like this : G = graph_theory.create_graph() , and then use the variable (G) to add nodes and etc...")
    print("- To visualize the graph, we use graph_theory.visualize_graph(G), we replace G by the specified graph")

class data_science:
    def find(df, index):
        return df.loc[index]

    def find_by_index(df, index):
        return df.iloc[index]

    def print_first_rows(df, rows):
        return df.head(rows)

    def print_last_rows(df, rows):
        print(df.tail(rows))

    def info(df):
        return df.info()

    def remove_missing_values(df):
        df.dropna(inplace=True)
        return df.to_string()

    def edit_missing_values(df, filling):
        df.fillna(filling, inplace=True)
        return df.to_string()

    def convert_to_datetime(df, column):
        df[column] = pd.to_datetime(df[column])
        return df.to_string()

    def replace(df, column, row, replacement):
        df.loc[row, column] = replacement

    def get_duplicates(df):
        return df.duplicated()

    def remove_duplicates(df):
        df.drop_duplicates(inplace=True)

    def sort_values(df, column, ascending_true_or_false):
        a = df.sort_values(by=column, ascending=ascending_true_or_false)
        return a

    def clean(df):
        data_science.remove_duplicates(df)
        data_science.remove_missing_values(df)
        data_science.reset_index(df)
        return df

    def set_index(df, row_index):
        df.set_index(row_index, inplace=True)
        return df

    def reset_index(df):
        a = df.columns

        def reset(d):
            d.reset_index(inplace=True)
            return d

        data_science.set_index(df, a[0])
        reset(df)
        return df

    def describe(df):
        return skim(df)

    def get_elements_with_type(df, type):
        return df.select_dtypes(include=type)

    def maximum(df, column):
        return df[column].max()

    def minimum(df, column):
        return df[column].min()

    def check_missing_values(df):
        return df.isnull().sum()

    def get_values_less_or_more_than(df, column, less_than_or_more_than, num):
        if str(less_than_or_more_than) == "more_than":
            return df[df[column] > num]
        if str(less_than_or_more_than) == "less_than":
            return df[df[column] < num]

    def shape(df):
        return df.shape

    def get_rows_with(df, column, object_name):
        return df[df[column] == object_name]

    def combine_dataframes(df1, df2):
        aa = pd.concat([df1, df2])
        a = pd.DataFrame(aa)
        return data_science.reset_index(a)

    def get_column(df, column):
        return df[column]

    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] > lower_bound) & (df[column] < upper_bound)]
        return df


class math:
    def mean_column(df, column):
        return df[column].mean()

    def median_column(df, column):
        return df[column].median()

    def corr_columns(df, column1, column2):
        df = df[column1].corr(df[column2])
        return df

    def variance(df, column):
        a = np.var(df[column])
        return a

    def rate(df, column):
        a = df[column]
        return sum(a) / len(a)

    def LinearRegression(df, feature_column, target_column, value_to_predict):
        X = df[[feature_column]]
        y = df[target_column]
        model = LinearRegression()
        model.fit(X, y)
        predicted_value = model.predict([[value_to_predict]])
        return predicted_value[0]


class plotting:
    def line_plot(df, title):
        sns.lineplot(data=df)
        plt.title(title)

    def dist_plot(df, column):
        df[column].plot.hist()

    def scatter_plot(df, column1_x, column2_y, xlabel, ylabel, title):
        plt.scatter(df[column1_x], df[column2_y])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()


class graph_theory:
    def create_graph():
        return nx.Graph()

    def add_node(graph, node):
        graph.add_node(node)

    def add_edge(graph, edge1, edge2):
        graph.add_edge(edge1, edge2)

    def visualize_graph(graph):
        nx.draw(graph, with_labels=True)
        plt.show()

    def clear(graph):
        graph.clear()

    def add_nodes_from(graph, n1, n2):
        a = []
        for i in range(n1, n2+1):
            a.append(i)
        graph.add_nodes_from(a)

    def add_edges_from(graph, n1, n2):
        a = []
        for i in range(n1, n2):
            a.append((i, i + 1))
        graph.add_edges_from(a)
