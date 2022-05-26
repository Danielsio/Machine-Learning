# This is a sample Python script.
import ex3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()

# dataset = ex3.load_train_data()
# datasetWithNans = ex3.drop_non_inform_columns(dataset)
# dr_filled = ex3.fill_titanic_nas(datasetWithNans)
# one_Hot = ex3.encode_one_hot(dr_filled)
# one_Hot_family = ex3.make_family(one_Hot)
# # ex3.survival_vs_gender(one_Hot_family)
# # ex3.survival_vs_family(one_Hot_family)
# # ex3.survival_vs_age(one_Hot_family)
# # ex3.survival_correlations(one_Hot_family)
# X_train, X_test, y_train, y_test = ex3.split_data(one_Hot_family)

if __name__ == '__main__':

    # an example of the usage of the functions

    df_train = ex3.load_train_data()
    ex3.disp_some_data(df_train)
    ex3.display_column_data(df_train, max_vals = 10)
    df_lean = ex3.drop_non_inform_columns(df_train)

    cols_with_nans = ex3.where_are_the_nans(df_lean)
    df_filled = ex3.fill_titanic_nas(df_lean)
    df_one_hot = ex3.encode_one_hot(df_filled)
    df_one_hot = ex3.make_family(df_one_hot)
    df_one_hot = ex3.add_log1p(df_one_hot)

    # survived_by_gender = ex3.survival_vs_gender(df_one_hot)
    # survived_by_class = ex3.survival_vs_class(df_one_hot)
    # survived_by_family = ex3.survival_vs_family(df_one_hot)
    # ex3.survival_vs_age(df_one_hot)

    important_corrs = ex3.survival_correlations(df_one_hot)
    print('\n\n', df_one_hot.columns)

    X_train, X_test, y_train, y_test = ex3.split_data(df_one_hot)
    ex3.train_logistic_regression(X_train, X_test, y_train, y_test)

