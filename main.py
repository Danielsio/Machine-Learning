# This is a sample Python script.
import ex3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()

dataset = ex3.load_train_data()
datasetWithNans = ex3.drop_non_inform_columns(dataset)
dr_filled = ex3.fill_titanic_nas(datasetWithNans)

print(dr_filled.loc[dr_filled["Age"] == 62])

#ex3.encode_one_hot(dr_filled)


