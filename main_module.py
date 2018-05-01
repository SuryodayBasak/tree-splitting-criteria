import numpy as np
import pandas as pd
import split_module

df = pd.read_csv('dataset.csv')
split_module.build_tree(df, 'hab_lbl', {0:0.1, 1:0.3, 2:0.6})
#print(df)
