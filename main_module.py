import numpy as np
import pandas as pd
import split_module

cols = ['hab_lbl', 'P. Composition Class', 'P. Min Mass (EU)', 'P. Mass (EU)', 'P. Radius (EU)']
df = pd.read_csv('dataset.csv', usecols=cols)
split_module.build_tree(df, 'hab_lbl', {0:1.01, 1:5, 2:5})
#print(df)
