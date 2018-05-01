import numpy as np
import pandas as pd
from anytree import Node, RenderTree
import split_module

cols = ['hab_lbl', 'P. Composition Class', 'P. Min Mass (EU)', 'P. Mass (EU)', 'P. Radius (EU)']
df = pd.read_csv('dataset.csv', usecols=cols)

nodes_list = []
idx = 0
root = None
tree = split_module.build_tree(df, 'hab_lbl', {0:1.01, 1:5, 2:5})
print("Tree = ", tree)
print(RenderTree(tree))
#print(RenderTree(nodes_list[0]))
#print(df)
