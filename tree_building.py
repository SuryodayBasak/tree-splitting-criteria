import numpy as np
import pandas as pd
from anytree import Node, RenderTree
from anytree.exporter import JsonExporter
import split_module
import json

cols = ['hab_lbl', 'P. Composition Class', 'P. Min Mass (EU)', 'P. Mass (EU)', 'P. Radius (EU)']
#df = pd.read_csv('dataset.csv', usecols=cols)
df = pd.read_csv('dataset.csv')
df = df.drop(df.query('hab_lbl == 0').sample(frac=.95).index)
#df = df.drop(['P. Habitable'], axis = 1)
#df = df.drop(['P. ESI'], axis = 1)
#df = df.drop(['P. HZI'], axis = 1)
#df = df.drop(['P. HZA'], axis = 1)
#df = df.drop(['P. HZC'], axis = 1)
#df = df.drop(['P. HZD'], axis = 1)
print(df)

nodes_list = []
idx = 0
name = 'root'
root = split_module.build_tree(df, 'hab_lbl', {0:1.01, 1:10, 2:10}, 0, name)

#Render the tree
for pre, fill, node in RenderTree(root):
    try:
        print("%s%s::%s::%f" % (pre, node.name, node.feat, node.thresh))
    except: #This would happen only with leaf nodes
        print("%s%s::%d" % (pre, node.name, node.cls))

#print(root.left_chld)

exporter = JsonExporter(indent=2, sort_keys=False)
json_tree = exporter.export(root)
#exporter.write(root, 'json_tree.json')
print(json_tree)

with open('json_tree.json', 'w') as outfile:
    json.dump(json_tree, outfile)
