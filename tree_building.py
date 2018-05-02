import numpy as np
import pandas as pd
from anytree import Node, RenderTree
from anytree.exporter import JsonExporter, DictExporter
import split_module
import json

#cols = ['hab_lbl', 'P. Composition Class', 'P. Min Mass (EU)', 'P. Mass (EU)', 'P. Radius (EU)']
#df = pd.read_csv('dataset.csv', usecols=cols)

for i in range (1, 21):
    df = pd.read_csv('dataset.csv')
    df = df.drop(df.query('hab_lbl == 0').sample(frac=.95).index)

    #Splitting into training and test sets
    train_test_split = np.random.rand(len(df)) < 0.8
    train = df[train_test_split]
    test = df[~train_test_split]
    #print(train)

    nodes_list = []
    idx = 0
    name = 'root'
    root = split_module.build_tree(train, 'hab_lbl',
                                    {0:1.01, 1:25, 2:25}, 0, name)

    print("----------------------------")
    print("TREE: " + str(i))
    #Render the tree
    for pre, fill, node in RenderTree(root):
        try:
            print("%s%s::%s::%f" % (pre, node.name, node.feat, node.thresh))
        except: #This would happen only with leaf nodes
            print("%s%s::%d" % (pre, node.name, node.cls))

    #print(root.left_chld)

    #Writeout JSON object
    exporter = JsonExporter(indent=2, sort_keys=False)
    json_tree = exporter.export(root)
    #exporter.write(root, 'json_tree.json')
    #print(json_tree)

    with open('trees/json_tree' + str(i) +'.txt', 'w') as outfile:
        json.dump(json_tree, outfile)

    #Convert tree to dict
    exporter = DictExporter()
    json_dict = exporter.export(root)

    count = 0
    for index, row in test.iterrows():
        count = count + 1
        print("CLASSIFYING SAMPLE: ", count)
        #print(row['P. Zone Class'])
        #Tree traversal for classifying test samples
        current_node = json_dict
        #print(current_node['name'])
        while True:
            if 'leaf' in current_node['name']:
                print("True Class = ", row['hab_lbl'],
                        "Predicted Class = ", current_node['cls'])
                break

            else:
                cur_feat = current_node['feat']
                cur_thresh = current_node['thresh']
                #print(cur_feat, cur_thresh)

                if row[cur_feat] < cur_thresh:
                    #pugglewuggle
                    if 'left' in current_node['children'][0]['name']:
                        current_node = current_node['children'][0]
                    else:
                        current_node = current_node['children'][1]
                else:
                    if 'right' in current_node['children'][0]['name']:
                        current_node = current_node['children'][0]
                    else:
                        current_node = current_node['children'][1]

            #print(current_node['children'])
            #print(current_node['children'][0])
            #print('')
            #print(current_node['children'][1])

    #print(datastore)
    #json.loads(json_raw)
    #Tree traversal
    #current_node = root
    #while 'leaf' not in current_node:
    #print(json_tree['children'])
