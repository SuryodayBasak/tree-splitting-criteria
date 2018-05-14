import numpy as np
import pandas as pd
from anytree import Node, RenderTree
from anytree.exporter import JsonExporter, DictExporter
import split_module
import json

#cols = ['hab_lbl', 'P. Composition Class', 'P. Min Mass (EU)', 'P. Mass (EU)', 'P. Radius (EU)']
cols = ['hab_lbl', 'P. Mass (EU)', 'P. Radius (EU)']
#df = pd.read_csv('dataset.csv', usecols=cols)

conf_mat = np.zeros((3, 3))

for i in range (1, 10):
    df = pd.read_csv('dataset.csv')
    #df = pd.read_csv('dataset.csv', usecols=cols)
    df = df.drop(df.query('hab_lbl == 0').sample(frac=.5).index)

    #Splitting into training and test sets
    #train_test_split = np.random.rand(len(df)) < 0.8
    mask_0 = np.random.rand(len(df.loc[df['hab_lbl'] == 0])) < 0.8
    mask_1 = np.random.rand(len(df.loc[df['hab_lbl'] == 1])) < 0.8
    mask_2 = np.random.rand(len(df.loc[df['hab_lbl'] == 2])) < 0.8
    train_test_split = np.concatenate((mask_0, mask_1, mask_2), axis = 0)
    train = df[train_test_split]
    test = df[~train_test_split]
    #print(train)

    nodes_list = []
    idx = 0
    name = 'root'
    root = split_module.build_tree(train, 'hab_lbl',
                                    {0:8, 1:3, 2:11}, 0, name)

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
        #print("CLASSIFYING SAMPLE: ", count)
        #print(row['P. Zone Class'])
        #Tree traversal for classifying test samples
        current_node = json_dict
        #print(current_node['name'])
        while True:
            if 'leaf' in current_node['name']:
                print("True Class = ", row['hab_lbl'],
                        "Predicted Class = ", current_node['cls'])
                conf_mat[int(row['hab_lbl']), int(current_node['cls'])] += 1
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

#Generating final confusion matrix
r, c = np.shape(conf_mat)
for i in range(r):
    r_sum = sum(conf_mat[i,:])
    for j in range(c):
        conf_mat[i, j] = (conf_mat[i, j]/r_sum) * 100

print(conf_mat)
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
