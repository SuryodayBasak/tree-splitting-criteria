import numpy as np
from anytree import Node, RenderTree

def purity (prob, el):
    """
    if len(prob) != len(el):
        print("The dimensions of p and el are unequal. Aborting.")
        return 0
    """
    index = 0
    for key, val in prob.items():
        #print(prob[key])
        #print(key)
        index = index + (prob[key]**el[key])

    return 0.5*(1 - index)


def find_prob(df, cls_lbl):
    n_entities = 0
    n_classes = df[cls_lbl].value_counts().to_dict()
    p_classes = {}

    for key, val in n_classes.items():
        n_entities += val

    for key, val in n_classes.items():
        p_classes[key] = val/n_entities

    return p_classes


def find_prob_part(lbls, cls_lbl):
    n_entities = 0
    n_classes = {}
    p_classes = {}

    for val in set(lbls):
        n_classes[val] = 0

    for val in lbls:
        n_classes[val] += 1
        n_entities += 1

    for key, val in n_classes.items():
        p_classes[key] = val/n_entities

    return p_classes


def check_purity(prob_dict):
    purity_flag = 0
    for key, val in prob_dict.items():
        if val == 1:
            purity_flag = 1
    return purity_flag


def build_tree (df, cls_lbl, el, lvl, name = "just-a-node"):
    #global nodes_idx
    max_split_purity = -999
    op_left = None
    op_right = None
    best_feature = None

    #Finding class-wise probability for this node
    p_classes = find_prob(df, cls_lbl)

    #Finding node purity
    node_purity = purity(p_classes, el)
    print(node_purity)

    #Check if this is a pure node
    if check_purity(p_classes) == 1:
        print("Reached a pure node")
        for key, items in p_classes.items():
            leaf_class = key
        #print(p_classes)
        #print(leaf_class)
        node = Node(name + " leaf -lvl-" + str(lvl), cls = leaf_class)
        return node

    else:
        print("Node is impure")

        #Get the feature names
        feature_names = list(df.columns)
        feature_names.remove(cls_lbl)
        #print(feature_names)

        for feature in feature_names:
            df = df.sort_values(feature)
            #feat_frame = df[[cls_lbl, feature]].copy()
            feat_frame = df[[cls_lbl]].copy()
            feat_order = list(feat_frame[cls_lbl])

            for partition_idx in range(1, len(feat_order)):
                left_part = feat_order[:partition_idx]
                right_part = feat_order[partition_idx:]
                #left_part = df[:partition_idx]
                #right_part = df[partition_idx:]

                prob_left = find_prob_part(left_part, cls_lbl)
                prob_right = find_prob_part(right_part, cls_lbl)

                purity_left = purity(prob_left, el)
                purity_right = purity(prob_right, el)

                #split_purity = 1 - purity_left - purity_right
                split_purity = node_purity - purity_left - purity_right
                if (split_purity > max_split_purity):
                    max_split_purity = split_purity
                    op_left = df[:partition_idx].copy()
                    op_right = df[partition_idx:].copy()
                    #print(op_right)
                    best_feature = feature

            print(max_split_purity, "Done with ", feature,
                        "| Best feat = ", best_feature)

        #Finding feature threshold
        #split_thresh = max(op_left[feature])
        split_thresh = min(op_right[best_feature])

        #Recursively moving ahead
        print ('-----')
        node = Node(name + '-lvl-' + str(lvl),
                    feat = best_feature, thresh = split_thresh)
        print("Building left child node.")
        left_chld = build_tree(op_left, cls_lbl, el, lvl+1, 'left')
        left_chld.parent = node

        print("Building right child node.")
        right_chld = build_tree(op_right, cls_lbl, el, lvl+1, 'right')
        right_chld.parent = node

        return node
