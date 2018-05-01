def impurity (prob, el):
    if len(prob) != len(el):
        print("The dimensions of p and el are unequal. Aborting.")
        return 0

    index = 1
    for key, val in prob.items():
        index = index*(prob[key]**el[key])

    return index

def find_prob(df, cls_lbl):
    n_entities = 0
    n_classes = df[cls_lbl].value_counts().to_dict()
    p_classes = {}

    for key, val in n_classes.items():
        n_entities += val

    for key, val in n_classes.items():
        p_classes[key] = val/n_entities
        
    return p_classes

def build_tree (df, cls_lbl, el):
    max_gain = -999
    op_left = None
    op_right = None
    best_feature = None

    """
    #Find the unique class labels
    lbls = df[cls_lbl].unique()
    #Dictionary to keep track of entities per class
    n_classes = df[cls_lbl].value_counts().to_dict()

    #Dictionary to find probabilities
    n_entities = 0

    p_classes = {}
    for key, val in n_classes.items():
        n_entities += val
    for key, val in n_classes.items():
        p_classes[key] = val/n_entities
    """
    p_classes = find_prob(df, cls_lbl)

    #Check if this is a pure node
    purity_flag = 0
    for key, val in p_classes.items():
        if val == 1:
            purity_flag = 1

    if purity_flag == 1:
        print("Reached a pure node")

    #Get the feature names
    feature_names = list(df.columns)
    feature_names.remove(cls_lbl)
    #print(feature_names)

    #Finding node impurity
    node_impurity = impurity(p_classes, el)
    print(node_impurity)

    for feature in feature_names:
        df = df.sort_values(feature)
        feat_frame = df[[cls_lbl, feature]].copy()
        #print(feat_frame)
