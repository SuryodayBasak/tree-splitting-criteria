import json

with open('json_tree.json', 'r') as f:
    datastore = json.load(f)

print(datastore)
