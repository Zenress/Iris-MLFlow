import json

deployed_model = {
    "name": "dtc_model-853e7",
    "deployed": True,
}

json_object = json.dumps(deployed_model, indent=4)


with open("iris-pipeline/models/deployed_model.json", "w") as file:
    file.write(json_object)
    
with open("iris-pipeline/models/deployed_model.json", 'r') as openfile:
    json_object2 = json.load(openfile)
    
print(json_object2['name'])

json_object2['name'] = "new name"

print(json_object2['name'])
