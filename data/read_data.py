import json
import random
file = "data_with_marker_tacred.json"
data = json.load(open(file, 'r', encoding='utf-8'))
train_data = {}
val_data = {}
test_data = {}
for relation in data.keys():
    count = 0
    count1 = 0
    rel_samples = data[relation]
    random.shuffle(rel_samples)
    print(len(rel_samples))
    for i, sample in enumerate(rel_samples):
        if i < len(rel_samples) // 5 and count <= 40:
            count += 1
            if relation not in test_data.keys():
                test_data[relation] = list()
            test_data[relation].append(sample)
        else:
            count1 += 1
            if relation not in train_data.keys():
                    train_data[relation] = list()
            train_data[relation].append(sample)  
            if count1 >= 320:
                break      
for relation in test_data.keys():
    print(len(test_data[relation]))
with open(file.replace(".json","_test.json"), 'w', encoding="utf-8") as f:
    json.dump(test_data,f,ensure_ascii=False)