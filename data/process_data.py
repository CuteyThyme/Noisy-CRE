import json
import random
file = "data_with_marker_tacred_train.json"
data = json.load(open(file, 'r', encoding='utf-8'))
train_num = 420
#base_relations = ['P156', 'P84', 'P39', 'P276', 'P410', 'P241', 'P177', 'P264']
#base_relations = ['per:cities_of_residence', 'per:other_family', 'org:founded', 'per:origin']
all_relations = list(data.keys())
noise_rate = 0.5

def add_symmetric_noise(data_item, all_relations):
    new_data_item = data_item.copy()
    cur_relation = data_item["relation"]
    random_target = all_relations.copy()
    if cur_relation in random_target:
        random_target.remove(cur_relation) # not change to current relation
    new_data_item["relation"] = random.choice(random_target)
    new_data_item["ori_relation"] = cur_relation
    return new_data_item

# initialize
for relation in data.keys():
    rel_samples = data[relation]
    for data_item in rel_samples:
        data_item["ori_relation"] = data_item["relation"]

# add random noise
for relation in data.keys():
    data_len = len(data[relation])
    flag = False
    flip_list = random.sample(list(range(data_len)),round(data_len*noise_rate))
    remove_items = list()
    for idx, data_item in enumerate(data[relation]):
        if idx in flip_list:
            new_data_item = add_symmetric_noise(data_item,all_relations)
            new_relation = new_data_item["relation"]
            assert new_data_item["relation"] != new_data_item["ori_relation"]
            assert "ori_relation" in new_data_item.keys()
            data[new_relation].append(new_data_item) # add to new relation as noise
            remove_items.append(data_item)
    if flag is False: # not base task
        for remove_item in remove_items:
            data[relation].remove(remove_item)
    for data_item in data[relation]:
        assert "ori_relation" in new_data_item.keys()
cnt = 0
for relation in data.keys():
    rel_samples = data[relation]
    random.shuffle(rel_samples)
    for sample in rel_samples:
        assert sample['relation'] == relation
        assert 'ori_relation' in sample.keys()
    print(sum([sample["relation"]!=sample["ori_relation"] for sample in rel_samples]),len(rel_samples))
    cnt+=len(rel_samples)    
print(cnt)

with open("data_with_marker_tacred_train_noise_{}.json".format(noise_rate), 'w', encoding="utf-8") as f:
    json.dump(data,f,ensure_ascii=False)
