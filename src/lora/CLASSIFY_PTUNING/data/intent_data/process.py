fd1 = open("train.json")
fw = open("new_train.json", "w")
fw2 = open("new_test.json", "w")

import json

data = json.load(fd1)
import random
random.seed(123)

random.shuffle(data)
train_size = int(len(data) * 0.9)
train = data[:train_size]
test = data[train_size:]

new_train = json.dumps(train, ensure_ascii=False, indent=4)
new_test = json.dumps(test, ensure_ascii=False, indent=4)
fw.write(new_train)
fw2.write(new_test)



"""
new_data = []
for item in info:
    new_item = {
        "instruction": item["question_prompt"],
        "input": "",
        "output": item["query"] 
    }
    new_data.append(new_item)

new_data = json.dumps(new_data, ensure_ascii=False, indent=4)
fw.write(new_data)
"""
