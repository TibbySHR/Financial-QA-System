fd1 = open("train.json")
fw = open("new_train.json", "w")
fw2 = open("new_test.json", "w")

import json

raw = json.load(fd1)
import random
random.seed(123)

data = []
for item in raw:
    new_item = {
        "instruction": item["question_prompt"],
        "input": "",
        "output": item["query"] 
    }
    data.append(new_item)

random.shuffle(data)
train_size = int(len(data) * 0.9)
train = data[:train_size]
test = data[train_size:]

new_train = json.dumps(train, ensure_ascii=False, indent=4)
new_test = json.dumps(test, ensure_ascii=False, indent=4)
fw.write(new_train)
fw2.write(new_test)

