import json
import datasets

dataset = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")

data_raw = []
data_train = dataset['eval']

for i,data_i in enumerate(data_train):
    data_temp = {}
    data_temp['instruction'] = data_i['instruction']
    data_temp['dataset'] = data_i['dataset']
    data_raw.append(data_temp)
    pass

# print('New data len \n',len(data_raw))
# with open("alpaca_eval/alpaca_eval_data.json", "w") as fw:
#     json.dump(data_raw, fw, indent=4)

# Saving data to a .jsonl file
with open("alpaca_eval/alpaca_eval_data.jsonl", "w") as file:
    for item in data_raw:
        json_str = json.dumps(item)  # Convert the data item to a JSON-formatted string
        file.write(json_str + "\n")  # Write the JSON string as a separate line in the file

pass