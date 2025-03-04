import json
import pandas as pd
from func_to_call import parse_all_data, parse_data_with_time
from metrics import ValidatorSimple

data_v1 = parse_all_data('datasets/val_set.json')

data_v2 = parse_data_with_time('datasets/val_set.json')

with open('parsed_tuning.json', 'w', encoding='utf-8') as f:
    json.dump(data_v1, f, ensure_ascii=False)

with open('parsed_dash.json', 'w', encoding='utf-8') as f:
    json.dump(data_v2, f, ensure_ascii=False)

with open('parsed_dash.json', 'r', encoding='utf-8') as f:
    formatted_data = json.load(f)

dataset = pd.DataFrame([],
    columns = ['question', 'answer', 'ground_truth', 'contexts', 'satisfactory', 'time spent'])

count = 0
for item in formatted_data:

    if count == 125:
        continue

    satisfactory = "yes"
    if 'refined_question' in item.keys():
        satisfactory = "no"
    new_data = [item['user_question'],
                item['saiga_answer'],
                item['giga_answer'],
                item['contexts'],
                satisfactory,
                item['response_time']]

    dataset.loc[count] = new_data
    count += 1


vs = ValidatorSimple(neural=True)
res = vs.validate_rag(dataset)
print(res.values())