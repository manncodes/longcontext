import os
import pandas as pd
import json
from tqdm import tqdm

import random

random.seed(42)

dataset_dir = 'fineweb-edu-dedup' # Read a large number of documents from the dataset and replace them with the paths of your dataset

input_files = os.listdir(dataset_dir)

random.shuffle(input_files)

all_data_lines =[]

for file in input_files:

    if file.endswith('.jsonl'):

        input_file = f'{dataset_dir}/{file}'
        data_lines = open(input_file).readlines()
        all_data_lines.extend(data_lines)
        if len(all_data_lines) >= 400000000:
            break

root_file = 'longmagpie/fineweb-edu-demo_output.jsonl'


max_doc=10


w_file = open(f'longmagpie/fineweb-edu-demo-multidoc{max_doc}_demo.jsonl', 'w+')

lines = open(root_file).readlines()

random.shuffle(lines)

for line in tqdm(lines):

    row = json.loads(line)

    ls_new = []

    if 'query' not in row.keys():

        continue

    selected_docs = [row['context']]

    num_docs = random.randint(1, max_doc)

    random_docs = random.sample(all_data_lines, num_docs - 1)

    selected_docs.extend([json.loads(doc)['text'] for doc in random_docs])

    random.shuffle(selected_docs)

    context = "\n\n===Document Separator===\n\n".join(selected_docs)

    ls_new.append({'content':context + row['query'], 'role':'user'})
    
    ls_new.append({'content': row['answer'], 'role':'assistant' })

    w_file.write(json.dumps(ls_new)+'\n')

w_file.close()