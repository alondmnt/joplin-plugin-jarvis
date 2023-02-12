import json
import os
import pandas as pd

prompt_dict = {}
for root, dirs, files in os.walk('assets/'):
    for file in files:
        if file.endswith('.csv'):
            ptype = file.split('/')[-1].split('.')[0]
            df = pd.read_csv(root + file)
            prompt_dict[ptype] = df.set_index('label')['value'].to_dict()

json.dump(prompt_dict, open('assets/prompts.json', 'w'), indent=2)
