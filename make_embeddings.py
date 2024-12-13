import os
import math
import json
import torch
from tqdm import tqdm
import numpy as np
from datetime import datetime, timedelta
from transformers import AutoModel, AutoTokenizer

path = 'path-to-data/'
companies = sorted(os.listdir(path))

start_date = datetime.strptime('2014-01-01', "%Y-%m-%d")
end_date = datetime.strptime('2016-01-01', "%Y-%m-%d")
delta = end_date - start_date

all_data = []
for i in range(delta.days + 1):
    data = {}
    day = str((start_date + timedelta(days=i)).date())
    for company in companies:
        co_days = os.listdir(path + company)
        if day in co_days:
            with open(path + company + '/' + day) as f:
                tweets = f.read().strip().split('\n')
                data[company] = [json.loads(tweet)['text'] for tweet in tweets]
        else:
            data[company] = []
    all_data += [data]


# name = 'ProsusAI/finbert'
name = 'google-bert/bert-base-uncased'
bert = AutoModel.from_pretrained(name).cuda()
tokenizer = AutoTokenizer.from_pretrained(name)


bs = 128
all_embeddings = []
with torch.no_grad():
    for data in tqdm(all_data):
        embeddings = []
        for company in companies:
            if len(data[company]) > 0:
                emb = []
                n = len(data[company])
                for i in range(math.ceil(n / bs)):
                    inputs = tokenizer(
                        data[company][i*bs:(i+1)*bs], 
                        return_tensors='pt', 
                        padding=True, 
                        truncation=True
                    )
                    input_ids = inputs['input_ids'].cuda()
                    attention_mask = inputs['attention_mask'].cuda()
                    emb += [
                        bert(
                            input_ids, 
                            attention_mask=attention_mask
                        ).last_hidden_state.mean(dim=1).detach().cpu()
                    ]
                emb = torch.cat(emb, dim=0).mean(dim=0)
            else:
                emb = torch.zeros(768)
            embeddings += [emb]
        all_embeddings += [torch.stack(embeddings)]
all_embeddings = torch.stack(all_embeddings)

np.save(
    f'{path}/{name}-embeddings.npy', 
    all_embeddings.transpose(0, 1).numpy()
)
