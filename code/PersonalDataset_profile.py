from torch.utils.data import Dataset
import pickle
import linecache
import json
import numpy as np
from tqdm import tqdm
import os

import copy

class PersonalDataset(Dataset):
    def __init__(self, path, max_input_len, max_new_len, max_his_len, llm_tokenizer, emb_tokenizer):

        self.path = path

        self.max_input_len = max_input_len
        self.max_new_len = max_new_len
        self.max_his_len = max_his_len
        self.max_all_len = max_input_len + max_new_len
        self.llm_tokenizer = llm_tokenizer
        self.emb_tokenizer = emb_tokenizer
        
                        
        self.total_len = len(open(self.path).readlines())

    
    def __len__(self):
        return self.total_len
    
    def pad_his(self, his_id) :
        his_id = [x + 1 for x in his_id[-self.max_his_len:]] + [0] * (self.max_his_len - len(his_id))
        return his_id

    def parse_data(self, line):
        data = json.loads(line)
        input_str = copy.deepcopy(data["input"])

        input_str = self.llm_tokenizer.decode(self.llm_tokenizer.encode(input_str)[:self.max_input_len][:-1])
        output_str = data["output"]

        his_id = self.pad_his(data["his_id"])

        return input_str, output_str, his_id
    
    def process_qry(self, query) :
        if ("1" in self.path) :
            substr = "title \""
            plc = query.find(substr) + len(substr)
            substr2 = "\","
            plc2 = query.find(substr2, plc)
            real_query = query[plc:plc2]        
        if ("2" in self.path) :
            substr = "description: "
            plc = query.find(substr) + len(substr)
            real_query = query[plc:]
        if ("3" in self.path) :
            substr = "review: "
            plc = query.find(substr) + len(substr)
            real_query = query[plc:]
        if ("4" in self.path) :
            substr = "article: "
            plc = query.find(substr) + len(substr)
            real_query = query[plc:]
        if ("5" in self.path) :
            substr = "paper: "
            plc = query.find(substr) + len(substr)
            real_query = query[plc:]
        if ("7" in self.path) :
            substr = "before or after it: "
            plc = query.find(substr) + len(substr)
            real_query = query[plc:]
    
        print(real_query)
        return real_query

        
    def __getitem__(self, idx):
        


        query, output_str, his_id = self.parse_data(linecache.getline(self.path, idx+1))
                
        
        inputs = self.llm_tokenizer("[INST_PER_TOKEN][SPC_PER_TOKEN]"+query, max_length=self.max_input_len+10, padding='max_length', truncation=True)
        targets = self.llm_tokenizer(output_str, max_length=self.max_new_len, padding='max_length', truncation=True)
        emb_inputs = self.emb_tokenizer("Represent this sentence for searching relevant passages:"+self.process_qry(query), max_length=self.max_input_len+20, padding='max_length', truncation=True)

        labels = [x if x != self.llm_tokenizer.pad_token_id else -100 for x in targets["input_ids"]]
        return {
            'llm_input_ids': np.array(inputs['input_ids'], dtype=np.int32),
            'llm_attention_mask': np.array(inputs['attention_mask'], dtype=np.int32),
            'labels': np.array(labels, dtype=np.int32),
            'emb_input_ids': np.array(emb_inputs['input_ids'], dtype=np.int32),
            'emb_attention_mask': np.array(emb_inputs['attention_mask'], dtype=np.int32),
            'emb_token_type_ids': np.array(emb_inputs['token_type_ids'], dtype=np.int32),
            'his_id': np.array(his_id, dtype=np.int32)
        }