from rank_bm25 import BM25Okapi
import os 
import json
import re
from transformers import AutoTokenizer
import argparse
from tqdm import tqdm


import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('./bge-base-en-v1.5')
model = AutoModel.from_pretrained('./bge-base-en-v1.5')



#def rank_corpus() :
def process(idx, entry) :
    question = entry["input"]
    profile = entry["profile"]

    new_entry = {}
    
    if (idx == 1) :
        substr = "title \""
        plc = question.find(substr) + len(substr)
        substr2 = "\","
        plc2 = question.find(substr2, plc)
        real_question = question[plc:plc2]
        
        new_entry["question"] = real_question
        new_entry["profile_txt"] = []
        new_entry["profile_id"] = []
        for his in profile :
            new_entry["profile_txt"].append(his["title"] + " " + his["abstract"])
            new_entry["profile_id"].append(his["id"])
    
    if (idx == 2) :
        substr = "description: "
        plc = question.find(substr) + len(substr)
        real_question = question[plc:]
        
        new_entry["question"] = real_question
        new_entry["profile_txt"] = []
        new_entry["profile_id"] = []
        for his in profile :
            new_entry["profile_txt"].append(his["description"])
            new_entry["profile_id"].append(his["id"])
    
    if (idx == 3) :
        substr = "review: "
        plc = question.find(substr) + len(substr)
        real_question = question[plc:]
        
        new_entry["question"] = real_question
        new_entry["profile_txt"] = []
        new_entry["profile_id"] = []
        for his in profile :
            new_entry["profile_txt"].append(his["text"])
            new_entry["profile_id"].append(his["id"])

    if (idx == 4) :
        substr = "article: "
        plc = question.find(substr) + len(substr)
        real_question = question[plc:]
        
        new_entry["question"] = real_question
        new_entry["profile_txt"] = []
        new_entry["profile_id"] = []
        for his in profile :
            new_entry["profile_txt"].append(his["text"] + " " + his["title"])
            new_entry["profile_id"].append(his["id"])

    if (idx == 5) :
        substr = "paper: "
        plc = question.find(substr) + len(substr)
        real_question = question[plc:]
        
        new_entry["question"] = real_question
        new_entry["profile_txt"] = []
        new_entry["profile_id"] = []
        for his in profile :
            new_entry["profile_txt"].append(his["title"] + " " + his["abstract"])
            new_entry["profile_id"].append(his["id"])

    if (idx == 7) :
        substr = "before or after it: "
        plc = question.find(substr) + len(substr)
        real_question = question[plc:]
        
        new_entry["question"] = real_question
        new_entry["profile_txt"] = []
        new_entry["profile_id"] = []
        for his in profile :
            new_entry["profile_txt"].append(his["text"])
            new_entry["profile_id"].append(his["id"])

    return new_entry


remove_chars = '[·’!"\#$%&\'()＃！（）*+,-./:;<=>?\@，：?￥★、…．＞【】［］《》？“”‘’\[\\]^_`{|}~]+'

def sort(corpus_embedding, query_embedding, his_len, new_entry) :
    
    all_list = []
    for i in range(his_len) :
        all_list.append((float(corpus_embedding[i] @ query_embedding), new_entry["profile_txt"][i], new_entry["profile_id"][i]))
    all_list.sort(key = lambda x : -x[0])

    return all_list

@torch.no_grad()
def get_embedding(sentences) :
    # Apply tokenizer
    print(len(sentences))
    #sentences = sentences[:2048]
    batch_size = 1024
    model.cuda(0)

    def mean_pooling(token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings
    
    embeddings_list = []
    for i in tqdm(range(0, len(sentences), batch_size)) :
        inputs = tokenizer(sentences[i :i + batch_size], padding=True, truncation=True, return_tensors='pt')
        for key in inputs :
            inputs[key] = inputs[key].cuda(0)
        outputs = model(**inputs)
        embeddings_list.append(torch.nn.functional.normalize(outputs[0][:,0], p=2, dim=1).detach().cpu())

        del inputs
        del outputs
        torch.cuda.empty_cache()
        print(i)
    embeddings = torch.cat(embeddings_list, 0)

    return embeddings

os.mkdir("./bge_emb")

for idx in range(1, 8) :
    if (idx == 6) :
        continue
    dir_name = "LaMP_time_" + str(idx)

    file_name = os.path.join(dir_name, "train_questions.json")
    print(file_name)
    dataset = json.load(open(file_name))

    datas = []
    
    all_txt_list = []
    for entry in dataset :
        new_entry = process(idx, entry)
        all_txt_list.extend(new_entry["profile_txt"] + [new_entry["question"]])
    corpus_embedding_all_list = get_embedding(all_txt_list)
    torch.save(corpus_embedding_all_list, "./bge_emb/task_"+str(idx)+"_train_bge.emb")


    file_name = os.path.join(dir_name, "dev_questions.json")
    print(file_name)
    dataset = json.load(open(file_name))

    datas = []
    
    all_txt_list = []
    for entry in dataset :
        new_entry = process(idx, entry)
        all_txt_list.extend(new_entry["profile_txt"] + [new_entry["question"]])
    corpus_embedding_all_list = get_embedding(all_txt_list)
    torch.save(corpus_embedding_all_list, "./bge_emb/task_"+str(idx)+"_dev_bge.emb")