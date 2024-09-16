import os 
import json
import re
import copy
from tqdm import tqdm
import random


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
        new_entry["profile_input"] = []
        new_entry["profile_output"] = []
        for his in profile :
            new_entry["profile_txt"].append(his["title"] + " " + his["abstract"])
            new_entry["profile_id"].append(his["id"])
            new_entry["profile_input"].append("Genearate a title for the following abstract of a paper: " + his["abstract"])
            new_entry["profile_output"].append(his["title"])


    if (idx == 2) :
        substr = "description: "
        plc = question.find(substr) + len(substr)
        real_question = question[plc:]
        
        new_entry["question"] = real_question
        new_entry["profile_txt"] = []
        new_entry["profile_id"] = []
        new_entry["profile_input"] = []
        new_entry["profile_output"] = []
        for his in profile :
            new_entry["profile_txt"].append(his["description"])
            new_entry["profile_id"].append(his["id"])
            new_entry["profile_input"].append("Which tag does this movie relate to among the following tags? Just answer with the tag name without further explanation. tags: [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story] description: " + his["description"])
            new_entry["profile_output"].append(his["tag"])
    
    if (idx == 3) :
        substr = "review: "
        plc = question.find(substr) + len(substr)
        real_question = question[plc:]
        
        new_entry["question"] = real_question
        new_entry["profile_txt"] = []
        new_entry["profile_id"] = []
        new_entry["profile_input"] = []
        new_entry["profile_output"] = []
        for his in profile :
            new_entry["profile_txt"].append(his["text"])
            new_entry["profile_id"].append(his["id"])
            new_entry["profile_input"].append("What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: " + his["text"])
            new_entry["profile_output"].append(his["score"])


    if (idx == 4) :
        substr = "article: "
        plc = question.find(substr) + len(substr)
        real_question = question[plc:]
        
        new_entry["question"] = real_question
        new_entry["profile_txt"] = []
        new_entry["profile_id"] = []
        new_entry["profile_input"] = []
        new_entry["profile_output"] = []
        for his in profile :
            new_entry["profile_txt"].append(his["text"] + " " + his["title"])
            new_entry["profile_id"].append(his["id"])
            new_entry["profile_input"].append("Generate a headline for the following article: " + his["text"])
            new_entry["profile_output"].append(his["title"])


    if (idx == 5) :
        substr = "paper: "
        plc = question.find(substr) + len(substr)
        real_question = question[plc:]
        # export http_proxy="http://172.19.56.199:3128"

        new_entry["question"] = real_question
        new_entry["profile_txt"] = []
        new_entry["profile_id"] = []
        new_entry["profile_input"] = []
        new_entry["profile_output"] = []
        for his in profile :
            new_entry["profile_txt"].append(his["title"] + " " + his["abstract"])
            new_entry["profile_id"].append(his["id"])
            new_entry["profile_input"].append("Genearate a title for the following abstract of a paper: " + his["abstract"])
            new_entry["profile_output"].append(his["title"])


    if (idx == 7) :
        substr = "before or after it: "
        plc = question.find(substr) + len(substr)
        real_question = question[plc:]
        
        new_entry["question"] = real_question
        new_entry["profile_txt"] = []
        new_entry["profile_id"] = []
        new_entry["profile_input"] = []
        new_entry["profile_output"] = []
        for his in profile :
            new_entry["profile_txt"].append(his["text"])
            new_entry["profile_id"].append(his["id"])
            new_entry["profile_input"].append("Generate a tweet: ")
            new_entry["profile_output"].append(his["text"])

    return new_entry


remove_chars = '[·’!"\#$%&\'()＃！（）*+,-./:;<=>?\@，：?￥★、…．＞【】［］《》？“”‘’\[\\]^_`{|}~]+'


for idx in range(1, 8) :
    if (idx == 6) :
        continue
    dir_name = "LaMP_time_" + str(idx)
    file_name = os.path.join(dir_name, "train_questions.json")
    print(file_name)
    dataset = json.load(open(file_name))
    os.mkdir(dir_name + "_id")

    datas = []
    
    start = 0
    for entry in dataset :
        new_entry = process(idx, entry)
        his_len = len(new_entry["profile_id"])

        his_id = list(range(start, start+his_len))
        start = start + his_len + 1

        output_entry = {}
        output_entry["input"] = entry["input"]
        output_entry["his_id"] = copy.deepcopy(his_id)
        output_entry["id"] = entry["id"]

        datas += [output_entry]

    
    file_name = os.path.join(dir_name, "train_outputs.json")
    print(file_name)
    outputs = json.load(open(file_name))

    cnt = 0
    for entry in outputs["golds"] :
        while ("output" in datas[cnt]) :
            cnt += 1
        datas[cnt]["output"] = entry["output"]
        assert entry["id"] == datas[cnt]["id"]
        cnt += 1
            
    
    f_out = open(os.path.join(dir_name + "_id", "train_aug_input.json"),"w")
    for data in datas:
        if (len(data["his_id"]) > 2) :
            for times in range(10) :
                del_n = int(len(data["his_id"]) * random.uniform(0, 0.5)) + 1
                new_his = random.sample(data["his_id"], len(data["his_id"]) - del_n)
                new_data = copy.deepcopy(data)
                new_data["his_id"] = new_his
                f_out.write(json.dumps(new_data, ensure_ascii=False) + "\n")
        f_out.write(json.dumps(data, ensure_ascii=False) + "\n")


for idx in range(1, 8) :
    if (idx == 6) :
        continue
    dir_name = "LaMP_time_" + str(idx)
    file_name = os.path.join(dir_name, "dev_questions.json")
    print(file_name)
    dataset = json.load(open(file_name))

    datas = []
    
    start = 0
    for entry in dataset :
        new_entry = process(idx, entry)
        his_len = len(new_entry["profile_id"])

        his_id = list(range(start, start+his_len))
        start = start + his_len + 1

        output_entry = {}
        output_entry["input"] = entry["input"]
        output_entry["his_id"] = copy.deepcopy(his_id)
        output_entry["id"] = entry["id"]

        datas += [output_entry]

    
    file_name = os.path.join(dir_name, "dev_outputs.json")
    print(file_name)
    outputs = json.load(open(file_name))

    cnt = 0
    for entry in outputs["golds"] :
        while ("output" in datas[cnt]) :
            cnt += 1
        datas[cnt]["output"] = entry["output"]
        assert entry["id"] == datas[cnt]["id"]
        cnt += 1
            
    
    f_out = open(os.path.join(dir_name + "_id", "dev_profile.json"),"w")
    for data in datas:
        f_out.write(json.dumps(data, ensure_ascii=False) + "\n")