import argparse
import numpy as np
import torch
import logging
import torch.nn as nn
import torch.nn.utils as utils
import torch.nn.functional as F
import transformers
import evaluate
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from transformers import AutoModelForCausalLM, T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import AutoTokenizer, AutoModel

from PersonalDataset_profile import PersonalDataset
from tqdm import tqdm
import os
from utils import *
from dataclasses import dataclass, field
from ModelForPer import PersonalLLM
from rouge_score import rouge_scorer, scoring

@dataclass
class ModelArguments:
    model_path: str = field(default="../flant5-base/", metadata={"help": "Path to the pretrain model."})
    emb_model_path: str = field(default="../contriever/", metadata={"help": "Path to the pretrain model."})

@dataclass
class DataArguments:
    train_file: str = field(default="../LaMP_time_4/train.json", metadata={"help": "Path to the train data."})
    dev_file: str = field(default="../LaMP_time_4/dev.json", metadata={"help": "Path to the dev data."})
    max_input_len: int = field(default=640, metadata={"help": "The max length of input"})
    max_his_len: int = field(default=512, metadata={"help": "The max length of history"})
    max_new_len: int = field(default=128, metadata={"help": "The max length of output"})
    
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def compute_metrics_classification_acc_f1(eval_preds) :
    preds, labels = eval_preds
    preds = [[max(0, idx) for idx in x] for x in preds]
    labels = [[max(0, idx) for idx in x] for x in labels]
    predictions = llm_tokenizer.batch_decode(preds, skip_special_tokens=True)
    references = llm_tokenizer.batch_decode(labels, skip_special_tokens=True)    

    all_labels = ["sci-fi", "based on a book", "comedy", "action", "twist ending", "dystopia", "dark comedy", "classic", "psychology", "fantasy", "romance", "thought-provoking", "social commentary", "violence", "true story"]

    def create_mapping(x):
        try:
            return all_labels.index(x)
        except:
            print(x)
            return -1

    predictions = [create_mapping(y) for y in predictions]
    references = [create_mapping(y) for y in references]

    result_acc = float(accuracy_score(references, predictions, normalize=True, sample_weight=None))
    result_f1 = float(f1_score(references, predictions, labels=list(range(len(all_labels))), pos_label=1, average="macro", sample_weight=None))
    result = {"accuracy" : result_acc, "f1" : result_f1}
    return result


def compute_metrics_classification_acc(eval_preds) :
    preds, labels = eval_preds
    preds = [[max(0, idx) for idx in x] for x in preds]
    labels = [[max(0, idx) for idx in x] for x in labels]
    predictions = llm_tokenizer.batch_decode(preds, skip_special_tokens=True)
    references = llm_tokenizer.batch_decode(labels, skip_special_tokens=True)


    all_labels = ["[1]","[2]"]

    def create_mapping(x):
        try:
            return all_labels.index(x)
        except:
            print(x)
            return -1

    predictions = [create_mapping(y) for y in predictions]
    references = [create_mapping(y) for y in references]

    result_acc = float(accuracy_score(references, predictions, normalize=True, sample_weight=None))
    result = {"accuracy" : result_acc}
    return result

def compute_metrics_classification(eval_preds) :
    preds, labels = eval_preds
    preds = [[max(0, idx) for idx in x] for x in preds]
    labels = [[max(0, idx) for idx in x] for x in labels]
    predictions = llm_tokenizer.batch_decode(preds, skip_special_tokens=True)
    references = llm_tokenizer.batch_decode(labels, skip_special_tokens=True)

    def create_mapping(x):
        try:
            return float(x)
        except:
            for z in x :
                if (z.isnumeric()) :
                    return float(int(z)) 
            print(x)
            return 1.0

    predictions = [create_mapping(y) for y in predictions]
    references = [create_mapping(y) for y in references]


    mae = 0.0
    rmse = 0.0
    
    length = len(preds)
    for i in range(length) :
        mae += math.fabs(predictions[i] - references[i]) / (length * 1.0)
        rmse += (predictions[i] - references[i]) ** 2 / (length * 1.0)

    rmse = math.sqrt(rmse)

    return {"mae": mae, "rmse": rmse}


def compute_metrics(eval_preds) :
    preds, labels = eval_preds

    preds = [[max(0, idx) for idx in x] for x in preds]
    labels = [[max(0, idx) for idx in x] for x in labels]
    predictions = llm_tokenizer.batch_decode(preds, skip_special_tokens=True)
    references = llm_tokenizer.batch_decode(labels, skip_special_tokens=True)

    
    rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

    scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=False)
    aggregator = scoring.BootstrapAggregator()


    overall_metric = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}
    for ref, pred in zip(references, predictions):
        score = scorer.score(ref, pred)
        if (local_rank == 0) :
            metric = {}
            for key in score :
                metric[key] = score[key].fmeasure
                overall_metric[key] += metric[key] / len(preds)
            #for key in metric :
            #    f_out.write(key + " " + str(metric[key]) + "\n")
        aggregator.add_scores(score)
    
    result = aggregator.aggregate()
    for key in result:
        result[key] = result[key].mid.fmeasure

    if (local_rank == 0) :
        print(overall_metric)
        print(result)
    return result

def train_model(model_args, data_args, training_args):
    
    train_dataset = PersonalDataset(data_args.train_file, data_args.max_input_len, data_args.max_new_len, data_args.max_his_len, llm_tokenizer, emb_tokenizer)
    eval_dataset = PersonalDataset(data_args.dev_file, data_args.max_input_len, data_args.max_new_len, data_args.max_his_len, llm_tokenizer, emb_tokenizer)

    #print(len(eval_dataset))
    #llm_model = AutoModelForCausalLM.from_pretrained(model_args.model_path)
    llm_model = T5ForConditionalGeneration.from_pretrained(model_args.model_path)
    llm_model.resize_token_embeddings(len(llm_tokenizer))
    emb_model = AutoModel.from_pretrained(model_args.emb_model_path)

    model = PersonalLLM(llm_model = llm_model, emb_model = emb_model, max_input_len = data_args.max_input_len, max_new_len = data_args.max_new_len, task_id=int(training_args.output_dir[-1]))

    if ("time_1" in data_args.dev_file) :
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics_classification_acc,
        )
    elif ("time_2" in data_args.dev_file) :
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics_classification_acc_f1,
        )
    elif ("time_3" in data_args.dev_file) :
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics_classification,
        )
    else :
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

    trainer.train()
    outputs = trainer.evaluate()
    print(outputs)


if __name__ == '__main__':
    set_seed()
    global llm_tokenizer, emb_tokenizer, f_out, local_rank

    local_rank = int(os.environ["LOCAL_RANK"])
    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    llm_tokenizer = AutoTokenizer.from_pretrained(model_args.model_path)
    llm_tokenizer.add_special_tokens({"additional_special_tokens":["[INST_PER_TOKEN]","[SPC_PER_TOKEN]"]})
    emb_tokenizer = AutoTokenizer.from_pretrained(model_args.emb_model_path)

    train_model(model_args, data_args, training_args)