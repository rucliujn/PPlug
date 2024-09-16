import transformers
import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
import json
import os

class PersonalLLM(nn.Module):
    def __init__(self, llm_model, emb_model, max_input_len, max_new_len, task_id):
        super(PersonalLLM, self).__init__()

        self.llm_model = llm_model
        self.emb_model = emb_model

        self.max_input_len = max_input_len
        self.max_new_len = max_new_len

        self.config = llm_model.config

        self.llm_config = self.llm_model.config
        self.llm_emb_size = self.llm_config.hidden_size

        self.emb_config = self.emb_model.config
        self.emb_emb_size = self.emb_config.hidden_size

        his_train_emb = torch.cat([torch.zeros(1, self.emb_emb_size), torch.load("../bge_emb/task_" + str(task_id) + "_train_bge.emb")], 0)
        self.his_train_emb_table = nn.Embedding(his_train_emb.size()[0], self.emb_emb_size)
        self.his_train_emb_table.weight = nn.Parameter(his_train_emb)
        for _, param in self.his_train_emb_table.named_parameters():
            param.requires_grad = False

        his_dev_emb = torch.cat([torch.zeros(1, self.emb_emb_size), torch.load("../bge_emb/task_" + str(task_id) + "_dev_bge.emb")], 0)
        self.his_dev_emb_table = nn.Embedding(his_dev_emb.size()[0], self.emb_emb_size)
        self.his_dev_emb_table.weight = nn.Parameter(his_dev_emb)
        for _, param in self.his_dev_emb_table.named_parameters():
            param.requires_grad = False

        # self.inst_token = nn.Parameter(torch.nn.functional.normalize(torch.rand(self.emb_emb_size), p=2, dim=0),requires_grad=True)
        self.inst_token = nn.Parameter(torch.rand(self.emb_emb_size),requires_grad=True)
        for _, param in self.llm_model.named_parameters():
            param.requires_grad = False

        self.mult_k = 1
        self.align_mlp_inst = nn.Sequential(nn.Linear(self.emb_emb_size, self.llm_emb_size * self.mult_k), nn.GELU(), nn.Linear(self.llm_emb_size * self.mult_k, self.llm_emb_size * self.mult_k))
        self.align_mlp = nn.Sequential(nn.Linear(self.emb_emb_size, self.llm_emb_size * self.mult_k), nn.GELU(), nn.Linear(self.llm_emb_size * self.mult_k, self.llm_emb_size * self.mult_k))


    def mean_pooling(self, token_embeddings, mask) :
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings


    def obtain_task_emb(self, emb_input_ids, emb_attention_mask, emb_token_type_ids) :
        task_inputs = {'input_ids': emb_input_ids, 'attention_mask': emb_attention_mask, 'token_type_ids': emb_token_type_ids}
        task_outputs = self.emb_model(**task_inputs)
        task_embeddings = torch.nn.functional.normalize(task_outputs[0][:,0], p=2, dim=1)
        return task_embeddings

    def obtain_profile_emb(self, his_id, task_embs) :
        
        his_mask = torch.eq(his_id, 0)
        bsz = his_mask.size()[0]
        his_mask = his_mask.repeat(1, self.mult_k).view(bsz * self.mult_k, -1)
        

        if (self.training) :
            his_embs = self.his_train_emb_table(his_id)
        else :
            his_embs = self.his_dev_emb_table(his_id)
        
        his_embs_align = self.align_mlp(his_embs).view(bsz * self.mult_k, -1, self.llm_emb_size)


        his_weight = torch.bmm(his_embs, task_embs.unsqueeze(-1)) #/ 0.5
        his_weight = his_weight.masked_fill(his_mask.unsqueeze(-1), -torch.inf)
        his_weight = torch.nn.functional.softmax(his_weight, dim=1)
        his_weight = his_weight.to(his_embs.dtype)
        profile_embs = torch.bmm(torch.transpose(his_embs_align, 1, 2), his_weight).squeeze(-1)

        return profile_embs



    def forward(self, llm_input_ids, llm_attention_mask, labels, emb_input_ids, emb_attention_mask, emb_token_type_ids, his_id):
        
        task_embs = self.obtain_task_emb(emb_input_ids, emb_attention_mask, emb_token_type_ids)
        profile_embs = self.obtain_profile_emb(his_id, task_embs) * 539.9738
        inst_embs = self.align_mlp_inst(self.inst_token) / 7.0 * 539.9738

        input_embs = self.llm_model.get_input_embeddings()(llm_input_ids)

        input_embs[llm_input_ids == self.llm_model.vocab_size-1] = profile_embs.view(-1, self.llm_emb_size)#bsz, hidden
        input_embs[llm_input_ids == self.llm_model.vocab_size-2] = inst_embs

        if self.training:
            reader_output = self.llm_model(
                inputs_embeds=input_embs,
                attention_mask=llm_attention_mask.long(),
                labels=labels.long(),
                use_cache=False,
                return_dict=True
            )
            loss = reader_output.loss
            return SequenceClassifierOutput(
                    loss=loss
                )
        else:
            reader_output = self.llm_model.generate(
                inputs_embeds=input_embs,
                attention_mask=llm_attention_mask,
                max_new_tokens=self.max_new_len,
                num_beams=4,
                num_return_sequences=1,
                return_dict_in_generate=True,
                do_sample=False, 
                return_dict=True
            )
            input_len = llm_input_ids.size(1)
            reader_output = reader_output['sequences']
            loss = torch.FloatTensor([0.0]).to(llm_input_ids.device)
            return [loss, reader_output]