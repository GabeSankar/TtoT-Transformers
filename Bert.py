from transformers import BertModel, BertConfig, BertTokenizer, TrainingArguments, Trainer, get_scheduler, AutoTokenizer
from datasets import load_dataset, load_metric
from torch.optim import AdamW
from torch.utils.data import DataLoader
import os
from datasets.combine import concatenate_datasets
import numpy as np
import json
import torch
from tqdm.auto import tqdm
import pandas as pd


class Bert:
    # WordPiece is a bool same with use only lowercase should be(true, true) for first test
    def __init__(self, vocabFile, wordPiece, onlyLowercase):
        #might remove
        # del self.model
        # del self.trainer
        #self.DatasetReparser("OldDatasets/train.json", "train.json")
        print("done reparsing train.json")
        #self.DatasetReparser("OldDatasets/test.json", "test.json")
        print("done reparsing test.json")
        torch.cuda.empty_cache()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.configuration = BertConfig()

        self.model = BertModel(self.configuration)
        self.model.to(torch.device("cpu"))
        #self.tokenizer = BertTokenizer( do_lower_case=onlyLowercase)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")
        data_collator = DataCollatorWithPadding(self.tokenizer)
        datasetFiles = {"train": "train.json", "test": "test.json"}
        dataset = load_dataset("json", data_files=datasetFiles)
        print(dataset)
        #tokenized_dataset_train = pa.table({})
        #tokenized_dataset_test = pa.table({})
    
        #for element in dataset["train"]:
            # print(type(element["text"]))    #might get index["input_ids"]
           # dataset["train"][i]["text"] = self.tokenizer_function(element["text"])
           # dataset["train"][i]["input"] = self.tokenizer_function(element["input"])
           # train_bar.update(1)
           # i=i+1
        #i=0
        #for element in dataset["test"]:
          #  dataset["test"][i]["text"] = self.tokenizer_function(element["text"])
           # dataset["test"][i]["input"] = self.tokenizer_function(element["input"])
            #i=i+1

        tokenized_dataset = dataset.map(self.tokenizer_function,batched=True)

        train_args = TrainingArguments(output_dir="Trainer", evaluation_strategy="epoch",learning_rate=2e-5,weight_decay=0.01)

        self.metric = load_metric("accuracy")
                                                                  
        #self.trainer = Trainer(model=self.model, args=train_args, train_dataset=train_dataset, eval_dataset=test_dataset, compute_metrics=self.metric)
        #tokenized_dataset.remove_columns(["input","text","input_ids","token_type_ids","attention_mask"])
        #tokenized_dataset.set_format("torch")
        small_train_dataset = tokenized_dataset["train"].shuffle(seed=42)
        small_test_dataset = tokenized_dataset["test"].shuffle(seed=42)
        self.trainer = Trainer(model=self.model,args=train_args,train_dataset=small_train_dataset,eval_dataset=small_test_dataset,compute_metrics=self.metric,data_collator=data_collator)
        print(small_train_dataset)
        self.train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8,collate_fn=data_collator)

        self.eval_dataloader = DataLoader(small_test_dataset, batch_size=8,collate_fn=data_collator)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(torch.cuda.is_available())
    def train(self, num_epochs):
        self.trainer.train()
    """
        num_training_steps = num_epochs * len(self.train_dataloader)
        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
        progress_bar = tqdm(range(num_training_steps))

        self.model.to(self.device)
        self.model.train()
        for epoch in range(num_epochs):
            for batch in self.train_dataloader:
                batch = {k: v.to(self.device) for k,v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
    """
    def test(self):
        self.model.eval()
        for batch in self.eval_dataloader:
            batch = {k: v.to(self.device) for k,v in batch.items}
            with torch.no_grad():
                outputs = self.model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            #fix ref
            self.metric.add_batch(predictions=predictions, references=batch["labels"])
        self.metric.compute()

    def tokenizer_function(self, example):
        #tokens = self.tokenizer.tokenize(example)
        #ids = self.tokenizer.convert_tokens_to_ids(tokens)
        #final_inputs = self.tokenizer.prepare_for_model(ids)
        return self.tokenizer(example["input"],example["text"],truncation=True,padding=True,max_length=512)

    #Note that this is only for wiki bio
    def DatasetReparser(self, originalFileDir, NewFileDir):
        of = open(originalFileDir)
        nf = open(NewFileDir, 'w',encoding="utf-8")
        with of as handle:
            json_data = [json.loads(line) for line in handle]
            print("done w/ json data")

        for unfilteredData in json_data:
        of = open(originalFileDir)
        nf = open(NewFileDir, 'w',encoding="utf-8")
        nf.seek(0)
        nf.truncate()
        with of as handle:
            json_data = [json.loads(line) for line in handle]
            print("done w/ json data")

        for unfilteredData in json_data:
            #print("in loop")
            data_buffer = []
            for element in unfilteredData['data']:
                data_buffer.append(' - '.join(element))
            data = "data: " + ' / '.join(data_buffer)
            docTitle = "document title: " + unfilteredData["doc_title"] + " | "
            docTitleBPE = "document title with bpe: " + unfilteredData["doc_title_bpe"] + " | "
            SecTitle = "section title: " + ' - '.join(unfilteredData["sec_title"]) + " | "
            text = json.dumps(unfilteredData["text"])
            finalInput = docTitle + docTitleBPE + SecTitle + data
            finalLine= "{\"input\":"+ json.dumps(finalInput).replace("@@ ","") + ", " + "\"text\":" + text.replace("@@ ","") +"}\n"
            nf.write(finalLine)
        #nf.truncate(50000)
