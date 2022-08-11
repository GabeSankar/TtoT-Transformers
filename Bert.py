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

        self.DatasetReparser("OldDatasets/train.json", "train.json")
        print("done reparsing train.json")
        self.DatasetReparser("OldDatasets/test.json", "test.json")
        print("done reparsing test.json")
        torch.cuda.empty_cache()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.configuration = BertConfig()

        self.model = BertModel(self.configuration)

        self.tokenizer = BertTokenizer(vocab_file=vocabFile, do_lower_case=onlyLowercase)
        #self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        datasetFiles = {"train": "train.json", "test": "test.json"}
        dataset = load_dataset("json", data_files=datasetFiles)
        print(dataset)
        tokenized_dataset = dataset.map(self.tokenizer_function, batched=True)


        train_args = TrainingArguments(output_dir="Trainer", evaluation_strategy="epoch")

        self.metric = load_metric("accuracy")

        #self.trainer = Trainer(model=self.model, args=train_args, train_dataset=train_dataset, eval_dataset=test_dataset, compute_metrics=self.metric)
        #tokenized_dataset.remove_column(["input"])
        #tokenized_dataset.rename_column("text", "labels")
        #tokenized_dataset.set_format("torch")
        print(tokenized_dataset)
        small_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(1000))
        small_test_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(1000))

        self.train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)

        self.eval_dataloader = DataLoader(small_test_dataset, batch_size=8)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
    def train(self, num_epochs):

        num_training_steps = num_epochs * len(self.train_dataloader)
        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
        progress_bar = tqdm(range(num_training_steps))

        self.model.to(self.device)
        self.model.train()
        for epoch in range(num_epochs):
            for batch in self.train_dataloader:
                batch = {k: v.to(self.device) for k,v in batch.items}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

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
        return self.tokenizer(example["input"], example["text"], truncation=True)

    #Note that this is only for wiki bio
    def DatasetReparser(self, originalFileDir, NewFileDir):
        of = open(originalFileDir)
        nf = open(NewFileDir, 'w',encoding="utf-8")
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
            finalLine= "{\"input\":"+ json.dumps(finalInput) + ", " + "\"text\":" + text +"}\n"
            nf.write(finalLine)


