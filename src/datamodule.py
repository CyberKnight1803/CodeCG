from typing import Optional
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from transformers import AutoTokenizer
import datasets as ds 

from config import (
    BATCH_SIZE,
    DATASET_NAME,
    MAX_SEQUENCE_LENGTH,
    NUM_WORKERS,
    PADDING,
    PATH_BASE_MODELS,
    NL_ENCODER_BASE_MODEL,
    NL_DECODER_BASE_MODEL,
    PATH_CACHE_DATASETS,
    PL,
    TOKENIZER_MODEL,
)

class NL2NLDM(pl.LightningDataModule):

    loader_cols = [
        "input_ids", 
        "attention_mask", 
        "target_input_ids", 
        "target_attention_mask"
    ]

    def __init__(
        self,
        tokenizer_model: str = TOKENIZER_MODEL,
        pl: str = PL,
        max_seq_len: int = MAX_SEQUENCE_LENGTH,
        padding: str = PADDING,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
    ) -> None:

        super().__init__() 

        self.tokenizer_model = tokenizer_model
        self.pl = pl 
        self.max_seq_len = max_seq_len
        self.padding = padding
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.tokenizer_model, 
            use_fast=True,
            cache_dir=PATH_BASE_MODELS,
        )


    def prepare_data(self) -> None:
        ds.load_dataset(DATASET_NAME, self.pl, cache_dir=PATH_CACHE_DATASETS)
    
    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = ds.load_dataset(DATASET_NAME, self.pl, cache_dir=PATH_CACHE_DATASETS)
        
        
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self._to_features,
                batched=True,
                batch_size=self.batch_size,
                num_proc=self.num_workers,
            )

            self.columns = [
                c for c in self.dataset[split].column_names if c in self.loader_cols
            ]

        self.dataset.set_format(type='torch', columns=self.columns)


    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.dataset['train'], 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.dataset['validation'], 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )
    
    def _to_features(self, batch, indices=None):
        
        features = self.tokenizer(
            text=batch['docstring'],
            max_length=self.max_seq_len,
            padding=self.padding,
            truncation=True,
        ) 

        targets = self.tokenizer(
            text=batch['docstring'], 
            max_length=self.max_seq_len,
            padding=self.padding,
            truncation=True,
        )

        features['target_input_ids'] = targets['input_ids']
        features['target_attention_mask'] = targets['attention_mask']

        features['labels'] = features['target_input_ids']
        return features 
