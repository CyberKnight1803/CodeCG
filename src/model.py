from turtle import forward
from typing import Optional
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import pytorch_lightning as pl 
from pytorch_lightning.utilities.types import STEP_OUTPUT

from transformers import (
    AutoModel,
    GPT2Config,
    RobertaConfig,
    EncoderDecoderModel,
)

from config import (
    LEARNING_RATE,
    NL_DECODER_BASE_MODEL,
    NL_ENCODER_BASE_MODEL, 
    PATH_BASE_MODELS, 
    VOCAB_SIZE,
    WEIGHT_DECAY
)

class LMHead(nn.Module):
    def __init__(
        self, 
        config, 
    ) -> None:
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))   
        self.decoder.bias = self.bias 

    def forward(self, features):
        x = self.dense(features)
        x = self.layer_norm(x)
        x = F.gelu(x)

        # project back to size of vocab with bias 
        x = self.decoder(x)
        return x 


class NLEncoder(nn.Module):
    """
        Natural Language Encoder
    """

    def __init__(
        self, 
        model_name_or_path: str = NL_ENCODER_BASE_MODEL, 
        vocab_size: int = VOCAB_SIZE
    ) -> None:

        super().__init__()
        if model_name_or_path.lower() == "roberta-base":
            self.config = RobertaConfig()
            self.config.vocab_size = vocab_size
        
        
        self.model = AutoModel.from_config(self.config)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask, 
            output_hidden_states=True
        )


class NLDecoder(nn.Module):
    """
        Natural Language Decoder
    """
    def __init__(
        self,
        model_name_or_path: str = NL_DECODER_BASE_MODEL,
        vocab_size: int = VOCAB_SIZE,
    ) -> None:

        super().__init__()

        if model_name_or_path.lower() == "gpt2":
            self.config = GPT2Config()
            self.config.add_cross_attention = True           # Setting this as decoder
            self.config.vocab_size = vocab_size 

        self.model = AutoModel.from_config(self.config)
    
    def forward(self, input_ids, attention_mask, encoder_hidden_states, encoder_attention_mask): 
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask
        )


class NL2NL(pl.LightningModule):
    def __init__(
        self, 
        encoder_model: str = NL_ENCODER_BASE_MODEL,
        decoder_model: str = NL_DECODER_BASE_MODEL,

        vocab_size: int = VOCAB_SIZE,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Initializing encoder-decoder 
        self.encoder = NLEncoder(encoder_model, vocab_size)
        self.decoder = NLDecoder(decoder_model, vocab_size)

        self.lm_head = LMHead(self.decoder.config)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, target_input_ids, target_attention_mask):
        encoder_outs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ) 
        encoder_hidden_states = encoder_outs.last_hidden_states 

        decoder_outs = self.decoder(
            input_ids=target_input_ids,
            attention_mask=target_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
        )

        lm_outs = self.lm_head(decoder_outs.last_hidden_states)
        return lm_outs 

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        outs = self(
            batch['input_ids'],
            batch['attention_mask'],
            batch['target_input_ids'],
            batch['target_attention_mask']
        )

        loss = self.loss_fn(
            outs.view(-1, self.decoder.config.vocab_size),
            batch['labels'].view(-1)
        )


        self.log('loss/train', loss)

        return loss 

    
    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        outs = self(
            batch['input_ids'],
            batch['attention_mask'],
            batch['target_input_ids'],
            batch['target_attention_mask']
        )

        loss = self.loss_fn(
            outs.view(-1, self.decoder.config.vocab_size),
            batch['labels'].view(-1)
        )


        self.log('loss/val', loss)
    
    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        outs = self(
            batch['input_ids'],
            batch['attention_mask'],
            batch['target_input_ids'],
            batch['target_attention_mask']
        )

        loss = self.loss_fn(
            outs.view(-1, self.decoder.config.vocab_size),
            batch['labels'].view(-1)
        )


        self.log('loss/test', loss) 

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(), 
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        return optimizer 
class PLEncoder(nn.Module):
    """
        PL Encoder
    """

    pass 

class PL2NLGenerator(nn.Module):
    """
        PL2NL Generator
    """
    
    pass 

class PL2NLDescriminator(nn.Module):
    pass 

class PL2NL(pl.LightningModule):
    pass 