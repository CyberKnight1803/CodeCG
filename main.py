import argparse
from json import decoder

import pytorch_lightning as pl 
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from config import (
    AVAIL_GPUS,
    BATCH_SIZE,
    DROPOUT_RATE,
    LEARNING_RATE, 
    MAX_EPOCHS,
    MAX_SEQUENCE_LENGTH,
    NL_DECODER_BASE_MODEL,
    NL_ENCODER_BASE_MODEL,
    NUM_WORKERS,
    PADDING,
    PL,
    WEIGHT_DECAY
)

from src.datamodule import NL2NLDM

def main(args):
    seed_everything(42)
    
    dm = NL2NLDM(
        encoder_base_model=args.nl_en_model, 
        decoder_base_model=args.nl_de_model, 
        pl=args.pl, 
        max_seq_len=args.max_seq_len, 
        padding=args.padding, 
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    dm.setup()

    print(next(iter(dm.train_dataloader())))

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="Set epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Set batch size")

    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Set learning rate")
    parser.add_argument("--wd", type=float, default=WEIGHT_DECAY, help="Set weight decay")
    parser.add_argument("--dropout", type=float, default=DROPOUT_RATE, help="Set dropout rate")

    parser.add_argument("--nl_en_model", type=str, default=NL_ENCODER_BASE_MODEL, help="Set NL encoder base model")
    parser.add_argument("--nl_de_model", type=str, default=NL_DECODER_BASE_MODEL, help="Set NL decoder base model")
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQUENCE_LENGTH, help="Set max length of a sequence")
    parser.add_argument("--padding", type=str, default=PADDING, help="Set padding type")
    
    parser.add_argument("--pl", type=str, default=PL, help="Set programming language")

    parser.add_argument("--gpus", type=int, default=AVAIL_GPUS, help="Set no. of GPUs")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Set no. of CPU Threads")

    

    args = parser.parse_args()

    main(args)

