import argparse

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
    PATH_BASE_MODELS,
    PATH_CACHE_DATASETS,
    PATH_LOGS,
    PL,
    PROJECT_NAME,
    TOKENIZER_MODEL,
    WEIGHT_DECAY
)

from src.datamodule import NL2NLDM
from src.model import NL2NL

def test_dm(args):
    seed_everything(42)
    
    dm = NL2NLDM(
        tokenizer_model=args.tokenizer,
        pl=args.pl, 
        max_seq_len=args.max_seq_len, 
        padding=args.padding, 
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    dm.setup()

    print(next(iter(dm.train_dataloader())))


def NL2NLRun(args):
    dm = NL2NLDM(
        tokenizer_model=args.tokenizer,
        pl=args.pl,
        path_base_models=args.path_base_models,
        path_cache_dataset=args.path_cache_datasets,
        max_seq_len=args.max_seq_len,
        padding=args.padding,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    model = NL2NL(
        encoder_model=args.nl_en_model,
        decoder_model=args.nl_de_model,
        learning_rate=args.lr,
        weight_decay=args.wd,
    )

    logger = TensorBoardLogger(
        save_dir=args.path_logs,
        run_name=args.run_name,
    )

    if args.logger == "wandb":
        logger = WandbLogger(
            save_dir=args.path_logs,
            name=args.run_name,
            id=args.run_name,
            project=PROJECT_NAME
        )


    trainer = pl.Trainer(
        logger=logger,
        accelerator="gpu", 
        devices=args.gpus,   
        max_epochs=args.epochs,
        log_every_n_steps=2,
        deterministic=True           # Hopefully get same results on different GPUs
    )

    trainer.fit(model, datamodule=dm)


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="Set epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Set batch size")

    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Set learning rate")
    parser.add_argument("--wd", type=float, default=WEIGHT_DECAY, help="Set weight decay")
    parser.add_argument("--dropout", type=float, default=DROPOUT_RATE, help="Set dropout rate")

    parser.add_argument("--tokenizer", type=str, default=TOKENIZER_MODEL, help="Set Tokenizer")
    parser.add_argument("--nl_en_model", type=str, default=NL_ENCODER_BASE_MODEL, help="Set NL encoder base model")
    parser.add_argument("--nl_de_model", type=str, default=NL_DECODER_BASE_MODEL, help="Set NL decoder base model")
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQUENCE_LENGTH, help="Set max length of a sequence")
    parser.add_argument("--padding", type=str, default=PADDING, help="Set padding type")
    
    parser.add_argument("--pl", type=str, default=PL, help="Set programming language")
    parser.add_argument("--logger", type=str, default="tensorboard", help="Set logger")

    parser.add_argument("--gpus", type=int, default=AVAIL_GPUS, help="Set no. of GPUs")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Set no. of CPU Threads")

    parser.add_argument("--path_logs", type=str, default=PATH_LOGS, help="Set path to log runs")
    parser.add_argument("--path_base_models", type=str, default=PATH_BASE_MODELS, help="Set path to save base pretrained models")
    parser.add_argument("--path_cache_datasets", type=str, default=PATH_CACHE_DATASETS, help="Set path to cache datasets")

    parser.add_argument("--run_name", type=str, required=True, help="Set exp run name")

    args = parser.parse_args()

    # test_dm(args)
    NL2NLRun(args)

