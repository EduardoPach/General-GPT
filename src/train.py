import argparse

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from src import utils

def train_loop(
    model, 
    tokenizer, 
    optimizer, 
    scheduler, 
    dataloader, 
    epochs=5
):
    
    model.train()
    for epoch in range(epochs):
       
        print(f"Training epoch: {epoch}")
        num_batches = len(dataloader)

        for batch_idx, (caps, _, embs, clip_pos) in  tqdm(enumerate(dataloader)):

            optimizer.zero_grad()

            loss = utils.train_lm(model, tokenizer, caps, embs, clip_pos)
            loss.backward()

            optimizer.step()
            scheduler.step()
            
            if batch_idx % 1000 == 0 and batch_idx != 0:
                print(f"Loss at batch {batch_idx} / {num_batches}  = {loss}")


def main(args: args.Namespace) -> None:
    """Main function for training the CLIP-guided language model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = 

    coco_dataset = utils.COCODataset(
        args.caption_file, 
        args.image_dir, 
        args.coco_embeddings_file
    )

    coco_dataloader = DataLoader(
        coco_dataset, 
        batch_size=16, 
        shuffle=True, 
        collate_fn=utils.collate_fn
    )

    epochs = 5
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=2000, 
        num_training_steps=epochs * len(coco_dataloader)
    )

    train_loop(
        model, 
        tokenizer, 
        optimizer, 
        scheduler, 
        coco_dataloader, 
        epochs=epochs
    )

