import argparse

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers.optimization import get_linear_schedule_with_warmup

from src import utils

def train_loop(
    model, 
    tokenizer, 
    optimizer, 
    scheduler, 
    dataloader, 
    device,
    epochs=5,
    use_amp: bool = True
):
    
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for epoch in range(epochs):
       
        print(f"Training epoch: {epoch}")
        num_batches = len(dataloader)

        for batch_idx, (caps, _, embs, clip_pos) in  tqdm(enumerate(dataloader), unit=" batch", total=len(dataloader)):


            loss = utils.train_lm(model, tokenizer, caps, embs, clip_pos, device, use_amp)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            if batch_idx % 1000 == 0 and batch_idx != 0:
                print(f"Loss at batch {batch_idx} / {num_batches}  = {loss}")


def main(args: argparse.Namespace) -> None:
    """Main function for training the CLIP-guided language model."""
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    config = utils.load_config("./config.yml")
    config = utils.update_named_tuple_from_args(config, args)

    model, tokenizer = utils.get_model_and_tokenizer(config.checkpoint, config.dtype, device)

    coco_dataset = utils.COCODataset(
        config.caption_file, 
        config.image_dir, 
        config.coco_embeddings_file
    )

    coco_dataloader = DataLoader(
        coco_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=utils.collate_fn
    )

    epochs = config.epochs
    use_amp = config.use_amp
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
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
        device,
        epochs,
        use_amp
    )

    repo_id = f"EduardoPacheco/{config.checkpoint}-clip-guided" 
    response = utils.push_to_hf_hub(model, tokenizer, repo_id)
    if response:
        print(f"Model pushed to HuggingFace Hub at {repo_id}")
    else:
        print("Something went wrong while pushing the model to HuggingFace Hub.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to train the CLIP-guided language model.")

    parser.add_argument(
        '--checkpoint', 
        type=str,
        default=None,
        help="Path to the checkpoint to load the model from."
    )

    parser.add_argument(
        '--caption-file', 
        type=str, 
        default=None,
        help="Path to the COCO captions JSON file."
    )

    parser.add_argument(
        '--image-dir', 
        type=str, 
        default=None,
        help="Path to the COCO images directory."
    )

    parser.add_argument(
        '--coco-embeddings-file', 
        type=str, 
        default=None,
        help="Path to the CLIP embeddings file from COCO images."
    )

    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=None, 
        help="Batch size."
    )

    parser.add_argument(
        '--lr', 
        type=float, 
        default=None, 
        help="Learning rate."
    )

    parser.add_argument(
        '--epochs', 
        type=int, 
        default=None, 
        help="Number of epochs."
    )

    parser.add_argument(
        '--use-amp', 
        default=None, 
        help="Whether to use automatic mixed precision.",
        action='store_true'
    )

    parser.add_argument(
        "--dtype", 
        type=str,
        help="Data type to use for the model.",
        default=None
    )

    args = parser.parse_args()

    main(args)
