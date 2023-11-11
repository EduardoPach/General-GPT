import argparse

import wandb
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers.optimization import get_linear_schedule_with_warmup

from src.train import utils
from src.dataset import get_dataloaders
from src.general_llm import GeneralLLM, GeneralLLMLoss

def train_loop(
    model: GeneralLLM, 
    tokenizer: AutoTokenizer, 
    optimizer: torch.optim.Optimizer, 
    scheduler: torch.optim.lr_scheduler.LambdaLR, 
    dataloader: DataLoader, 
    device: str,
    epochs: int = 5,
    use_amp: bool = True
) -> None:
    
    model.train()
    loss_fn = GeneralLLMLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for epoch in range(epochs):
       
        print(f"Training epoch: {epoch}")
        num_batches = len(dataloader)

        for batch_idx, (caps, _, embs, clip_pos) in  tqdm(enumerate(dataloader), unit=" batch", total=num_batches):
            targets = tokenizer(
                caps, 
                padding=True, 
                return_tensors='pt', 
                return_attention_mask=True
            )

            input_ids = targets['input_ids'].to(device)
            attention_mask = targets['attention_mask'].to(device)
            embs = embs.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, loss_embedding, loss_caption = utils.step(
                    model, 
                    input_ids,
                    attention_mask,
                    embs,
                    clip_pos,
                    loss_fn
                )
                

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            wandb.log({"train_loss": loss.item()})
            wandb.log({"train_loss_embedding": loss_embedding.item()})
            wandb.log({"train_loss_caption": loss_caption.item()})
            if batch_idx % 1000 == 0 and batch_idx != 0:
                print(f"Loss at batch {batch_idx} / {num_batches}  = {loss}")


def main(args: argparse.Namespace) -> None:
    """Main function for training the CLIP-guided language model."""
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    config = utils.load_config("./config.yml")
    config = utils.update_named_tuple_from_args(config, args)

    llm, tokenizer = utils.get_model_and_tokenizer(config.checkpoint, config.dtype, device)
    model = GeneralLLM(llm)

    train_dataloader = get_dataloaders(
        config.train_caption_file,
        config.train_image_dir,
        config.train_coco_embeddings_file,
        config.train_batch_size,
        shuffle=True
    )

    # val_dataloader = get_dataloaders(
    #     config.val_caption_file,
    #     config.val_image_dir,
    #     config.val_coco_embeddings_file,
    #     config.val_batch_size,
    #     shuffle=False
    # )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=2000, 
        num_training_steps=config.epochs * len(train_dataloader)
    )

    with wandb.init(project="clip-guided-lm", config=config._asdict(), mode=args.logging_mode) as run:
        train_loop(
            model, 
            tokenizer, 
            optimizer, 
            scheduler, 
            train_dataloader, 
            device,
            config.epochs,
            config.use_amp
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
        '--train-caption-file', 
        type=str, 
        default=None,
        help="Path to the COCO captions JSON file."
    )

    parser.add_argument(
        '--train-image-dir', 
        type=str, 
        default=None,
        help="Path to the COCO images directory."
    )

    parser.add_argument(
        '--train-coco-embeddings-file', 
        type=str, 
        default=None,
        help="Path to the CLIP embeddings file from COCO images."
    )

    parser.add_argument(
        '--train-batch-size', 
        type=int, 
        default=None, 
        help="Batch size."
    )

    parser.add_argument(
        '--val-caption-file', 
        type=str, 
        default=None,
        help="Path to the COCO captions JSON file."
    )

    parser.add_argument(
        '--val-image-dir', 
        type=str, 
        default=None,
        help="Path to the COCO images directory."
    )

    parser.add_argument(
        '--val-coco-embeddings-file', 
        type=str, 
        default=None,
        help="Path to the CLIP embeddings file from COCO images."
    )

    parser.add_argument(
        '--val-batch-size', 
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

    parser.add_argument(
	"--logging-mode",
	type=str,
	help="Selects which mode of logging we are at",
	choices=["online", "offline","disabled"],
	default="disabled"
    )

    args = parser.parse_args()

    main(args)
