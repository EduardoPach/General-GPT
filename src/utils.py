import os
import argparse
from collections import namedtuple
from typing import List, Tuple, Any, Dict


import yaml
import torch
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, LlamaForCausalLM

from dotenv import load_dotenv

load_dotenv()


def load_config(file_path: str) -> namedtuple:
    """Loads a YAML configuration file and returns a named tuple.

    Args:
        file_path (str): Path to the YAML configuration file.

    Raises:
        ValueError: In case the YAML file is invalid or empty.
        ValueError: In case there is an error reading or processing the YAML file.

    Returns:
        namedtuple: Named tuple containing the configuration parameters.
    """
    try:
        with open(file_path, 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)  # Load the YAML data
            if data is None or not isinstance(data, dict):
                raise ValueError("Invalid YAML file content. It should be a dictionary.")

            # Define a named tuple with keys from the dictionary
            Config = namedtuple('Config', data.keys())

            # Create an instance of the named tuple with values from the dictionary
            instance = Config(**data)

            return instance
    except Exception as e:
        raise ValueError(f"Error reading or processing YAML file: {str(e)}")

def update_named_tuple_from_args(named_tuple: namedtuple, args: argparse.Namespace) -> namedtuple:
    """
    Update a named tuple with values from argparse arguments, if arguments are not None.

    Args:
        named_tuple (namedtuple): The named tuple to update.
        args (argparse.Namespace): The argparse arguments.

    Returns:
        namedtuple: The updated named tuple.
    """
    updated_values = {}
    Config = namedtuple('Config', named_tuple._fields)
    for key in named_tuple._fields:
        arg_value = getattr(args, key)
        if arg_value is not None:
            updated_values[key] = arg_value
        else:
            updated_values[key] = getattr(named_tuple, key)
    
    updated_named_tuple = Config(**updated_values)
    return updated_named_tuple

def collate_fn(batch):
    captions, img_ids, clip_embs, clip_pos = [], [], [], []
    for cap, im_id, emb, p in batch:
        captions.append(cap)
        img_ids.append(im_id)
        clip_embs.append(emb)
        clip_pos.append(p)
    
    clip_embs = torch.cat(clip_embs, dim=0)
    
    return captions, img_ids, clip_embs, clip_pos


def load_embedding_file(embedding_file: str) -> torch.FloatTensor:
    """Loads a numpy file containing CLIP embeddings.

    Args:
        embedding_file (str): Path to the numpy file.

    Returns:
        torch.FloatTensor: Tensor containing the CLIP embeddings.
    """
    coco_images = np.load(embedding_file)
    return torch.from_numpy(coco_images).float()

class COCODataset(Dataset):
    """COCO Dataset class.

    Args:
        caption_file (str): Path to the COCO captions JSON file.
        image_dir (str): Path to the COCO images directory.
        coco_embeddings_file (str): Path to the CLIP embeddings file from COCO images.
    """
    def __init__(self, caption_file: str, image_dir: str, coco_embeddings_file: str) -> None:
        self.coco_captions = COCO(caption_file)
        coco_images_tensor = load_embedding_file(coco_embeddings_file)

        self.files = os.listdir(image_dir)
        self.file_id_dict = {}
        self.coco_data = []

        # Create data tuples: (caption, image_id, CLIP embedding, index of embedding)
        for i, f in enumerate(self.files):
            file_id = int(f[:-4])
            self.file_id_dict[i] = file_id
            for ann_id in self.coco_captions.getAnnIds(imgIds=file_id):
                if np.random.random() < 0.5:
                    text = "Caption: [" + self.coco_captions.loadAnns(ids=ann_id)[0]["caption"].strip() + "]. [CLIP OUT][\CLIP OUT]"
                    clip_pos = -1
                else:
                    text = "[CLIP IN] [\CLIP IN] Caption: [" + self.coco_captions.loadAnns(ids=ann_id)[0]["caption"].strip() + "]."
                    clip_pos = 1

                self.coco_data.append(
                    (
                        text, 
                        self.coco_captions.loadAnns(ids=ann_id)[0]["image_id"],
                        coco_images_tensor[i].unsqueeze(0),
                        clip_pos
                    )
                )

    def __len__(self) -> int:
        return len(self.coco_data)

    def __getitem__(self, idx) -> Tuple[str, int, torch.FloatTensor, int]:
        return self.coco_data[idx]

def train_lm(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    caps: List[str], 
    embs: torch.FloatTensor, 
    clip_pos: List[int],
    device: str,
    use_amp: bool=True
) -> torch.FloatTensor:
    #TODO: Optimize training
    
    criterion_ce = torch.nn.CrossEntropyLoss()
    criterion_mse = torch.nn.MSELoss()

    targets = tokenizer(
        caps, 
        padding=True, 
        return_tensors='pt', 
        return_attention_mask=True
    )
    targets_ids = targets['input_ids'].to(device)
    targets_mask = targets['attention_mask'].to(device)

    with torch.cuda.amp.autocast(enabled=use_amp):
        if isinstance(model, GPT2LMHeadModel):
            token_embs = model.transformer.wte(targets_ids)
        elif isinstance(model, LlamaForCausalLM):
            token_embs = model.model.embed_tokens(targets_ids)
        else:
            raise ValueError(f"Expected model to be of type GPT2LMHeadModel or LlamaForCausalLM but got {type(model)}")

        # Create placeholder tensors to concatenate CLIP embeddings in the case of CLIP->caption
        input_clip_embs = torch.zeros((token_embs.size(0), token_embs.size(1)+1, token_embs.size(2)), device=device)
        target_clip_mask = torch.zeros((targets_mask.size(0), targets_mask.size(1)+1), dtype=torch.int64, device=device)
        target_clip_ids = torch.zeros((targets_ids.size(0), targets_ids.size(1)+1), dtype=torch.int64, device=device)

        embedding_ids = []  # Store indices of caption->CLIP examples
        captioning_ids = [] # Store indices of CLIP->caption examples

        for c in range(len(caps)):
            # Retrieval task (caption->CLIP)
            if clip_pos[c] < 0:
                embedding_ids.append(c)

                tok_count = torch.sum(targets_mask[c])
                pos = tok_count - abs(clip_pos[c]) # position of [\CLIP OUT] token, while ignoring variable padding 

                # Add period token to match placeholder tensor size
                target_clip_ids[c] = torch.cat(
                    (
                        targets_ids[c, :pos+1], 
                        torch.full((1,), fill_value=13).to(device), 
                        targets_ids[c, pos+1:]
                    ), 
                    dim=0
                )
                target_clip_mask[c, :tok_count+1] = 1

                with torch.no_grad():
                    input_clip_embs[c] = torch.cat(
                        (
                            token_embs[c, :pos+1], 
                            model.transformer.wte(torch.tensor([13]).to(device)).reshape(1, -1), 
                            token_embs[c, pos+1:]
                        ), 
                        dim=0
                    )
                    
            else:
                # Captioning task (CLIP->caption)
                captioning_ids.append(c)
                pos = clip_pos[c] # position of [\CLIP IN] token

                input_clip_embs[c] = torch.cat((token_embs[c, :pos], embs[c].reshape(1, -1).to(device), token_embs[c, pos:]), dim=0) # Add input CLIP embedding
                target_clip_ids[c] = torch.cat((targets_ids[c, :pos], torch.zeros((1,)).to(device), targets_ids[c, pos:]), dim=0)    # Add dummy token; is ignored in loss 
                target_clip_mask[c] = torch.cat((targets_mask[c, :pos], torch.ones((1,)).to(device), targets_mask[c, pos:]), dim=0)  # Avoid masking new token

        outputs = model(
            inputs_embeds=input_clip_embs,
            return_dict=True,
            output_hidden_states=True,
            attention_mask=target_clip_mask
        )

        # Fetch last layer's hidden_state for [\CLIP OUT] tokens
        last_hiddens = []
        for idx in embedding_ids:
            tok_count = torch.sum(targets_mask[idx])                      # Use original position since we concatenated an additional token
            pos = (tok_count - abs(clip_pos[idx])) - 1                    # Subtract 1 since we shift targets
            last_hiddens.append(outputs['hidden_states'][-1][idx][pos])
            
        caption_to_clip_loss = criterion_ce(outputs['logits'][embedding_ids, :-1].view(-1, outputs['logits'].size(-1)), target_clip_ids[embedding_ids, 1:].view(-1)) \
                                + criterion_mse(torch.stack(last_hiddens, dim=0).to(device), embs[embedding_ids].to(device))

        clip_to_caption_loss = torch.nn.functional.cross_entropy(
            outputs['logits'][captioning_ids, :-1].reshape(-1, outputs['logits'].size(-1)), 
            targets_ids[captioning_ids].flatten(), 
            ignore_index=0
        )

        loss = caption_to_clip_loss + clip_to_caption_loss

    return loss

def get_model_and_tokenizer(checkpoint: str, dtype: str, device: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Loads a pretrained model and tokenizer.

    Args:
        checkpoint (str): Path to the checkpoint.
        dtype (str): Data type to use for the model.
        device (str): Device to load the model on.

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: Tuple containing the model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, token=os.environ["HF_READ_TOKEN"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["[CLIP IN]", "[\CLIP IN]", "[CLIP OUT]", "[\CLIP OUT]"]})

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, 
        device_map=device, 
        token=os.environ["HF_READ_TOKEN"],
        torch_dtype=getattr(torch, dtype)
    )
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

def push_to_hf_hub(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, repo_id: str) -> bool:
    """Uploads a model and tokenizer to Hugging Face Hub.

    Args:
        model (AutoModelForCausalLM): The model to upload.
        tokenizer (AutoTokenizer): The tokenizer to upload.
        repo_id (str): The repository ID to use for the upload.

    Returns:
        bool: True if the upload was successful, False otherwise.
    """
    try:
        model.push_to_hub(repo_id, token=os.environ["HF_WRITE_TOKEN"])
        tokenizer.push_to_hub(repo_id, token=os.environ["HF_WRITE_TOKEN"])
        return True
    except Exception as e:
        print(f"Error uploading to Hugging Face Hub: {str(e)}")
        return False