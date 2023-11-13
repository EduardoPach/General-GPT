import os
import argparse
from collections import namedtuple
from typing import List, Tuple, Any, Dict

import yaml
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.general_llm import GeneralLLM, GeneralLLMLoss


load_dotenv()

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


def prepare_inputs(
    model: GeneralLLM, 
    input_ids: torch.LongTensor,  
    attention_mask: torch.LongTensor, 
    clip_embeds: torch.FloatTensor, 
    clip_pos: List[int]
) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor, List[int], List[int]]:
    device = input_ids.device

    # Generate token embeddings
    token_embs = model.token_embeder(input_ids)
    # Project CLIP embeddings to match token embeddings size
    clip_embeds = model.clip_projection_input(clip_embeds)
    
    # Create placeholder tensors to concatenate CLIP embeddings in the case of CLIP->caption
    input_clip_embs = torch.zeros((token_embs.size(0), token_embs.size(1)+1, token_embs.size(2)), device=device)
    target_clip_mask = torch.zeros((attention_mask.size(0), attention_mask.size(1)+1), dtype=torch.int64, device=device)
    target_clip_ids = torch.zeros((input_ids.size(0), input_ids.size(1)+1), dtype=torch.int64, device=device)

    embedding_ids = []  # Store indices of caption->CLIP examples
    captioning_ids = [] # Store indices of CLIP->caption examples

    for c in range(len(clip_pos)):
        # Retrieval task (caption->CLIP)
        if clip_pos[c] < 0:
            embedding_ids.append(c)

            tok_count = torch.sum(attention_mask[c])
            pos = tok_count - abs(clip_pos[c]) # position of [\CLIP OUT] token, while ignoring variable padding 

            # Add period token to match placeholder tensor size
            target_clip_ids[c] = torch.cat(
                (
                    input_ids[c, :pos+1], 
                    torch.full((1,), fill_value=13).to(device), 
                    input_ids[c, pos+1:]
                ), 
                dim=0
            )
            target_clip_mask[c, :tok_count+1] = 1

            with torch.no_grad():
                period_token = model.token_embeder(torch.tensor([13]).to(device)).reshape(1, -1)

                input_clip_embs[c] = torch.cat(
                    (
                        token_embs[c, :pos+1], 
                        period_token, 
                        token_embs[c, pos+1:]
                    ), 
                    dim=0
                )
                
        else:
            # Captioning task (CLIP->caption)
            captioning_ids.append(c)
            pos = clip_pos[c] # position of [\CLIP IN] token

            input_clip_embs[c] = torch.cat((token_embs[c, :pos], clip_embeds[c].reshape(1, -1).to(device), token_embs[c, pos:]), dim=0) # Add input CLIP embedding
            target_clip_ids[c] = torch.cat((input_ids[c, :pos], torch.zeros((1,)).to(device), input_ids[c, pos:]), dim=0)    # Add dummy token; is ignored in loss 
            target_clip_mask[c] = torch.cat((attention_mask[c, :pos], torch.ones((1,)).to(device), attention_mask[c, pos:]), dim=0)  # Avoid masking new token

    return input_clip_embs, target_clip_ids, target_clip_mask, embedding_ids, captioning_ids

def step(
    model: GeneralLLM,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    embs: torch.FloatTensor,
    clip_pos: List[int],
    loss_fn: GeneralLLMLoss
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    input_clip_embs, target_clip_ids, target_clip_mask, embedding_ids, captioning_ids = \
        prepare_inputs(
            model,
            input_ids, 
            attention_mask, 
            embs, 
            clip_pos
        )
    
    outputs = model(
        inputs_embeds=input_clip_embs,
        attention_mask=target_clip_mask,
        return_dict=True,
        output_hidden_states=True
    )

    logits = outputs.logits
    hidden_states = outputs.hidden_states

    pred_clip_embeds = []
    for idx in embedding_ids:
        tok_count = torch.sum(attention_mask[idx])
        pos = (tok_count - abs(clip_pos[idx])) - 1
        pred_clip_embeds.append(hidden_states[-1][idx][pos])
    
    if len(pred_clip_embeds) > 0:
        pred_clip_embeds = torch.stack(pred_clip_embeds)
        pred_clip_embeds = model.clip_projection_output(pred_clip_embeds)

    loss, loss_embedding, loss_caption = loss_fn(
        logits,
        embs,
        input_ids,
        pred_clip_embeds,
        target_clip_ids,
        embedding_ids,
        captioning_ids
    )

    return loss, loss_embedding, loss_caption