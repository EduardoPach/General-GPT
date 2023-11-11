import torch
import numpy as np


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

def build_retrieval_input(caption: str, add_special_token: bool = True) -> str:
    """Builds the input string for General-LLM
    retrieval task.

    Args:
        caption (str): The caption.

    Returns:
        str: The input string for the model.
    """
    output = "Caption: [" + caption.strip() + "]."
    if add_special_token:
        output += " [CLIP OUT][\CLIP OUT]"
    return output

def build_caption_input(caption: str) -> str:
    """Builds the input string for General-LLM
    captioning task.

    Args:
        caption (str): The caption.

    Returns:
        str: The input string for the model.
    """
    output = "[CLIP IN] [\CLIP IN] Caption: [" + caption.strip() + "]."
    return output