import os
from typing import Tuple

import torch
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader

from src.dataset import utils

class COCODataset(Dataset):
    """COCO Dataset class.

    Args:
        caption_file (str): Path to the COCO captions JSON file.
        image_dir (str): Path to the COCO images directory.
        coco_embeddings_file (str): Path to the CLIP embeddings file from COCO images.
    """
    def __init__(self, caption_file: str, image_dir: str, coco_embeddings_file: str) -> None:
        self.coco_captions = COCO(caption_file)
        coco_images_tensor = utils.load_embedding_file(coco_embeddings_file)

        self.files = os.listdir(image_dir)
        self.file_id_dict = {}
        self.coco_data = []

        # Create data tuples: (caption, image_id, CLIP embedding, index of embedding)
        for i, f in enumerate(self.files):
            file_id = int(f[:-4])
            self.file_id_dict[i] = file_id
            for ann_id in self.coco_captions.getAnnIds(imgIds=file_id):
                caption = self.coco_captions.loadAnns(ids=ann_id)[0]["caption"]
                
                if np.random.random() < 0.5:
                    text = utils.build_retrieval_input(caption=caption)
                    clip_pos = -1
                else:
                    text = utils.build_caption_input(caption=caption)
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

    def __getitem__(self, idx: int) -> Tuple[str, int, torch.FloatTensor, int]:
        return self.coco_data[idx]

def get_dataloaders(
    caption_file: str, 
    image_dir: str, 
    coco_embeddings_file: str, 
    batch_size: int = 16, 
    **kwargs
) -> DataLoader:
    """_summary_

    Args:
        caption_file (str): _description_
        image_dir (str): _description_
        coco_embeddings_file (str): _description_
        batch_size (int, optional): _description_. Defaults to 16.

    Returns:
        DataLoader: _description_
    """
    coco_dataset = COCODataset(
        caption_file=caption_file,
        image_dir=image_dir,
        coco_embeddings_file=coco_embeddings_file
    )
    coco_dataloader = DataLoader(
        coco_dataset, 
        batch_size=batch_size, 
        collate_fn=utils.collate_fn,
        **kwargs
    )

    return coco_dataloader