import os
import argparse

import clip
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

def main(args: argparse.Namespace) -> None:
    """Main function for embedding images using CLIP.

    Parameters
    ----------
    args : argparse.Namespace
        Namespace containing arguments passed to the script.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = args.model_name
    image_dir = args.image_dir
    output_dir = args.output_dir
    batch_size = args.batch_size
    filename = f"{output_dir}/coco_train2017_clip_{model_name.replace('/', '-')}_embs.npy"

    clip_model, preprocess = clip.load(model_name, device=device) # output dim of 768
    clip_model.to(device).eval()

    image_tensors = []

    for i in tqdm(range(0, len(os.listdir(image_dir)), batch_size)):
        images = []

        with torch.no_grad():
            for filename in os.listdir(image_dir)[i:i+batch_size]: 
                images.append(preprocess(Image.open(os.path.join(image_dir, filename))).unsqueeze(0))

            image_tensors.append(clip_model.encode_image(torch.cat(images, dim=0).to(device)).detach().cpu())

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.save(
        filename, 
        torch.cat(image_tensors, dim=0).numpy(), 
        allow_pickle=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your script description")

    parser.add_argument(
        '--model_name', 
        type=str, 
        help="Name of the CLIP model to use",
        choices=['RN50',
            'RN101',
            'RN50x4',
            'RN50x16',
            'RN50x64',
            'ViT-B/32',
            'ViT-B/16',
            'ViT-L/14',
            'ViT-L/14@336px'
        ]
    )
    parser.add_argument('--image_dir', type=str, help="Directory containing images", default="./train2017")
    parser.add_argument('--output_dir', type=str, help="Output directory", default="./coco_embs")
    parser.add_argument('--batch_size', type=int, help="Batch size", default=512)

    args = parser.parse_args()

    main(args)