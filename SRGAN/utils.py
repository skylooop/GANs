import torch
import os
import config
import numpy as np
from PIL import Image
from torchvision.utils import save_image


def plot_examples(low_res_folder, gen):
    files = os.listdir(low_res_folder)

    gen.eval()
    for file in files:
        image = Image.open("Data/LR/" + file)
        with torch.no_grad():
            upscaled_img = gen(
                config.test_transform(image=np.asarray(image))["image"]
                .unsqueeze(0)
                .to(config.DEVICE)
            )
        save_image(upscaled_img * 0.5 + 0.5, f"Data/saved/{file}")
    gen.train()