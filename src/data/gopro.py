from __future__ import annotations

import random
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class GoProDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train", crop_size: int = 256) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.crop_size = crop_size
        self.blur_dir = self.root_dir / split / "blur"
        self.sharp_dir = self.root_dir / split / "sharp"
        if not self.blur_dir.exists() or not self.sharp_dir.exists():
            raise FileNotFoundError(
                f"Expected GoPro layout at {self.blur_dir} and {self.sharp_dir}, but one or both are missing."
            )
        self.image_names = sorted(
            file.name for file in self.blur_dir.iterdir() if file.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if not self.image_names:
            raise RuntimeError(f"No images found in {self.blur_dir}.")

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_name = self.image_names[index]
        blur = Image.open(self.blur_dir / image_name).convert("RGB")
        sharp = Image.open(self.sharp_dir / image_name).convert("RGB")

        if self.split == "train":
            blur, sharp = self._random_crop_pair(blur, sharp)
            blur, sharp = self._random_flip_pair(blur, sharp)
        else:
            blur, sharp = self._center_crop_pair_to_multiple(blur, sharp, multiple=8)

        return TF.to_tensor(blur), TF.to_tensor(sharp)

    def _random_crop_pair(self, blur: Image.Image, sharp: Image.Image) -> tuple[Image.Image, Image.Image]:
        width, height = blur.size
        crop_h = crop_w = self.crop_size
        if height < crop_h or width < crop_w:
            raise ValueError(
                f"Crop size {self.crop_size} is larger than image size {(width, height)} for split={self.split}."
            )
        top = random.randint(0, height - crop_h)
        left = random.randint(0, width - crop_w)
        return TF.crop(blur, top, left, crop_h, crop_w), TF.crop(sharp, top, left, crop_h, crop_w)

    def _random_flip_pair(self, blur: Image.Image, sharp: Image.Image) -> tuple[Image.Image, Image.Image]:
        if random.random() > 0.5:
            blur, sharp = TF.hflip(blur), TF.hflip(sharp)
        if random.random() > 0.5:
            blur, sharp = TF.vflip(blur), TF.vflip(sharp)
        return blur, sharp

    def _center_crop_pair_to_multiple(
        self,
        blur: Image.Image,
        sharp: Image.Image,
        multiple: int,
    ) -> tuple[Image.Image, Image.Image]:
        width, height = blur.size
        new_height = (height // multiple) * multiple
        new_width = (width // multiple) * multiple
        return TF.center_crop(blur, (new_height, new_width)), TF.center_crop(sharp, (new_height, new_width))
