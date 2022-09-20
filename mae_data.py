import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms

logger = logging.getLogger(__name__)

##############################################################################
# Helper Classes
##############################################################################

DEFAULT_PPB = 16
MAX_SEQ_LENGTH = 9


def patchify(imgs: torch.Tensor, patch_size: int = DEFAULT_PPB):
    """
    imgs: (N, 1, H, W) x: (N, L, patch_size**2 * 1)
    or
    img: (1, H, W) x: (L, patch_size**2 * 1)
    """
    is_single_image = len(imgs.shape) == 3
    if is_single_image:
        imgs = imgs.unsqueeze(0)

    p = patch_size
    assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

    h = imgs.shape[2] // p
    w = imgs.shape[3] // p
    x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
    x = torch.einsum("nchpwq->nhwpqc", x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 1))

    if is_single_image:
        return x.squeeze(0)
    return x


def unpatchify(x: torch.Tensor, patch_size: int = DEFAULT_PPB):
    """
    x: (N, L, patch_size**2 * 3) --> imgs: (N, 1, H, W)
    or
    x: (L, patch_size**2 * 3) --> img: (1, H, W)
    """
    is_single_image = len(x.shape) == 2
    if is_single_image:
        x = x.unsqueeze(0)

    p = patch_size
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
    x = torch.einsum("nhwpqc->nchpwq", x)
    imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))

    if is_single_image:
        return imgs.squeeze(0)
    return imgs


##############################################################################
# Main Data Class
##############################################################################


class PretrainDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transforms: transforms.Compose,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.transforms = transforms

        self.pngs_list = list(
            Path(root_dir).rglob("*.png")
        )
        self.pngs_list = self.pngs_list

    def __len__(self):
        return len(self.pngs_list)

    def __getitem__(self, index: int):
        pil_img = Image.open(str(self.pngs_list[index]))
        if self.transforms is not None:
            pil_img = self.transforms(pil_img)
        return pil_img


class PretrainDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        batch_size: int = 64,
        shuffle_dataset: bool = True,
        num_workers: int = 4,
        pin_memory: bool = False,
        size: Tuple[int, int] = (16, 144),
        augment: bool = True,
        num_samples: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.augment = augment
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.shuffle_dataset = shuffle_dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.size = size
        self.num_samples = num_samples

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            dataset_size = len(
                list(Path(self.root_dir).rglob("*.png"))
            )
            # validation split --> 20% of total dataset size
            split = int(np.floor(0.2 * dataset_size))

            pretrain_dset_full = PretrainDataset(
                root_dir=self.root_dir,
                transforms=self.transforms(
                    do_squarify=False,
                    do_normalize=True,
                    do_random_erase=True
                )
            )
            self.train_data, self.val_data = random_split(
                pretrain_dset_full,
                [int(dataset_size - split), split]
            )

        if stage == "test" or stage is None:
            self.data_test = PretrainDataset(
                root_dir=self.root_dir,
                transforms=self.transforms(
                    do_normalize=True
                )
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            shuffle=False,
            drop_last=False,
        )

    def transforms(self,
                   do_resize: bool = True,
                   size: Union[int, Tuple[int, int]] = (16, 144),
                   do_squarify: bool = False,
                   do_normalize: bool = False,
                   do_random_erase: bool = False,
                   image_mean: Optional[float] = (0.1307,),
                   image_std: Optional[float] = (0.3081,),
                   random_erase_params: Optional[
                       Tuple[float,
                             int,
                             Tuple[float, float],
                             Tuple[float, float]
                             ]
                   ] = (0.25, 0, (0.01, 0.1), (0.3, 3.3),),
                   **kwargs,
                   ) -> transforms.Compose:
        r"""
        Returns a composition of transformations we want to apply to our images
        We always convert to RGB and tensorize the images

        Args:
            do_resize (`bool`, *optional*, defaults to `True`):
                Whether to resize the input to a certain `size`.
            size: (`int` or `Tuple(int)`):
                Resize the input to the given size. If a tuple is provided, it should 
                be (width, height). If only an integer is provided, then the input will 
                be resized to (size, size). Only has an effect if `do_resize` is set 
                to `True`.
            do_squarify (`bool`, defaults to `False`):
                Whether to squarify images, e.g. from 16x8464 to 368x368. This is 
                necessary for some models.
            do_normalize (`bool`, defaults to `False`):
                Whether to apply normalization with image_mean and image_std
            do_random_erase (`bool`, defaults to `False`):
                Whether to apply random erase data augmentation with random_erase_params
            image_mean (`float`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
                Mean value used for normalization. Defaults to ImageNet mean
            image_std (`float`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
                Standard deviation value used for normalization. Defaults to ImageNet std
            random_erase_params (`Tuple[float, int, Tuple[float, float], Tuple[float, float]]`, *optional*,
                                defaults to `(0.25, 0, (0.01, 0.1), (0.3, 3.3))`):
                Parameter tuple to be used when applying random erase data augmentation. 
                Is a tuple of (random_erase_probability, random_erase_value, 
                random_erase_scale, random_erase_ratio).
                More information here: https://pytorch.org/vision/main/generated/torchvision.transforms.RandomErasing.html

        Returns:
            A composition of transformations of type [~`Compose`]
        """

        # tforms = [
        #     transforms.Lambda(
        #         lambda img: img.convert("L") if img.mode != "L" else img
        #     )
        # ]
        tforms = []

        # Optionally, resize to specified size
        if do_resize and size:
            tforms.append(
                transforms.Resize(
                    size=size,
                    interpolation=transforms.InterpolationMode.BICUBIC
                )
            )

        # Tensorize image
        tforms.append(transforms.ToTensor())

        # Optionally, turn into square image by patchifying and unpatchifying
        if do_squarify:
            tforms.extend(
                [
                    transforms.Lambda(lambda img: patchify(img)),
                    transforms.Lambda(lambda img: unpatchify(img))
                ]
            )

        # Optionally, apply random erase data augmentation
        if do_random_erase:
            random_erase_p, random_erase_val, random_erase_scale, random_erase_ratio = random_erase_params
            logger.info(
                f"Applying random erase transformation w/ p={random_erase_p}, val={random_erase_val} "
                f"scale={random_erase_scale}, ratio={random_erase_ratio}"
            )
            tforms.append(
                transforms.RandomErasing(
                    p=random_erase_p,
                    value=random_erase_val,
                    scale=random_erase_scale,
                    ratio=random_erase_ratio
                )
            )

        # Optionally, apply normalization
        if do_normalize:
            logger.info(
                f"Applying normalization with mean={image_mean}, std={image_std}"
            )
            tforms.append(
                transforms.Normalize(mean=image_mean, std=image_std)
            )

        return transforms.Compose(tforms)
