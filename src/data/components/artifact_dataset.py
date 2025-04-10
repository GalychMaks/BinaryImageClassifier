import os
from typing import List

from PIL import Image
from torch.utils.data import Dataset


class ArtifactImageDataset(Dataset):
    """
    Custom dataset for artifact detection in generated face images.

    Expects filenames in the format: ``image_<frame_index>_<label>.png`` where
    ``<label>`` is either 0 (artifact) or 1 (artifact-free).
    """

    def __init__(self, image_paths: List[str], transform=None):
        """
        Initialize the dataset.

        :param image_paths: List of image file paths.
        :param transform: Optional transform to apply to the images.
        """
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        """
        Return the number of samples in the dataset.

        :return: Number of image paths.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Load and return a single (image, label) pair.

        :param idx: Index of the image to load.
        :return: Tuple of (image, label).
        """
        img_path = self.image_paths[idx]
        label = int(os.path.basename(img_path).split("_")[-1].split(".")[0])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
