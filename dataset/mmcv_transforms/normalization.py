# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from .basetransform import BaseTransform



class RescaleTOOneNegOne(BaseTransform):
    """Transform the images into a range between 0 and 1.

    Required keys are the keys in attribute "keys", added or modified keys are
    the keys in attribute "keys".
    It also supports rescaling a list of images.

    Args:
        keys (Sequence[str]): The images to be transformed.
    """

    def __init__(self, keys):
        self.keys = keys

    def transform(self, results):
        """transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        for key in self.keys:
            if isinstance(results[key], list):
                results[key] = [
                    v.astype(np.float32) / 255. * 2 - 1 for v in results[key]
                ]
            else:
                results[key] = results[key].astype(np.float32) / 255. * 2 - 1
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'



