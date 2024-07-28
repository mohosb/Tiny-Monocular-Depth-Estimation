import os
import sys
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from PIL import Image

class RawNYUDepthV2(Dataset):
    def __init__(self, root, room_type='living_room', transform=lambda image: np.array(image)):
        self.paths = glob(os.path.join(root, f'{room_type}*/r-*.ppm'))
        self.paths.sort()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index])
        image = self.transform(image)
        return image

    def clean(self, verbose=False):
        for i in range(len(self)):
            try:
                self.__getitem__(i)
            except OSError:
                path = self.paths[i]
                os.remove(path)
                if verbose:
                    print('Removing', path)
        return self

if __name__ == '__main__':
    ds = RawNYUDepthV2(sys.argv[1], '*room*')
    ds.clean(True)

