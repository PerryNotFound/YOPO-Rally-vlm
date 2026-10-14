import time

import torch
from torch.utils.data import DataLoader


@torch.inference_mode()
def test_yopo_dataset(yopo_data_loader: DataLoader):
    start_time = time.time()
    for index, (image, state, output) in enumerate(yopo_data_loader):
        pass
    print(f"Time: {time.time() - start_time}")
