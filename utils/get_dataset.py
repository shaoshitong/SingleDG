import hub,os
import numpy as np
ds = hub.load('hub://activeloop/pacs-val')


if not os.path.exists("/home/sst/dataset/PACS/val"):
    os.makedirs("/home/sst/dataset/PACS/val")

np.save("/home/sst/dataset/PACS/val/images.npy",ds.images.numpy())
np.save("/home/sst/dataset/PACS/val/labels.npy",ds.labels.numpy())
np.save("/home/sst/dataset/PACS/val/domains.npy",ds.domains.numpy())