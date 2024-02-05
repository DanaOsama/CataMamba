import torch
import matplotlib.pyplot as plt
import ignite
import tempfile
import sys
import shutil
import os
import logging

from monai.data import CacheDataset, DataLoader



# class IXIDataset(Randomizable, CacheDataset):
#     resource = "http://biomedic.doc.ic.ac.uk/" + "brain-development/downloads/IXI/IXI-T1.tar"
#     md5 = "34901a0593b41dd19c1a1f746eac2d58"

#     def __init__(
#         self,
#         root_dir,
#         section,
#         transform,
#         download=False,
#         seed=0,
#         val_frac=0.2,
#         test_frac=0.2,
#         cache_num=sys.maxsize,
#         cache_rate=1.0,
#         num_workers=0,
#     ):
#         if not os.path.isdir(root_dir):
#             raise ValueError("Root directory root_dir must be a directory.")
#         self.section = section
#         self.val_frac = val_frac
#         self.test_frac = test_frac
#         self.set_random_state(seed=seed)
#         dataset_dir = os.path.join(root_dir, "ixi")
#         tarfile_name = f"{dataset_dir}.tar"
#         if download:
#             download_and_extract(self.resource, tarfile_name, dataset_dir, self.md5)
#         # as a quick demo, we just use 10 images to show

#         self.datalist = [
#             {"image": os.path.join(dataset_dir, "IXI314-IOP-0889-T1.nii.gz"), "label": 0},
#             {"image": os.path.join(dataset_dir, "IXI249-Guys-1072-T1.nii.gz"), "label": 0},
#             {"image": os.path.join(dataset_dir, "IXI609-HH-2600-T1.nii.gz"), "label": 0},
#             {"image": os.path.join(dataset_dir, "IXI173-HH-1590-T1.nii.gz"), "label": 1},
#             {"image": os.path.join(dataset_dir, "IXI020-Guys-0700-T1.nii.gz"), "label": 0},
#             {"image": os.path.join(dataset_dir, "IXI342-Guys-0909-T1.nii.gz"), "label": 0},
#             {"image": os.path.join(dataset_dir, "IXI134-Guys-0780-T1.nii.gz"), "label": 0},
#             {"image": os.path.join(dataset_dir, "IXI577-HH-2661-T1.nii.gz"), "label": 1},
#             {"image": os.path.join(dataset_dir, "IXI066-Guys-0731-T1.nii.gz"), "label": 1},
#             {"image": os.path.join(dataset_dir, "IXI130-HH-1528-T1.nii.gz"), "label": 0},
#         ]
#         data = self._generate_data_list()
#         super().__init__(
#             data,
#             transform,
#             cache_num=cache_num,
#             cache_rate=cache_rate,
#             num_workers=num_workers,
#         )

#     def randomize(self, data=None):
#         self.rann = self.R.random()

#     def _generate_data_list(self):
#         data = []
#         for d in self.datalist:
#             self.randomize()
#             if self.section == "training":
#                 if self.rann < self.val_frac + self.test_frac:
#                     continue
#             elif self.section == "validation":
#                 if self.rann >= self.val_frac:
#                     continue
#             elif self.section == "test":
#                 if self.rann < self.val_frac or self.rann >= self.val_frac + self.test_frac:
#                     continue
#             else:
#                 raise ValueError(
#                     f"Unsupported section: {self.section}, " "available options are ['training', 'validation', 'test']."
#                 )
#             data.append(d)
#         return data