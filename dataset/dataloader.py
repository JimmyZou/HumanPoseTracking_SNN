import os
import numpy as np
import torch
from torch.utils.data import Dataset
import sys

sys.path.append("../")
from dataset.dataloader_mmhpsd import MMHPSDataloader
from dataset.dataloader_h36m import H36MDataloader
from dataset.dataloader_amass import AMASSDataloader
from dataset.dataloader_phspd import PHSPDataloader


class EventDataloader(Dataset):
    def __init__(
        self,
        data_dir="/home/shihao/data",
        smpl_dir="../smpl_model/models/smpl/SMPL_MALE.pkl",
        mode="train",
        num_steps=64,
        max_gap=4,
        channel=4,  # corresponds to frame
        img_size=256,
        modality="events",
        augmentation=False,
        use_mmhpsd=True,
        use_h36m=False,
        use_amass=False,
        use_phspd=False,
        use_mmhpsd_synthesis=False,
    ):
        self.data_dir = data_dir
        self.mode = mode
        self.num_steps = num_steps
        self.max_gap = max_gap
        self.img_size = img_size
        self.channel = channel
        self.cam_intr = np.array([1679.3, 1679.3, 641, 641]) * self.img_size / 1280.0
        assert modality in ["events", "rgb", "all"]
        self.modality = modality
        self.aug = augmentation
        self.smpl_dir = smpl_dir

        self.use_mmhpsd = use_mmhpsd
        self.mmhpsd_dataloader = None
        self.use_mmhpsd_synthesis = use_mmhpsd_synthesis
        self.use_h36m = use_h36m
        self.h36m_dataloader = None
        self.use_amass = use_amass
        self.amass_dataloader = None
        self.use_phspd = use_phspd
        self.phspd_dataloader = None

        self.all_clips = self.obtain_all_clips()

    def __len__(self):
        return len(self.all_clips)

    def __getitem__(self, idx):
        dataset_name, action, frame_idx, set_idx = self.all_clips[idx]
        if dataset_name == "mmhpsd":
            return self.mmhpsd_dataloader[set_idx]
        elif dataset_name == "h36m":
            return self.h36m_dataloader[set_idx]
        elif dataset_name == "amass":
            return self.amass_dataloader[set_idx]
        elif dataset_name == "phspd":
            return self.phspd_dataloader[set_idx]
        else:
            raise ValueError("Dataset name {} unkonw.".format(dataset_name))

    def obtain_all_clips(self):
        all_clips = []
        if self.use_mmhpsd:
            self.mmhpsd_dataloader = MMHPSDataloader(
                data_dir=self.data_dir,
                smpl_dir=self.smpl_dir,
                mode=self.mode,
                num_steps=self.num_steps,
                max_gap=self.max_gap,
                channel=self.channel,  # corresponds to frame
                img_size=self.img_size,
                modality=self.modality,
                augmentation=self.aug,
                use_synthesis=False,
            )
            all_clips += self.mmhpsd_dataloader.all_clips

        if self.use_mmhpsd_synthesis and self.mode == "train":
            self.mmhpsd_dataloader = MMHPSDataloader(
                data_dir=self.data_dir,
                smpl_dir=self.smpl_dir,
                mode=self.mode,
                num_steps=self.num_steps,
                max_gap=self.max_gap,
                channel=self.channel,  # corresponds to frame
                img_size=self.img_size,
                modality=self.modality,
                augmentation=self.aug,
                use_synthesis=True,
            )
            all_clips += self.mmhpsd_dataloader.all_clips

        if self.use_h36m:
            self.h36m_dataloader = H36MDataloader(
                data_dir=self.data_dir,
                smpl_dir=self.smpl_dir,
                mode=self.mode,
                num_steps=self.num_steps,
                max_gap=self.max_gap,
                channel=self.channel,  # corresponds to frame
                img_size=self.img_size,
                modality=self.modality,
                augmentation=self.aug,
            )
            all_clips += self.h36m_dataloader.all_clips

        if self.use_amass and self.mode == "train":
            self.amass_dataloader = AMASSDataloader(
                data_dir=self.data_dir,
                smpl_dir=self.smpl_dir,
                mode=self.mode,
                num_steps=self.num_steps,
                max_gap=self.max_gap,
                channel=self.channel,  # corresponds to frame
                img_size=self.img_size,
                modality=self.modality,
                augmentation=self.aug,
            )
            all_clips += self.amass_dataloader.all_clips

        if self.use_phspd and self.mode == "train":
            self.phspd_dataloader = PHSPDataloader(
                data_dir=self.data_dir,
                smpl_dir=self.smpl_dir,
                mode=self.mode,
                num_steps=self.num_steps,
                max_gap=self.max_gap,
                channel=self.channel,  # corresponds to frame
                img_size=self.img_size,
                modality=self.modality,
                augmentation=self.aug,
            )
            all_clips += self.phspd_dataloader.all_clips

        print("[in total {}] {} samples".format(self.mode, len(all_clips)))
        return all_clips


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    data_train = EventDataloader(
        data_dir="/home/shihao/data",
        smpl_dir="../smpl_model/models/smpl/SMPL_MALE.pkl",
        mode="train",
        num_steps=8,
        max_gap=4,
        channel=4,  # corresponds to frame
        img_size=256,
        modality="events",
        augmentation=True,
        use_mmhpsd=False,
        use_mmhpsd_synthesis=True,
        use_h36m=False,
        use_amass=False,
        use_phspd=False,
    )
    # for i in range(len(data_train)):
    #     if i % 1000 == 0:
    #         print(i, len(data_train))

    #     sample = data_train[i]
    #     for k, v in sample.items():
    #         if k == "info":
    #             continue
    #         if torch.isnan(v).any():
    #             print(sample["info"])

    # data_train.mmhpsd_dataloader.visualize(8000, "/home/shihao/demo_events")

    # data_test = EventDataloader(
    #     data_dir="/data/shihao",
    #     smpl_dir="../smpl_model/basicModel_m_lbs_10_207_0_v1.0.0.pkl",
    #     mode="test",
    #     num_steps=64,
    #     max_gap=4,
    #     channel=4,  # corresponds to frame
    #     img_size=256,
    #     modality="all",
    #     augmentation=False,
    #     use_mmhpsd=True,
    #     use_h36m=False,
    #     use_amass=False,
    #     use_phspd=False,
    # )
