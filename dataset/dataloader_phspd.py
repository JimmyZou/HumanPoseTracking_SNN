import os
import numpy as np
import cv2
from torch.utils.data import Dataset
import joblib
import torch
import sys

sys.path.append("../")
from utils.SMPL import SMPL
from dataset.util import projection_np
from dataset.data_transforms import (
    get_aug_config,
    generate_patch_image,
    trans_point2d,
    estimate_translation_np,
    rotate_2d_matrix,
)

# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
# FLIP_IDX = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]


class PHSPDataloader(Dataset):
    def __init__(
        self,
        data_dir="/data/shihao",
        smpl_dir="../../smpl_model/models/smpl/SMPL_MALE.pkl",
        mode="train",
        num_steps=64,
        max_gap=4,
        channel=4,  # corresponds to frame
        img_size=256,
        modality="events",
        augmentation=False,
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
        self.all_clips = self.obtain_all_clips()
        self.aug = augmentation

        self.device = torch.device("cpu")
        self.smplmodel = SMPL(smpl_dir, self.num_steps + 1).to(self.device)

    def __len__(self):
        return len(self.all_clips)

    def __getitem__(self, idx):
        dataset_name, action, frame_idx, _ = self.all_clips[idx]
        gap = np.random.randint(1, self.max_gap)

        if self.aug:
            # get augmentation configuration
            rot, do_flip, bbx, scale, trans2d = get_aug_config(
                (self.img_size, self.img_size), (self.img_size, self.img_size)
            )
            # print(rot, do_flip, bbx, scale, trans2d)
        else:
            do_flip = False

        events, imgs, joints3d, joints2d, trans, theta, beta = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for t in range(self.num_steps + 1):
            idx = frame_idx + t * gap

            # load pose: joints3d=joints3d, joints2d=joints2d, trans=trans, body_pose=theta, beta=beta
            if do_flip:
                pose_t = np.load(
                    "{}/phspd_events/{}/poses_flip/pose_{:04d}.npz".format(
                        self.data_dir, action, idx
                    )
                )
            else:
                pose_t = np.load(
                    "{}/phspd_events/{}/poses/pose_{:04d}.npz".format(
                        self.data_dir, action, idx
                    )
                )
            joints3d.append(pose_t["joints3d"])  # [24, 3]
            joints2d.append(pose_t["joints2d"][:, 0:2])  # [24, 3]
            trans.append(pose_t["trans"])  # [1, 3]
            theta.append(pose_t["body_pose"])  # [1, 72]
            beta.append(pose_t["beta"])  # [1, 10]

            if self.modality == "rgb" or self.modality == "all":
                img_name = "{}/phspd_rgb/{}/imgs/color_{:04d}.jpg".format(
                    self.data_dir, action, idx
                )
                img = cv2.imread(img_name)[:, :, ::-1]
                if img.shape[0] != self.img_size:
                    img = cv2.resize(img, (self.img_size, self.img_size))
                imgs.append(img)

            if self.modality == "events" or self.modality == "all":
                # N inter events -- N+1 poses
                if t == self.num_steps:
                    break

                # load events frame
                events_frames = []
                for i in range(gap):
                    frame_name = "{}/phspd_events/{}/events_frame_{:02d}/events_{:04d}.npz".format(
                        self.data_dir, action, self.channel, idx + i
                    )
                    events_frame_t = (
                        np.unpackbits(np.load(frame_name)["events_frame"])
                        .reshape(self.img_size, self.img_size, self.channel)
                        .astype(np.float32)
                    )
                    events_frames.append(events_frame_t)
                events_frames = np.concatenate(
                    events_frames, axis=-1
                )  # [H, W, C * gap]
                # [H, W, C * gap] -> [H, W, gap, C] -> [H, W, C]
                events_frames = np.sum(
                    events_frames.reshape(
                        self.img_size, self.img_size, gap, self.channel
                    ),
                    axis=2,
                )
                events.append((events_frames > 0).astype(np.float32))

        joints3d = np.stack(joints3d, axis=0)  # [T+1, 24, 3]
        joints2d = np.stack(joints2d, axis=0)  # [T+1, 24, 2]
        trans = np.stack(trans, axis=0)  # [T+1, 1, 3]
        theta = np.concatenate(theta, axis=0)  # [T+1, 72]
        beta = np.concatenate(beta, axis=0)  # [T+1, 10]
        if self.modality == "events" or self.modality == "all":
            events = np.stack(events, axis=0)  # [T, H, W, C]
        if self.modality == "rgb" or self.modality == "all":
            imgs = np.stack(imgs, axis=0).astype(np.float32) / 255.0  # [T+1, H, W, C]

        # print(theta.shape, beta.shape, trans.shape, joints2d.shape, joints3d.shape, events.shape, imgs.shape)

        if self.aug:
            # transform events frame
            if self.modality == "events" or self.modality == "all":
                # events [T, H, W, C] -> [H, W, T, C]
                events_aug = np.transpose(events, [1, 2, 0, 3]).reshape(
                    [self.img_size, self.img_size, -1]
                )

                # transform events frame
                events_aug = generate_patch_image(
                    events_aug, do_flip, trans2d, (self.img_size, self.img_size)
                )
                events_aug = np.transpose(
                    events_aug.reshape(
                        [self.img_size, self.img_size, self.num_steps, self.channel]
                    ),
                    [2, 0, 1, 3],
                )

            if self.modality == "rgb" or self.modality == "all":
                # transform rgb
                imgs_aug = np.transpose(imgs, [1, 2, 0, 3]).reshape(
                    [self.img_size, self.img_size, -1]
                )
                imgs_aug = generate_patch_image(
                    imgs_aug, do_flip, trans2d, (self.img_size, self.img_size)
                )
                imgs_aug = np.transpose(
                    imgs_aug.reshape(
                        [self.img_size, self.img_size, self.num_steps + 1, 3]
                    ),
                    [2, 0, 1, 3],
                )

            # transform 2d joints
            joints2d_aug = trans_point2d(joints2d, trans2d)

            # # transform 3d joints
            # trans_aug, aug_joints3d = [], []
            # for t in range(joints3d.shape[0]):
            #     joints3d_t = joints3d[t]
            #     # rotation_mat = rotate_2d_matrix(-np.pi * rot / 180)
            #     # joints3d[:, 0:2] = np.dot(rotation_mat, joints3d[:, 0:2].T).T
            #     joints3d_t[:, 0:2] = rotate_2d(joints3d_t[:, 0:2].T, -np.pi * rot / 180).T
            #     aug_trans_t = estimate_translation_np(
            #         joints3d_t, joints2d_aug[t], np.ones([joints3d_t.shape[0]]), self.cam_intr)
            #     aug_trans_t = np.expand_dims(aug_trans_t, axis=0)
            #     trans_aug.append(aug_trans_t)
            #     joints3d_t = joints3d_t + aug_trans_t
            #     aug_joints3d.append(joints3d_t)
            # trans_aug = np.stack(trans_aug, axis=0) + trans
            # joints3d_aug = np.stack(aug_joints3d, axis=0)

            # transform theta and get new joints3d and translation
            rotation_mat = np.eye(3)  # [3, 3]
            rotation_mat[0:2, 0:2] = rotate_2d_matrix(-np.pi * rot / 180)
            theta_aug = []
            for t in range(theta.shape[0]):
                theta_t = theta[t]
                r = cv2.Rodrigues(theta_t[0:3])[0]
                r_aug = np.dot(rotation_mat, r)
                theta_t[0:3] = cv2.Rodrigues(r_aug)[0][:, 0]
                theta_aug.append(theta_t)
            theta_aug = np.stack(theta_aug, axis=0)

            with torch.no_grad():
                verts, joints3d, _ = self.smplmodel(
                    beta=torch.from_numpy(beta).float().to(self.device),
                    theta=torch.from_numpy(theta_aug).float().to(self.device),
                    get_skin=True,
                )
                joints3d = joints3d.detach().cpu().numpy()
                verts = verts.detach().cpu().numpy()

            trans_aug = []
            for t in range(joints3d.shape[0]):
                joints3d_t = joints3d[t]
                aug_trans_t = estimate_translation_np(
                    joints3d_t,
                    joints2d_aug[t],
                    np.ones([joints3d_t.shape[0]]),
                    self.cam_intr,
                )
                aug_trans_t = np.expand_dims(aug_trans_t, axis=0)
                trans_aug.append(aug_trans_t)
            trans_aug = np.stack(trans_aug, axis=0)
            joints3d_aug = joints3d + trans_aug
            verts_aug = verts + trans_aug

            sample = {}
            sample["theta"] = torch.from_numpy(theta_aug).float()  # [T+1, 72]
            sample["beta"] = torch.from_numpy(beta).float()  # [T+1, 10]
            sample["trans"] = torch.from_numpy(trans_aug).float()  # [T+1, 1, 3]
            sample["joints2d"] = torch.from_numpy(
                joints2d_aug / self.img_size
            ).float()  # [T+1, 24, 2]
            sample["joints3d"] = torch.from_numpy(joints3d_aug).float()  # [T+1, 24, 3]
            sample["verts"] = torch.from_numpy(verts_aug).float()  # [T+1, 6890, 3]
            sample["info"] = [dataset_name, action, frame_idx, gap]
            if self.modality == "events" or self.modality == "all":
                sample["events"] = torch.from_numpy(
                    np.transpose(events_aug, [0, 3, 1, 2])
                ).float()  # [T, C, H, W]
            if self.modality == "rgb" or self.modality == "all":
                sample["imgs"] = torch.from_numpy(
                    np.transpose(imgs_aug, [0, 3, 1, 2])
                ).float()  # [T+1, C, H, W]
            # print(theta_aug.shape, beta.shape, trans_aug.shape, joints2d_aug.shape,
            #       joints3d_aug.shape, events_aug.shape, imgs_aug.shape)
        else:
            with torch.no_grad():
                verts, _, _ = self.smplmodel(
                    beta=torch.from_numpy(beta).float().to(self.device),
                    theta=torch.from_numpy(theta).float().to(self.device),
                    get_skin=True,
                )
            verts = verts.detach().cpu().numpy() + trans

            sample = {}
            sample["theta"] = torch.from_numpy(theta).float()  # [T+1, 72]
            sample["beta"] = torch.from_numpy(beta).float()  # [T+1, 10]
            sample["trans"] = torch.from_numpy(trans).float()  # [T+1, 1, 3]
            sample["joints2d"] = torch.from_numpy(
                joints2d / self.img_size
            ).float()  # [T+1, 24, 2]
            sample["joints3d"] = torch.from_numpy(joints3d).float()  # [T+1, 24, 3]
            sample["verts"] = torch.from_numpy(verts).float()  # [T+1, 6890, 3]
            sample["info"] = [dataset_name, action, frame_idx, gap]
            if self.modality == "events" or self.modality == "all":
                sample["events"] = torch.from_numpy(
                    np.transpose(events, [0, 3, 1, 2])
                ).float()  # [T, C, H, W]
            if self.modality == "rgb" or self.modality == "all":
                sample["imgs"] = torch.from_numpy(
                    np.transpose(imgs, [0, 3, 1, 2])
                ).float()  # [T+1, C, H, W]

        return sample

    def obtain_all_clips(self):
        save_filename = "{}/phspd_{}_{:02d}{:02d}.pkl".format(
            self.data_dir, self.mode, self.num_steps, self.max_gap
        )
        if os.path.exists(save_filename):
            print("load from {}".format(save_filename))
            all_clips = joblib.load(save_filename)
        else:
            all_clips = []
            tmp = sorted(os.listdir("{}/phspd_events".format(self.data_dir)))
            action_names = []
            # parameters to pick up clips from a video
            gap = 2
            skip = self.num_steps * gap
            for action in tmp:
                subject = action.split("_")[0]
                if self.mode == "test":
                    if subject in ["subject07", "subject04", "subject11"]:
                        action_names.append(action)
                else:
                    if subject not in ["subject07", "subject04", "subject11"]:
                        action_names.append(action)
                    gap = self.max_gap
                    skip = self.num_steps // 2

            count = 0
            for action in action_names:
                all_pose_files = list(
                    sorted(
                        os.listdir(
                            "{}/phspd_events/{}/poses".format(self.data_dir, action)
                        )
                    )
                )
                n = len(all_pose_files)
                for idx in range(0, n - self.num_steps * gap - 1, skip):
                    pose_file = all_pose_files[idx]

                    frame_idx = int(pose_file.split("_")[1].split(".")[0])
                    end_frame_idx = frame_idx + self.num_steps * gap + 1
                    end_pose_idx = int(
                        all_pose_files[idx + self.num_steps * gap + 1]
                        .split("_")[1]
                        .split(".")[0]
                    )
                    if end_frame_idx == end_pose_idx:
                        # action, frame_idx
                        all_clips.append(("phspd", action, frame_idx, count))
                        count += 1

            joblib.dump(all_clips, save_filename, compress=3)
            print("save as {}".format(save_filename))

        print("[phspd {}] {} samples".format(self.mode, len(all_clips)))
        return all_clips

    def visualize(self, idx, save_dir):
        assert self.modality == "all"
        sample = self.__getitem__(idx)
        dataset_name, action, frame_idx, gap = sample["info"]
        print(action, frame_idx, gap)

        # import smplx
        # device = torch.device("cpu")
        # smplmodel = smplx.create("../../smpl_model/models/", model_type="smpl",
        #                          gender="male", ext="pkl", batch_size=self.num_steps+1).to(device)
        # with torch.no_grad():
        #     outputp = smplmodel(betas=sample['beta'].to(device),
        #                         global_orient=sample['theta'][:, :3].to(device),
        #                         body_pose=sample['theta'][:, 3:].to(device),
        #                         transl=sample['trans'][:, 0, :].to(device),
        #                         return_verts=True)
        # verts = outputp.vertices.detach().cpu().numpy()  # [T+1, 6890, 3] w/ trans
        # joints3d = outputp.joints.detach().cpu().numpy()[:, 0:24, :]

        with torch.no_grad():
            verts, joints3d, _ = self.smplmodel(
                beta=sample["beta"].to(self.device),
                theta=sample["theta"].to(self.device),
                get_skin=True,
            )
        trans = sample["trans"].cpu().numpy()
        verts = verts.detach().cpu().numpy() + trans
        # joints3d = joints3d.detach().cpu().numpy() + trans

        joints2d = sample["joints2d"].cpu().numpy()  # [T+1, 24, 2]
        joints3d = sample["joints3d"].cpu().numpy()  # [T+1, 24, 3]
        events = np.transpose(
            sample["events"].cpu().numpy(), [0, 2, 3, 1]
        )  # [T, H, W, C]
        imgs = np.transpose(
            sample["imgs"].cpu().numpy(), [0, 2, 3, 1]
        )  # [T+1, H, W, C]

        from dataset.util import render_model

        frames = []
        for t in range(self.num_steps):
            img = (imgs[t] * 255).astype(np.uint8)

            render_img = (
                render_model(
                    verts[t],
                    self.smplmodel.faces,
                    self.img_size,
                    self.img_size,
                    np.asarray(self.cam_intr[0:4]),
                    np.zeros([3]),
                    np.zeros([3]),
                    near=0.1,
                    far=20,
                    img=img,
                )[:, :, 0:3]
                * 255
            ).astype(np.uint8)

            # for i in range(self.channel):
            #     events_frame = np.zeros([self.img_size, self.img_size, 3], dtype=np.uint8) + 255
            #     events_frame[events[t, :, :, i] > 0, :] = np.array([87, 148, 191], dtype=np.uint8)
            #     frame = np.concatenate([img, events_frame, render_img], axis=1)
            #     frames.append(frame)

            img_joints2d = img.copy()
            for point in joints2d[t]:
                cv2.circle(
                    img_joints2d, (int(point[0]), int(point[1])), 1, (255, 0, 0), -1
                )

            img_joints3d = img.copy()
            proj_joints2d = projection_np(joints3d[t], self.cam_intr, True)
            for point in proj_joints2d:
                cv2.circle(
                    img_joints3d, (int(point[0]), int(point[1])), 1, (0, 0, 255), -1
                )

            # print('joints2d error', np.max(np.sqrt(np.sum((joints2d[t] - proj_joints2d[:, 0:2])**2, axis=-1))))

            events_frame = (
                np.zeros([self.img_size, self.img_size, 3], dtype=np.uint8) + 255
            )
            events_frame[np.sum(events[t], axis=-1) > 0, :] = np.array(
                [87, 148, 191], dtype=np.uint8
            )
            frame = np.concatenate(
                [img, events_frame, render_img, img_joints2d, img_joints3d], axis=1
            )
            frames.append(frame)

        import imageio

        demo_name = "{}/phspd_{}_{:04d}.gif".format(save_dir, action, frame_idx)
        imageio.mimsave(demo_name, frames, "GIF", duration=gap * 1.0 / 15.0)
        print("save as {}".format(demo_name))


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    data_train = PHSPDataloader(
        data_dir="/home/shihao/data",
        smpl_dir="../smpl_model/models/smpl/SMPL_MALE.pkl",
        mode="test",
        num_steps=8,
        max_gap=4,
        channel=4,  # corresponds to frame
        img_size=256,
        modality="all",
        augmentation=False,
    )
    data_train.visualize(3666, "../data")
