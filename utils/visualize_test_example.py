import numpy as np
from dataset.util import render_model
import os
import cv2


def visualize_events_motion(
    events, verts, faces, cam_intr, samples_info, output_dir, epoch, vis_sample_idx=1
):
    # events_frames: [B, T, C, H, W]
    # verts: [B, T+1, 6890, 3]
    # faces: [N, 3]
    os.makedirs("{}/imgs".format(output_dir), exist_ok=True)

    vis_events = events[:vis_sample_idx]
    vis_events = np.transpose(
        vis_events.cpu().numpy(), [0, 1, 3, 4, 2]
    )  # [B, T, H, W, C]

    B, T, H, W, C = vis_events.shape
    events_frame = np.zeros([B, T, H, W, 3], dtype=np.uint8) + 255
    events_frame[np.sum(vis_events, axis=-1) > 0, :] = np.array(
        [87, 148, 191], dtype=np.uint8
    )
    # append last frame
    events_frame = np.concatenate(
        [events_frame, np.zeros([B, 1, H, W, 3], dtype=np.uint8) + 255], axis=1
    )
    # [B, T+1, H, W, 3]  -> [B, H, T+1, W, 3] -> [B, H, (T+1)*W, 3]
    events_frame = np.reshape(
        np.transpose(events_frame, [0, 2, 1, 3, 4]), [B, H, (T + 1) * W, 3]
    )

    # verts: [B, T+1, 6890, 3]
    vis_verts = verts[:vis_sample_idx].cpu().numpy()
    # faces: [N, 3]
    faces = faces.cpu().numpy()

    for i in range(B):
        # [H, (T+1)*W, 3]
        frame = events_frame[i]

        # render each time step
        render_imgs = []
        for t in range(vis_verts.shape[1]):
            render_img = (
                render_model(
                    vis_verts[i, t],
                    faces,
                    H,
                    W,
                    cam_intr,
                    np.zeros([3]),
                    np.zeros([3]),
                    near=0.1,
                    far=20,
                    img=None,
                )[:, :, 0:3]
                * 255
            ).astype(np.uint8)
            render_imgs.append(render_img)
        # [T+1, H, W, 3] -> [H, (T+1)*W, 3]
        render_imgs = np.stack(render_imgs, axis=0)
        render_imgs = np.reshape(
            np.transpose(render_imgs, [1, 0, 2, 3]), [H, (T + 1) * W, 3]
        )

        # save image
        # sample["info"] = [dataset_name, action, frame_idx, gap]
        dataset_name = samples_info[0][i]
        action = samples_info[1][i]
        start_index = samples_info[2][i]
        gap = samples_info[3][i]
        end_index = start_index + (T + 1) * gap
        seq_name = "epoch{:02d}_{}_{}_{:04d}_{:04d}".format(
            epoch, dataset_name, action, start_index, end_index
        )

        img = np.concatenate([frame, render_imgs], axis=0)
        cv2.imwrite("{}/imgs/{}.jpg".format(output_dir, seq_name), img[:, :, ::-1])


def visualize_events_motion_inference(
    rgb,
    events,
    verts,
    faces,
    cam_intr,
    samples_info,
    pred_data,
    output_dir="../examples",
    save_npz=False,
):
    os.makedirs("{}/web_demo_imgs".format(output_dir), exist_ok=True)
    os.makedirs("{}/preds".format(output_dir), exist_ok=True)
    ################## prepare rgb imgs #######################
    vis_rgb = np.transpose(rgb.cpu().numpy(), [0, 1, 3, 4, 2])  # [B, T, H, W, C]
    vis_rgb = (vis_rgb * 255).astype(np.uint8)
    B, T, H, W, C = vis_rgb.shape
    # [B, T+1, H, W, 3]  -> [B, H, T+1, W, 3] -> [B, H, (T+1)*W, 3]
    vis_rgb = np.reshape(np.transpose(vis_rgb, [0, 2, 1, 3, 4]), [B, H, T * W, C])

    ################# prepare events frame ###################
    # [B, T, C, H, W] -> [B, T, H, W, C]
    vis_events = np.transpose(events.cpu().numpy(), [0, 1, 3, 4, 2])
    B, T, H, W, C = vis_events.shape

    events_frame = np.zeros([B, T, H, W, 3], dtype=np.uint8) + 255
    events_frame[np.sum(vis_events, axis=-1) > 0, :] = np.array(
        [87, 148, 191], dtype=np.uint8
    )
    # [B, T, H, W, 3]  -> [B, H, T, W, 3] -> [B, H, T*W, 3]
    events_frame = np.reshape(
        np.transpose(events_frame, [0, 2, 1, 3, 4]), [B, H, T * W, 3]
    )

    ################# prepare smpl verts ###################
    # verts: [B, T+1, 6890, 3]
    vis_verts = verts.cpu().numpy()
    # faces: [N, 3]
    faces = faces.cpu().numpy()

    pred_theta_rotmats, pred_beta, pred_trans, pred_joints3d, pred_verts = pred_data

    for i in range(B):
        # [H, T*W, 3]
        events_imgs = events_frame[i]
        rgb_imgs = vis_rgb[i]

        # render each time step
        render_pred_imgs = []
        for t in range(T):
            render_pred_img = (
                render_model(
                    vis_verts[i, t],
                    faces,
                    H,
                    W,
                    cam_intr,
                    np.zeros([3]),
                    np.zeros([3]),
                    near=0.1,
                    far=20,
                    img=None,
                )[:, :, 0:3]
                * 255
            ).astype(np.uint8)
            render_pred_imgs.append(render_pred_img)

        # [T, H, W, 3] -> [H, T*W, 3]
        render_pred_imgs = np.stack(render_pred_imgs, axis=0)
        render_pred_imgs = np.reshape(
            np.transpose(render_pred_imgs, [1, 0, 2, 3]), [H, T * W, 3]
        )

        # save image
        # sample["info"] = [clip_filename, img_start_idx, img_end_idx,
        # events_start_idx, events_end_idx, ]
        clip_filename = samples_info[0][i]
        img_start_idx = samples_info[1][i]
        events_start_idx = samples_info[3][i]
        seq_name = "{}_img_{:04d}_events_{:04d}".format(
            clip_filename, img_start_idx, events_start_idx
        )

        img = np.concatenate([rgb_imgs, events_imgs, render_pred_imgs], axis=0)
        cv2.imwrite(
            "{}/web_demo_imgs/{}.jpg".format(output_dir, seq_name), img[:, :, ::-1]
        )

        if save_npz:
            np.savez_compressed(
                "{}/preds/{}.npz".format(output_dir, seq_name),
                pred_beta=pred_beta[i],
                pred_theta_rotmats=pred_theta_rotmats[i],
                pred_trans=pred_trans[i],
                pred_joints3d=pred_joints3d[i],
                pred_verts=pred_verts[i],
            )


def visualize_events_motion_test(
    rgb,
    events,
    verts_gt,
    verts,
    faces,
    cam_intr,
    samples_info,
    output_dir,
    gt_data,
    pred_data,
    save_npz=False,
    vis_sample_idx=None,
):
    # events/rgb: [B, T, C, H, W]
    # verts: [B, T+1, 6890, 3]
    # faces: [N, 3]
    os.makedirs("{}/imgs".format(output_dir), exist_ok=True)
    os.makedirs("{}/preds".format(output_dir), exist_ok=True)
    if not vis_sample_idx:
        # batch size
        vis_sample_idx = events.shape[0]

    ################## prepare rgb imgs #######################
    vis_rgb = rgb[:vis_sample_idx]
    vis_rgb = np.transpose(vis_rgb.cpu().numpy(), [0, 1, 3, 4, 2])  # [B, T, H, W, C]
    vis_rgb = (vis_rgb * 255).astype(np.uint8)

    B, T_rgb, H, W, C = vis_rgb.shape
    # [B, T+1, H, W, 3]  -> [B, H, T+1, W, 3] -> [B, H, (T+1)*W, 3]
    vis_rgb = np.reshape(np.transpose(vis_rgb, [0, 2, 1, 3, 4]), [B, H, T_rgb * W, C])

    ################# prepare events frame ###################
    vis_events = events[:vis_sample_idx]
    # [B, T, H, W, C]
    vis_events = np.transpose(vis_events.cpu().numpy(), [0, 1, 3, 4, 2])
    B, T, H, W, C = vis_events.shape

    events_frame = np.zeros([B, T, H, W, 3], dtype=np.uint8) + 255
    events_frame[np.sum(vis_events, axis=-1) > 0, :] = np.array(
        [87, 148, 191], dtype=np.uint8
    )
    # append last frame
    events_frame = np.concatenate(
        [events_frame, np.zeros([B, 1, H, W, 3], dtype=np.uint8) + 255], axis=1
    )
    # [B, T+1, H, W, 3]  -> [B, H, T+1, W, 3] -> [B, H, (T+1)*W, 3]
    events_frame = np.reshape(
        np.transpose(events_frame, [0, 2, 1, 3, 4]), [B, H, (T + 1) * W, 3]
    )

    ################# prepare smpl verts ###################
    # verts: [B, T+1, 6890, 3]
    vis_verts_gt = verts_gt[:vis_sample_idx].cpu().numpy()
    vis_verts = verts[:vis_sample_idx].cpu().numpy()
    # faces: [N, 3]
    faces = faces.cpu().numpy()

    gt_theta, gt_beta, gt_trans, gt_joints3d, gt_verts = gt_data
    pred_theta_rotmats, pred_beta, pred_trans, pred_joints3d, pred_verts = pred_data

    for i in range(B):
        # save image
        # sample["info"] = [dataset_name, action, frame_idx, gap]
        dataset_name = samples_info[0][i]
        action = samples_info[1][i]
        start_index = samples_info[2][i]
        gap = samples_info[3][i]
        end_index = start_index + (T + 1) * gap
        seq_name = "{}_{}_{:04d}_{:04d}".format(
            dataset_name, action, start_index, end_index
        )
        if "subject01" in action or "subject07" in action:
            continue

        # [H, (T+1)*W, 3]
        events_imgs = events_frame[i]
        rgb_imgs = vis_rgb[i]

        # render each time step
        render_gt_imgs, render_pred_imgs = [], []
        for t in range(T + 1):
            # render gt imgs
            render_gt_img = (
                render_model(
                    vis_verts_gt[i, t],
                    faces,
                    H,
                    W,
                    cam_intr,
                    np.zeros([3]),
                    np.zeros([3]),
                    near=0.1,
                    far=20,
                    img=None,
                )[:, :, 0:3]
                * 255
            ).astype(np.uint8)
            render_gt_imgs.append(render_gt_img)

            # render pred imgs
            render_pred_img = (
                render_model(
                    vis_verts[i, t],
                    faces,
                    H,
                    W,
                    cam_intr,
                    np.zeros([3]),
                    np.zeros([3]),
                    near=0.1,
                    far=20,
                    img=None,
                )[:, :, 0:3]
                * 255
            ).astype(np.uint8)
            render_pred_imgs.append(render_pred_img)

        # [T+1, H, W, 3] -> [H, (T+1)*W, 3]
        render_pred_imgs = np.stack(render_pred_imgs, axis=0)
        render_pred_imgs = np.reshape(
            np.transpose(render_pred_imgs, [1, 0, 2, 3]), [H, (T + 1) * W, 3]
        )

        render_gt_imgs = np.stack(render_gt_imgs, axis=0)
        render_gt_imgs = np.reshape(
            np.transpose(render_gt_imgs, [1, 0, 2, 3]), [H, (T + 1) * W, 3]
        )

        img = np.concatenate(
            [rgb_imgs, events_imgs, render_gt_imgs, render_pred_imgs], axis=0
        )
        color = (255, 0, 0)
        cv2.putText(
            img,
            "Ground-truth",
            (20, 100 + 256 * 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color=color,
            thickness=2,
        )
        cv2.putText(
            img,
            "Prediction",
            (20, 100 + 256 * 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color=color,
            thickness=2,
        )
        cv2.imwrite("{}/imgs/{}.jpg".format(output_dir, seq_name), img[:, :, ::-1])

        if save_npz:
            np.savez_compressed(
                "{}/preds/{}.npz".format(output_dir, seq_name),
                gt_beta=gt_beta[i],
                gt_theta=gt_theta[i],
                gt_trans=gt_trans[i],
                gt_joints3d=gt_joints3d[i],
                gt_verts=gt_verts[i],
                pred_beta=pred_beta[i],
                pred_theta_rotmats=pred_theta_rotmats[i],
                pred_trans=pred_trans[i],
                pred_joints3d=pred_joints3d[i],
                pred_verts=pred_verts[i],
            )


def visualize_rgb_motion(
    rgb, verts, faces, cam_intr, samples_info, output_dir, epoch, vis_sample_idx=1
):
    # rgb: [B, T, C, H, W]
    # verts: [B, T+1, 6890, 3]
    # faces: [N, 3]
    os.makedirs("{}/imgs".format(output_dir), exist_ok=True)

    vis_rgb = rgb[:vis_sample_idx]
    vis_rgb = np.transpose(vis_rgb.cpu().numpy(), [0, 1, 3, 4, 2])  # [B, T, H, W, C]
    vis_rgb = (vis_rgb * 255).astype(np.uint8)

    B, T, H, W, C = vis_rgb.shape
    # append last frame
    vis_rgb = np.concatenate(
        [vis_rgb, np.zeros([B, 1, H, W, C], dtype=np.uint8) + 255], axis=1
    )
    # [B, T+1, H, W, 3]  -> [B, H, T+1, W, 3] -> [B, H, (T+1)*W, 3]
    vis_rgb = np.reshape(np.transpose(vis_rgb, [0, 2, 1, 3, 4]), [B, H, (T + 1) * W, C])

    # verts: [B, T+1, 6890, 3]
    vis_verts = verts[:vis_sample_idx].cpu().numpy()
    # faces: [N, 3]
    faces = faces.cpu().numpy()

    for i in range(B):
        # [H, (T+1)*W, 3]
        frame = vis_rgb[i]

        # render each time step
        render_imgs = []
        for t in range(vis_verts.shape[1]):
            render_img = (
                render_model(
                    vis_verts[i, t],
                    faces,
                    H,
                    W,
                    cam_intr,
                    np.zeros([3]),
                    np.zeros([3]),
                    near=0.1,
                    far=20,
                    img=None,
                )[:, :, 0:3]
                * 255
            ).astype(np.uint8)
            render_imgs.append(render_img)
        # [T+1, H, W, 3] -> [H, (T+1)*W, 3]
        render_imgs = np.stack(render_imgs, axis=0)
        render_imgs = np.reshape(
            np.transpose(render_imgs, [1, 0, 2, 3]), [H, (T + 1) * W, C]
        )

        # save image
        # sample["info"] = [dataset_name, action, frame_idx, gap]
        dataset_name = samples_info[0][i]
        action = samples_info[1][i]
        start_index = samples_info[2][i]
        gap = samples_info[3][i]
        end_index = start_index + (T + 1) * gap
        seq_name = "epoch{:02d}_{}_{}_{:04d}_{:04d}".format(
            epoch, dataset_name, action, start_index, end_index
        )

        img = np.concatenate([frame, render_imgs], axis=0)
        cv2.imwrite("{}/imgs/{}.jpg".format(output_dir, seq_name), img[:, :, ::-1])
