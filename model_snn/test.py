import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from spikingjelly.activation_based import neuron, functional
import os
import time
import numpy as np
import random
import sys

sys.path.append("../")
from dataset.dataloader import EventDataloader
from utils.loss_funcs import (
    compute_losses,
    compute_mpjpe,
    compute_pa_mpjpe,
    compute_pelvis_mpjpe,
)

from utils.visualize_test_example import visualize_events_motion_test
import collections

from model_snn.spiking_model import SpikePoseNet
from model_snn import misc


def test(args):
    # misc.init_distributed_mode(args)
    device = torch.device(args.device)
    # torch.cuda.set_device(device)
    dtype = torch.float32

    # testset do not need sampler
    dataset_test = EventDataloader(
        data_dir=args.data_dir,
        smpl_dir=args.smpl_dir,
        mode="test",
        num_steps=args.num_frames,
        max_gap=args.max_gap,
        channel=args.channel,  # corresponds to frame
        img_size=args.img_size,
        modality="all",
        augmentation=False,
        use_mmhpsd=args.use_mmhpsd,
        use_mmhpsd_synthesis=args.use_mmhpsd_synthesis,
        use_h36m=args.use_h36m,
        use_amass=args.use_amass,
        use_phspd=args.use_phspd,
    )
    test_generator = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_worker,
        pin_memory=args.pin_memory,
    )
    print("[smpl_dir] {}".format(args.smpl_dir))

    # build model_SpikeNN
    model = SpikePoseNet(
        num_frames=args.num_frames,
        channel=args.channel,
        model_name=args.backbone,
        return_interm_layers=True,
        use_tc=False,  # if true, reshape events [B, T, C, H, W] (t=T) to [B, T*C, 1, H, W] (t=T*C)
        spiking_neuron=args.neuron,
        surrogate_function=args.surrogate,
        cnf=args.cnf,
        drop_prob=args.drop_prob,
        n_layers=args.n_layers,
        detach_reset=bool(args.detach_reset),
        # hard_reset: reset to 0, soft_reset: substract v_threshold
        v_reset=0.0 if args.hard_reset else None,
        cam_intr=dataset_test.cam_intr,
        img_size=args.img_size,
        smpl_dir=args.smpl_dir,
        batch_size=args.batch_size,
        use_rnn=args.use_rnn,
        use_recursive=args.use_recursive,
        use_transformer=args.use_transformer,
        n_head=args.n_head,
    )
    mse_func = torch.nn.MSELoss()

    # set DDP
    model.to(device)
    # model_without_ddp = model
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model, device_ids=[args.rank], output_device=args.rank
    #     )
    #     model_without_ddp = model.module
    functional.reset_net(model)
    functional.set_step_mode(model, "m")
    functional.set_backend(model, "cupy", getattr(neuron, args.neuron))
    # print(model)

    # resume
    if args.model_dir:  # load checkpoint
        print("[model_snn dir] model_snn loaded from {}".format(args.model_dir))
        checkpoint = torch.load(args.model_dir, map_location="cpu")
        state_dict = {}
        for k, v in checkpoint["model_state_dict"].items():
            state_dict[k.replace("module.", "")] = v
        model.load_state_dict(state_dict)
        start_epoch = checkpoint["epoch"] + 1

    print(
        "------------------------------------- test -----------------------------------------"
    )
    model.eval()
    results = collections.defaultdict(list)
    all_spiking_rates = collections.defaultdict(list)
    start_time = time.time()
    # deactivate autograd to reduce memory usage
    with torch.set_grad_enabled(False):
        for iter, data in enumerate(test_generator):
            print("Test Iter {} / {}".format(iter, len(test_generator)))
            # data: {events, theta, beta, trans, joints2d, joints3d, info}
            for k in data.keys():
                if k != "info":
                    data[k] = data[k].to(device=device, dtype=dtype)

            out = model(data["events"])  # [B, T, C, H, W]
            functional.reset_net(model)

            loss_dict = compute_losses(out, data, mse_func, device, args)
            mpjpe = compute_mpjpe(
                out["joints3d"].detach(), data["joints3d"]
            )  # [B, T, 24]
            pa_mpjpe = compute_pa_mpjpe(
                out["joints3d"].detach(), data["joints3d"]
            )  # [B, T, 24]
            pel_mpjpe = compute_pelvis_mpjpe(
                out["joints3d"].detach(), data["joints3d"]
            )  # [B, T, 24]
            loss = (
                loss_dict["trans"]
                + loss_dict["theta"]
                + loss_dict["beta"]
                + loss_dict["joints3d"]
                + loss_dict["joints2d"]
            )

            if args.save_attention_map:
                os.makedirs("{}/preds".format(args.output_dir), exist_ok=True)
                for i in range(data["events"].shape[0]):
                    score = out["attention_score"][i].cpu().numpy()
                    dataset_name = data["info"][0][i]
                    action = data["info"][1][i]
                    start_index = data["info"][2][i]
                    seq_name = "{}_{}_{:04d}_att_maps".format(
                        dataset_name, action, start_index
                    )
                    np.savez_compressed(
                        "{}/preds/{}.npz".format(args.output_dir, seq_name),
                        attention_score=score,
                    )

            if args.save_vis:
                # collect data to save as npz
                gt_data = [
                    data["theta"].cpu().numpy(),
                    data["beta"].cpu().numpy(),
                    data["trans"].cpu().numpy(),
                    data["joints3d"].cpu().numpy(),
                    data["verts"].cpu().numpy(),
                ]
                pred_data = [
                    out["pred_rotmats"].detach().cpu().numpy(),
                    out["beta"].detach().cpu().numpy(),
                    out["trans"].detach().cpu().numpy(),
                    out["joints3d"].detach().cpu().numpy(),
                    out["verts"].detach().cpu().numpy(),
                ]

                # align verts before render and visualize
                pred_joints3d = out["joints3d"].detach()  # [B, T, 24, 3]
                target_joints3d = data["joints3d"].detach()  # [B, T, 24, 3]
                left_heap_idx, right_heap_idx = 1, 2
                pred_pel = (
                    pred_joints3d[:, :, left_heap_idx : left_heap_idx + 1, :]
                    + pred_joints3d[:, :, right_heap_idx : right_heap_idx + 1, :]
                ) / 2
                target_pel = (
                    target_joints3d[:, :, left_heap_idx : left_heap_idx + 1, :]
                    + target_joints3d[:, :, right_heap_idx : right_heap_idx + 1, :]
                ) / 2
                pred_verts = out["verts"]
                pred_verts = pred_verts - pred_pel + target_pel

                visualize_events_motion_test(
                    data["imgs"],
                    data["events"],
                    data["verts"],
                    pred_verts,
                    out["faces"],
                    out["cam_intr"],
                    data["info"],
                    args.output_dir,
                    gt_data=gt_data,
                    pred_data=pred_data,
                    save_npz=True,
                )

            # collect results
            results["scalar/trans"].append(loss_dict["trans"].detach())
            results["scalar/theta"].append(loss_dict["theta"].detach())
            results["scalar/beta"].append(loss_dict["beta"].detach())
            results["scalar/joints3d"].append(loss_dict["joints3d"].detach())
            results["scalar/joints2d"].append(loss_dict["joints2d"].detach())
            results["scalar/loss"].append(loss.detach())
            results["scalar/mpjpe"].append(torch.mean(mpjpe.detach()))
            results["scalar/pa_mpjpe"].append(torch.mean(pa_mpjpe.detach()))
            results["scalar/pel_mpjpe"].append(torch.mean(pel_mpjpe.detach()))
            for k, v in out["layer_spiking_rates"].items():
                all_spiking_rates[k].append(v)

        results["trans"] = torch.mean(torch.stack(results["scalar/trans"], dim=0))
        results["theta"] = torch.mean(torch.stack(results["scalar/theta"], dim=0))
        results["beta"] = torch.mean(torch.stack(results["scalar/beta"], dim=0))
        results["joints3d"] = torch.mean(torch.stack(results["scalar/joints3d"], dim=0))
        results["joints2d"] = torch.mean(torch.stack(results["scalar/joints2d"], dim=0))
        results["loss"] = torch.mean(torch.stack(results["scalar/loss"], dim=0))
        results["mpjpe"] = torch.mean(torch.stack(results["scalar/mpjpe"], dim=0))
        results["pa_mpjpe"] = torch.mean(torch.stack(results["scalar/pa_mpjpe"], dim=0))
        results["pel_mpjpe"] = torch.mean(
            torch.stack(results["scalar/pel_mpjpe"], dim=0)
        )
        results["layer_spiking_rates"] = " ".join(
            [
                "{}:{:.4f}".format(k, torch.mean(torch.stack(all_spiking_rates[k])))
                for k in all_spiking_rates.keys()
            ]
        )

        # macs, params = clever_format([macs, params], "%.3f")
        # print(macs, params, spiking_rate)

        end_time = time.time()
        time_used = (end_time - start_time) / 60.0
        print(
            ">>> loss {:.4f}, tran {:.4f}, theta {:.4f}, theta {:.4f}, "
            "joints3d {:.4f}, joints2d {:.4f} \n"
            "    spiking rate {} \n"
            "    time used: {:.2f} mins \n"
            "    mpjpe {:.4f}, pa_mpjpe {:.4f}, pel_mpjpe {:.4f} ".format(
                results["loss"],
                results["trans"],
                results["theta"],
                results["beta"],
                results["joints3d"],
                results["joints2d"],
                results["layer_spiking_rates"],
                time_used,
                1000 * results["mpjpe"],
                1000 * results["pa_mpjpe"],
                1000 * results["pel_mpjpe"],
            )
        )


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    # training environemnt and setting
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--visible_gpus", type=str, default="2,3")
    parser.add_argument("--data_dir", type=str, default="/home/shihao/data")
    parser.add_argument(
        "--output_dir", type=str, default="/data/shihao/exp_eventformer"
    )
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument(
        "--smpl_dir",
        type=str,
        default="../smpl_model/models/smpl/SMPL_MALE.pkl",
    )
    parser.add_argument("--num_worker", type=int, default=4)
    parser.add_argument("--pin_memory", type=int, default=0)
    parser.add_argument("--use_amp", type=int, default=0)
    parser.add_argument("--save_vis", type=int, default=0)
    parser.add_argument("--save_attention_map", type=int, default=0)

    # dataloader parameters
    parser.add_argument("--channel", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=64)
    parser.add_argument("--max_gap", type=int, default=4)
    parser.add_argument("--use_mmhpsd", type=int, default=1)
    parser.add_argument("--use_mmhpsd_synthesis", type=int, default=0)
    parser.add_argument("--use_h36m", type=int, default=0)
    parser.add_argument("--use_amass", type=int, default=0)
    parser.add_argument("--use_phspd", type=int, default=0)
    parser.add_argument("--use_augmentation", type=int, default=1)
    parser.add_argument("--use_geodesic_loss", type=int, default=1)

    # Spiking model
    parser.add_argument("--backbone", type=str, default="sew_resnet50")
    parser.add_argument("--neuron", type=str, default="IFNode")
    parser.add_argument("--surrogate", type=str, default="ATan")
    parser.add_argument("--detach_reset", type=int, default=1)
    parser.add_argument("--hard_reset", type=int, default=0)
    parser.add_argument("--cnf", type=str, default="AND")
    parser.add_argument("--drop_prob", type=float, default=0.1)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--use_rnn", type=int, default=0)
    parser.add_argument("--use_recursive", type=int, default=1)
    parser.add_argument("--use_transformer", type=int, default=0)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--d_hidden", type=int, default=1024)

    # loss
    parser.add_argument("--trans_loss", type=float, default=10)
    parser.add_argument("--theta_loss", type=float, default=10)
    parser.add_argument("--beta_loss", type=float, default=1)
    parser.add_argument("--joints3d_loss", type=float, default=1)
    parser.add_argument("--joints2d_loss", type=float, default=10)

    # train parameters
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_regressor", type=float, default=0.1)
    parser.add_argument("--lr_scheduler", type=str, default="CosineAnnealingLR")
    parser.add_argument("--lr_decay_rate", type=float, default=0.1)
    parser.add_argument("--lr_decay_step", type=float, default=20)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=-1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--local_rank", default=-1, type=int, help="node rank for distributed training"
    )

    args = parser.parse_args()
    return args


def print_args(args):
    """Prints the argparse argmuments applied
    Args:
      args = parser.parse_args()
    """
    _args = vars(args)
    max_length = max([len(k) for k, _ in _args.items()])
    for k, v in _args.items():
        print(" " * (max_length - len(k)) + k + ": " + str(v))


def main():
    seed = 666
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # set environment
    args = get_args()
    print_args(args)
    os.environ["OMP_NUM_THREADS"] = "1"
    # https://discuss.pytorch.org/t/distributed-data-parallel-freezes-without-error-message/8009/27
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    torch.set_num_threads(1)

    # train
    test(args)


if __name__ == "__main__":
    main()
