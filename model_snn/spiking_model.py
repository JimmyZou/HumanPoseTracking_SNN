import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter
from spikingjelly.activation_based import (
    surrogate,
    neuron,
    functional,
    base,
    layer,
    rnn,
)
from spikingjelly.activation_based.model import sew_resnet

import sys

sys.path.append("../")
from utils.SMPL import SMPL
from utils.geometry import projection_torch, rot6d_to_rotmat


class Backbone(nn.Module):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        name="sew_resnet34",
        return_interm_layers=False,
        input_channel=1,
        spiking_neuron=neuron.ParametricLIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
        v_reset=None,
        cnf="ADD",
    ):
        super().__init__()
        print("[backbone] {}".format(name))
        backbone = getattr(sew_resnet, name)(
            replace_stride_with_dilation=None,
            pretrained=False,
            norm_layer=None,
            zero_init_residual=True,
            spiking_neuron=spiking_neuron,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            v_reset=v_reset,
            cnf=cnf,
        )
        backbone.conv1 = layer.Conv2d(
            input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        backbone.fc = nn.Sequential()
        self.num_channels = 512 if name in ("sew_resnet18", "sew_resnet34") else 2048

        if return_interm_layers:
            return_layers = {
                "layer1": "layer1",
                "layer2": "layer2",
                "layer3": "layer3",
                "layer4": "layer4",
            }
        else:
            return_layers = {"layer4": "layer4"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, x):
        xs = self.body(x)
        return xs


class Regressor(nn.Module):
    def __init__(self, channel=384, pose_dim=24 * 6):
        super(Regressor, self).__init__()
        # pose_dim=144 6D rotation matrix R=3x3
        self.decpose = nn.Linear(channel, pose_dim)
        self.dectrans = nn.Linear(channel, 3)
        self.decbeta = nn.Linear(channel, 10)

    def forward(self, x):
        # x: [B, T + 1, C]
        pose = self.decpose(x)  # [B, T + 1, pose_dim]
        tran = self.dectrans(x)  # [B, T + 1, 3]
        beta = self.decbeta(x)  # [B, T + 1, 10]
        return pose, tran, beta


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        n_layers=2,
        input_size=512,
        d_hidden=512 * 2,
        n_head=8,
        drop_prob=0.1,
        spiking_neuron="ParametricLIFNode",
        surrogate_function="ATan",
        cnf="ADD",
        detach_reset=True,
        v_reset=None,
    ):
        super(TransformerEncoder, self).__init__()
        from model_snn.spiking_transformer_spatiotemporal import TranformerEncoderLayer

        self.n_layers = n_layers
        self.input_size = input_size  # channel
        self.encoder = []
        for _ in range(self.n_layers):
            self.encoder.append(
                TranformerEncoderLayer(
                    d_model=input_size,
                    d_hidden=d_hidden,
                    n_head=n_head,
                    drop_prob=drop_prob,
                    spiking_neuron=spiking_neuron,
                    surrogate_function=surrogate_function,
                    cnf=cnf,
                    detach_reset=detach_reset,
                    v_reset=v_reset,
                )
            )
        self.encoder = nn.Sequential(*self.encoder)

    def forward(self, x):
        # x: spikes [T+1, B, C]
        assert x.shape[-1] == self.input_size

        for i in range(self.n_layers):
            x, score = self.encoder[i](x)
        return x, score


class SpikePoseNet(nn.Module):
    def __init__(
        self,
        num_frames=64,
        channel=4,
        model_name="sew_resnet34",
        return_interm_layers=True,
        use_tc=False,  # if true, reshape events [B, T, C, H, W] (t=T) to [B, T*C, 1, H, W] (t=T*C)
        spiking_neuron="ParametricLIFNode",
        surrogate_function="ATan",
        cnf="ADD",
        drop_prob=0.1,
        n_layers=1,
        detach_reset=True,
        v_reset=None,
        cam_intr=None,
        img_size=256,
        smpl_dir="../smpl_model/models/smpl/SMPL_MALE.pkl",
        batch_size=8,
        d_hidden=1024,
        pose_dim=24 * 6,
        use_rnn=False,
        use_recursive=False,
        use_transformer=False,
        n_head=1,
    ):
        super(SpikePoseNet, self).__init__()
        self.cam_intr = cam_intr
        self.num_frames = num_frames
        self.smpl = SMPL(smpl_dir, batch_size * (num_frames + 1))
        self.img_size = img_size

        # v_reset=None means soft_reset: substract v_threshold after spiking
        # v_reset=0.0 means hard_reset: reset to 0 after spiking
        self.backbone = Backbone(
            name=model_name,
            return_interm_layers=return_interm_layers,
            input_channel=1 if use_tc else channel,
            spiking_neuron=getattr(neuron, spiking_neuron),
            surrogate_function=getattr(surrogate, surrogate_function)(),
            detach_reset=detach_reset,
            v_reset=v_reset,
            cnf=cnf,
        )
        self.use_transformer = use_transformer
        if self.use_transformer:
            self.tranformer_encoder = TransformerEncoder(
                n_layers=n_layers,
                input_size=self.backbone.num_channels,
                d_hidden=d_hidden,
                n_head=n_head,
                drop_prob=drop_prob,
                spiking_neuron=spiking_neuron,
                surrogate_function=surrogate_function,
                cnf=cnf,
                detach_reset=detach_reset,
                v_reset=v_reset,
            )
        else:
            pass
        print("[warning] transformer={}".format(str(self.use_transformer)))

        self.avg_pool = nn.AdaptiveAvgPool3d((num_frames + 1, 1, 1))
        print("[warning] AdaptiveAvgPool3d")

        self.regressor = Regressor(
            channel=self.backbone.num_channels, pose_dim=pose_dim
        )

    def forward(self, events):
        B = events.shape[0]
        # events: [B, T, C, H, W] -> [T, B, C, H, W]
        input = events.permute(1, 0, 2, 3, 4)
        layer_spiking_rates = {"input": torch.mean((input != 0).float())}

        x = self.backbone(input)
        for name, feat in x.items():
            layer_spiking_rates[name] = torch.mean((feat.detach() != 0).float())

        x = x["layer4"]  # [T, B, C, H/32, W/32]

        if self.use_transformer:
            x = x.permute(0, 1, 3, 4, 2)  # [T, B, H, W, C]
            x, score = self.tranformer_encoder(x)  # spikes [T, B, H, W, C]
            layer_spiking_rates["transformer"] = torch.mean((x.detach() != 0).float())
            x = self.avg_pool(x.permute(1, 4, 0, 2, 3))  # [B, C, T+1, 1, 1]
            x = x.squeeze(-1).squeeze(-1).permute(2, 0, 1)  # [T+1, B, C]
        else:
            score = None
            x = self.avg_pool(x.permute(1, 2, 0, 3, 4))  # [B, C, T+1, 1, 1]
            x = x.squeeze(-1).squeeze(-1).permute(2, 0, 1)  # [T+1, B, C]

        x = x.permute(1, 0, 2)  # [B, T+1, C]
        N = x.shape[1]
        # pose [B, T+1, 24*6], trans [B, T+1, 3], beta [B, T+1, 10]
        pose, trans, beta = self.regressor(x)
        trans = trans.unsqueeze(dim=2)  # [B, T+1, 1, 3]
        pred_rotmats = rot6d_to_rotmat(pose).view(B, N, 24, 3, 3)

        verts, joints3d, _ = self.smpl(
            beta=beta.view(-1, 10),
            theta=None,
            get_skin=True,
            rotmats=pred_rotmats.view(-1, 24, 3, 3),
        )
        verts = verts.view(B, N, verts.size(1), verts.size(2)) + trans.detach()

        results = {}
        results["layer_spiking_rates"] = layer_spiking_rates
        results["attention_score"] = score
        results["beta"] = beta  # [B, N, 10]
        results["pred_rotmats"] = pred_rotmats  # [B, N, 24, 3, 3]
        results["trans"] = trans  # [B, N, 1, 3]
        results["verts"] = verts  # [B, N, 6890, 3]
        results["joints3d"] = (
            joints3d.view(B, N, joints3d.size(1), joints3d.size(2)) + trans.detach()
        )  # [B, N, 24, 3]
        results["faces"] = torch.from_numpy(self.smpl.faces).long()
        if self.cam_intr is not None:
            # [B, T, 24, 2]
            results["joints2d"] = projection_torch(
                results["joints3d"], self.cam_intr, self.img_size, self.img_size
            )
            results["cam_intr"] = self.cam_intr
        return results


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda:2" if use_cuda else "cpu")
    # base.check_backend_library("cupy")

    # device = torch.device("cpu")
    device = torch.device("cuda:0")
    base.check_backend_library("cupy")

    num_frames = 8
    batch_size = 1
    model = SpikePoseNet(
        num_frames=num_frames,
        channel=4,
        model_name="sew_resnet34",
        return_interm_layers=True,
        use_tc=False,  # if true, reshape events [B, T, C, H, W] (t=T) to [B, T*C, 1, H, W] (t=T*C)
        spiking_neuron="ParametricLIFNode",
        surrogate_function="ATan",
        cnf="ADD",
        drop_prob=0.1,
        n_layers=2,
        d_hidden=1024,
        detach_reset=True,  # detach backward on reset path of neuron
        v_reset=None,  # None means soft reset
        cam_intr=None,
        img_size=256,
        smpl_dir="../smpl_model/models/smpl/SMPL_MALE.pkl",
        batch_size=batch_size,
        pose_dim=24 * 6,
        use_rnn=False,
        use_recursive=False,
        use_transformer=True,
        n_head=1,
    )
    model = model.to(device=device)
    functional.reset_net(model)
    functional.set_step_mode(model, "m")
    functional.set_backend(model, "cupy", getattr(neuron, "ParametricLIFNode"))
    # print(model.backbone)
    print("set up model...")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("# of Parameters {}".format(n_parameters))

    model.train()
    _x = (torch.rand([batch_size, num_frames, 4, 256, 256]) > 0.7).to(
        device=device, dtype=torch.float32
    )
    output = model(_x)
    for k, v in output.items():
        try:
            print(k, v.size())
        except:
            print(k)
