import torch
import torch.nn as nn
import math
from spikingjelly.activation_based import (
    surrogate,
    neuron,
    functional,
    base,
    layer,
)
from collections import OrderedDict


class HammingDistanceAttention(nn.Module):
    def __init__(self):
        super(HammingDistanceAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        _, _, _, d = k.size()

        # 1. dot product Query with Key^T to compute similarity
        score = 0.5 * (1 + torch.matmul(2 * q - 1, 2 * k.transpose(2, 3) - 1) / d)

        # 2. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 3. multiply with Value
        v = torch.matmul(score, v)

        return v, score


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model,
        d_hidden,
        n_head,
        spiking_neuron,
        surrogate_function,
        detach_reset,
        v_reset,
    ):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head

        self.linear_q = layer.Linear(d_model, d_hidden, bias=False)
        self.bn_q = layer.BatchNorm2d(d_hidden)
        self.sn_q = getattr(neuron, spiking_neuron)(
            surrogate_function=getattr(surrogate, surrogate_function)(),
            detach_reset=detach_reset,
            v_reset=v_reset,
        )

        self.linear_k = layer.Linear(d_model, d_hidden, bias=False)
        self.bn_k = layer.BatchNorm2d(d_hidden)
        self.sn_k = getattr(neuron, spiking_neuron)(
            surrogate_function=getattr(surrogate, surrogate_function)(),
            detach_reset=detach_reset,
            v_reset=v_reset,
        )

        self.linear_v = layer.Linear(d_model, d_hidden)
        self.bn_v = layer.BatchNorm2d(d_hidden)
        self.sn_v = getattr(neuron, spiking_neuron)(
            surrogate_function=getattr(surrogate, surrogate_function)(),
            detach_reset=detach_reset,
            v_reset=v_reset,
        )

        self.linear_out = layer.Linear(d_hidden, d_model)
        self.bn_out = layer.BatchNorm2d(d_model)
        self.sn_out = getattr(neuron, spiking_neuron)(
            surrogate_function=getattr(surrogate, surrogate_function)(),
            detach_reset=detach_reset,
            v_reset=v_reset,
        )

        # self.attention = ScaleDotProductAttention()
        self.attention = HammingDistanceAttention()
        # self.attention = CosineSimilarityAttention()

        for m in self.modules():
            if isinstance(m, layer.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (layer.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, q, k, v, postional_encoding):
        # q, k, v: [t, b, h, w, c]

        # 1. dot product with weight matrices
        # [t, b, h, w, c]

        q = self.linear_q(q) + postional_encoding
        # [T, B, H, W, C] -> [T, B, C, H, W]
        q = self.bn_q(q.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        q = self.sn_q(q)

        k = self.linear_k(k) + postional_encoding
        # [T, B, H, W, C] -> [T, B, C, H, W]
        k = self.bn_k(k.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        k = self.sn_k(k)

        v = self.linear_v(v) + postional_encoding

        # 2. split tensor by number of heads
        # [b, head, t, h, w, c_v] d_tensor = d_model // n_head
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        # [b, head, t, h, w, c_v] -> [b, head, thw, c_v]
        b, n_head, t, h, w, c_v = v.shape
        v, score = self.attention(q.flatten(2, 4), k.flatten(2, 4), v.flatten(2, 4))
        # [b, head, thw, c_v] -> [b, head, t, h, w, c_v]
        v = v.reshape(b, n_head, t, h, w, c_v)

        # 4. concat and pass to linear layer
        # [t, b, h, w, c]
        v = self.concat(v)
        # [t, b, h, w, c] -> [t, b, c, h, w]
        v = self.bn_v(v.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        v = self.sn_v(v)

        out = self.linear_out(v)
        # [t, b, h, w, c] -> [t, b, c, h, w]
        out = self.bn_out(out.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        out = self.sn_out(out)

        # 5. visualize attention map need to return attention score as well
        return out, score

    def split(self, tensor):
        """
        split tensor by number of head
        :param tensor: [t, b, h, w, c]
        :return: [b, head, t, h, w, c_v]
        """
        t, b, h, w, c = tensor.shape

        c_v = c // self.n_head
        tensor = tensor.reshape(t, b, h, w, self.n_head, c_v)
        # [t, b, h, w, head, c_v] -> [b, head, t, h, w, c_v]
        tensor = tensor.permute(1, 4, 0, 2, 3, 5)
        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)
        :param tensor: [b, head, t, h, w, c_v]
        :return: [t, b, h, w, c]
        """
        b, head, t, h, w, c_v = tensor.shape
        c = head * c_v
        # [b, head, t, h, w, c_v] -> [t, b, h, w, head, c_v]
        tensor = tensor.permute(2, 0, 3, 4, 1, 5).reshape(t, b, h, w, c)
        return tensor


class FeedForwardNet(nn.Module):
    def __init__(
        self,
        d_model,
        ffn_hidden,
        spiking_neuron="ParametricLIFNode",
        surrogate_function="ATan",
        detach_reset=True,
        v_reset=None,
        drop_prob=0.1,
    ):
        super(FeedForwardNet, self).__init__()
        self.dropout1 = layer.Dropout(p=drop_prob)
        self.linear1 = layer.Linear(d_model, ffn_hidden, bias=False)
        self.bn1 = layer.BatchNorm2d(ffn_hidden)
        self.sn1 = getattr(neuron, spiking_neuron)(
            surrogate_function=getattr(surrogate, surrogate_function)(),
            detach_reset=detach_reset,
            v_reset=v_reset,
        )

        self.dropout2 = layer.Dropout(p=drop_prob)
        self.linear2 = layer.Linear(ffn_hidden, d_model, bias=False)
        self.bn2 = layer.BatchNorm2d(d_model)
        self.sn2 = getattr(neuron, spiking_neuron)(
            surrogate_function=getattr(surrogate, surrogate_function)(),
            detach_reset=detach_reset,
            v_reset=v_reset,
        )

        for m in self.modules():
            if isinstance(m, layer.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (layer.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: [T, B, H, W, C]
        x = self.dropout1(x)
        x = self.linear1(x)
        # [T, B, H, W, C] -> [T, B, C, H, W]
        x = self.bn1(x.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        x = self.sn1(x)

        x = self.dropout2(x)
        x = self.linear2(x)
        # [T, B, H, W, C] -> [T, B, C, H, W]
        x = self.bn2(x.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        # print("zero initilization", torch.sum((x != 0).float()))
        x = self.sn2(x)
        return x


class TranformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_hidden,
        n_head,
        drop_prob=0.1,
        spiking_neuron="ParametricLIFNode",
        surrogate_function="ATan",
        cnf="ADD",
        detach_reset=True,
        v_reset=None,
    ):
        super(TranformerEncoderLayer, self).__init__()
        self.cnf = cnf
        self.positional_encoding = PositionEmbeddingSine(channel=d_hidden)
        self.attention = MultiHeadAttention(
            d_model,
            d_hidden,
            n_head,
            spiking_neuron,
            surrogate_function,
            detach_reset,
            v_reset,
        )

        self.ffn = FeedForwardNet(
            d_model,
            d_hidden,
            spiking_neuron,
            surrogate_function,
            detach_reset,
            v_reset,
            drop_prob,
        )

        # zero initilization of the last layer before residual layer
        for m in self.modules():
            if isinstance(m, MultiHeadAttention):
                # nn.init.constant_(m.w_concat.linear.weight, 0)
                nn.init.constant_(m.bn_out.weight, 0)
            if isinstance(m, FeedForwardNet):
                # nn.init.constant_(m.ffn.linear2.weight, 0)
                nn.init.constant_(m.bn2.weight, 0)

    @staticmethod
    def sew_function(x: torch.Tensor, y: torch.Tensor, cnf: str):
        if cnf == "ADD":
            return x + y
        elif cnf == "AND":
            return x * y
        elif cnf == "IAND":
            return x * (1.0 - y)
        else:
            raise NotImplementedError

    def forward(self, x):
        # 1. compute self attention, [t, b, h, w, c]
        _x = x
        pos_enc = self.positional_encoding(x)
        x, score = self.attention(q=x, k=x, v=x, postional_encoding=pos_enc)

        # 2. residual x and _x spikes [t, b, h, w, c]
        x = self.sew_function(x, _x, self.cnf)

        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 4. residual x and _x spikes [t, b, h, w, c]
        x = self.sew_function(x, _x, self.cnf)
        return x, score


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, channel, normalize=True, scale=2 * math.pi):
        super().__init__()
        self.normalize = normalize
        self.scale = scale
        self.channel = channel

    def forward(self, x):
        T, B, H, W, _ = x.shape
        C = self.channel
        device = x.device

        # [T, H, W]
        x_embed = torch.arange(W, device=device)[None, None, :].repeat(T, H, 1)
        y_embed = torch.arange(H, device=device)[None, :, None].repeat(T, 1, W)
        # z_embed = torch.arange(T)[:, None, None].repeat(1, H, W)
        if self.normalize:
            x_embed = x_embed / x_embed[:, :, -1:] * self.scale
            y_embed = y_embed / y_embed[:, -1:, :] * self.scale
            # z_embed = z_embed / z_embed[-1:, :, :] * self.scale

        # [C / 2]
        dim_t = torch.arange(0, C, step=2, device=device).float()
        dim_t = 10000 ** (dim_t / C)

        # [T, H, W, C / 2]
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t

        # [T, H, W, 2, C/4] -> [T, H, W, C/2]
        pos_x = torch.stack(
            [pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=3
        ).flatten(3)
        pos_y = torch.stack(
            [pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=3
        ).flatten(3)

        # [T, 1, H, W, C]
        pos = torch.cat([pos_y, pos_x], dim=-1).unsqueeze(1) / T
        return pos


if __name__ == "__main__":
    device = torch.device("cpu")

    B, T, H, W, C = 1, 16, 8, 8, 128
    x = (torch.rand(T, B, H, W, C) > 0.8).float()
    model = TranformerEncoderLayer(
        d_model=C,
        d_hidden=C * 2,
        n_head=2,
        drop_prob=0.1,
        spiking_neuron="ParametricLIFNode",
        surrogate_function="ATan",
        cnf="ADD",
        detach_reset=False,
        v_reset=None,
    )
    functional.reset_net(model)
    functional.set_step_mode(model, "m")
    print("input firing rate", torch.mean((x > 0).float()))
    out = model(x)  # [T, B, H, W, C]
    print(out.shape)
    print("out firing rate", torch.mean((out > 0).float()))
    print(
        "backbone output (check spikes)",
        torch.sum((out != 0) & (out != 1) & (out != 2)),
    )

    # for n, p in model.named_parameters():
    #     # print(n, p.shape, p.requires_grad)
    #     if "norm" in n:
    #         print(n, p)

    # for m in model.modules():
    #     print(m)

    # pos = PositionEmbeddingSine(size=[8, 8, 8, 512], device=device)
    # print(pos.positional_encoding.shape)
