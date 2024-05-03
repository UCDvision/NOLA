# Modified version of code from LoRA-ViT

import math
import pdb

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from safetensors import safe_open
from safetensors.torch import save_file
from timm.models.vision_transformer import VisionTransformer as timm_ViT
from torch import Tensor
from torch.nn.parameter import Parameter

from base_vit import ViT

from utils.utils import normalize_int, unnormalize_int


class LoRA_qkv(nn.Module):
    """In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    """

    def __init__(
            self,
            qkv: nn.Module,
            lora_q: nn.Module,
            lora_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.lora_q = lora_q
        self.lora_v = lora_v

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.lora_q(x)
        new_v = self.lora_v(x)
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv


class LoRALayer(nn.Module):
    """Low-rank matrix multiplication of LoRA
    """
    def __init__(self, rank, dim_in, dim_out, bias=False, alpha=32.):
        super(LoRALayer, self).__init__()
        self.r = rank
        self.d = dim_out
        self.k = dim_in
        self.lora_B = nn.Parameter(torch.zeros((dim_out, rank)))
        self.lora_A = nn.Parameter(torch.zeros((rank, dim_in)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.bias = bias
        self.alpha = alpha
        if bias:
            self.lora_b = nn.Parameter(torch.zeros((dim_out,)))
        self.__init_weights__()

    def __init_weights__(self):
        # Both A and B are initialized to normal here unlike zeros for A in LoRA
        nn.init.zeros_(self.lora_B)
        # nn.init.normal_(self.lora_B)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        if self.bias:
            nn.init.zeros_(self.lora_b)

    def forward(self, x):
        x = (x @ self.lora_A.T) @ self.lora_B.T
        return x


class LoRA_ViT_timm(nn.Module):
    def __init__(self, vit_model: timm_ViT, r: int, num_classes: int = 0, lora_layer=None):
        super(LoRA_ViT_timm, self).__init__()

        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.blocks)))

        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []
        self.w_Qs = []  # These are lora layers
        self.w_Vs = []

        # lets freeze first
        for param in vit_model.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(vit_model.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features

            # Modify code to use our LoRA layer
            lora_q = LoRALayer(r, self.dim, self.dim)
            lora_v = LoRALayer(r, self.dim, self.dim)
            blk.attn.qkv = LoRA_qkv(w_qkv_linear, lora_q, lora_v)

        self.reset_parameters()
        self.lora_vit = vit_model
        if num_classes > 0:
            self.lora_vit.reset_classifier(num_classes=num_classes)

    def save_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """
        assert filename.endswith(".safetensors")
        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.head.weight}
        save_file(fc_tensors, filename)

    def load_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")
        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        with safe_open(filename, framework="pt") as f:
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.head.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both nola (A, B, alpha, beta) and fc parameters.
        """

        assert filename.endswith(".safetensors")

        wa = {}
        wb = {}
        for i, blk in enumerate(self.lora_vit.blocks):
            wa[f"q_A_{i:03d}"] = blk.attn.qkv.lora_q.lora_A.data
            wb[f"q_B_{i:03d}"] = blk.attn.qkv.lora_q.lora_B.data

            wa[f"v_A_{i:03d}"] = blk.attn.qkv.lora_v.lora_A.data
            wb[f"v_B_{i:03d}"] = blk.attn.qkv.lora_v.lora_B.data

        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.head.weight}

        # merged_dict = {**a_tensors, **b_tensors, **fc_tensors}
        merged_dict = {**wa, **wb, **fc_tensors}
        save_file(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load both lora and fc parameters.
        """

        assert filename.endswith(".safetensors")

        with safe_open(filename, framework="pt") as f:
            for i, blk in enumerate(self.lora_vit.blocks):
                saved_key = f"q_A_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                blk.attn.qkv.lora_q.lora_A = Parameter(saved_tensor)
                saved_key = f"q_B_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                blk.attn.qkv.lora_q.lora_B = Parameter(saved_tensor)

                saved_key = f"v_A_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                blk.attn.qkv.lora_v.lora_A = Parameter(saved_tensor)
                saved_key = f"v_B_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                blk.attn.qkv.lora_v.lora_B = Parameter(saved_tensor)

            _in = self.lora_vit.head.in_features
            _out = self.lora_vit.head.out_features
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.head.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)
        for w_Q in self.w_Qs:
            nn.init.kaiming_uniform_(w_Q.lora_B.weight, a=math.sqrt(5))
            nn.init.zeros_(w_Q.lora_A.weight)
        for w_V in self.w_Vs:
            nn.init.kaiming_uniform_(w_V.lora_B.weight, a=math.sqrt(5))
            nn.init.zeros_(w_V.lora_A.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.lora_vit(x)


if __name__ == "__main__":  # Debug
    img = torch.randn(2, 3, 224, 224)
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    lora_vit = LoRA_ViT_timm(vit_model=model, r=4, num_classes=10)
    pred = lora_vit(img)
    print(pred.shape)

    img = torch.randn(2*20, 3, 224, 224)
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    lora_vit = LoRA_ViT_timm(vit_model=model, r=4, num_classes=10)
    pred = lora_vit.forward3D(img)
    print(pred.shape)
