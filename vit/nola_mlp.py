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

from NOLA import NOLA2 as NOLALayer
# from NOLA_fly import NOLA_FLY as NOLALayer


class NOLA_mlp(nn.Module):
    """In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    """

    def __init__(self, mlp_fc: nn.Module, nola_fc: nn.Module):
        super().__init__()
        self.mlp_fc = mlp_fc
        # self.dim = qkv.in_features
        self.nola_fc = nola_fc

    def forward(self, x, mode='train'):
        mlp_fc = self.mlp_fc(x)  # B,N,3*org_C
        new_fc = self.nola_fc(x)
        mlp_fc += new_fc
        return mlp_fc


class NOLAmlp_ViT_timm(nn.Module):
    def __init__(self, vit_model: timm_ViT, r: int, num_classes: int = 0, peft_layer=None,
                 ka: int = 1024, kb: int = 1024):
        super(NOLAmlp_ViT_timm, self).__init__()

        assert r > 0
        if peft_layer:
            self.peft_layer = peft_layer
        else:
            self.peft_layer = list(range(len(vit_model.blocks)))

        # let's freeze first
        for param in vit_model.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(vit_model.blocks):
            # pdb.set_trace()
            # If we only want few peft layer instead of all
            if t_layer_i not in self.peft_layer:
                continue
            mlp_fc1 = blk.mlp.fc1
            mlp_fc2 = blk.mlp.fc2
            self.fc1_in = mlp_fc1.in_features
            self.fc1_out = mlp_fc1.out_features
            self.fc2_in = mlp_fc2.in_features
            self.fc2_out = mlp_fc2.out_features

            nola_fc1 = NOLALayer(self.fc1_in, self.fc1_out, ka=ka, kb=kb, rank=r)
            nola_fc2 = NOLALayer(self.fc2_in, self.fc2_out, ka=ka, kb=kb, rank=r)
            blk.mlp.fc1 = NOLA_mlp(mlp_fc1, nola_fc1)
            blk.mlp.fc2 = NOLA_mlp(mlp_fc2, nola_fc2)

        # self.reset_parameters()
        self.peft_vit = vit_model
        if num_classes > 0:
            self.peft_vit.reset_classifier(num_classes=num_classes)

    def save_nola_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both nola (A, B, alpha, beta) and fc parameters.
        """

        assert filename.endswith(".safetensors")

        wa = {}
        alpha = {}
        beta = {}
        for i, blk in enumerate(self.peft_vit.blocks):
            # Weight matrices (W_A, W_B) don't need to be saved since they are just random
            # matrices, can be easily re-generated using same seed. Only coefficients
            # alpha and beta are saved. Save first 10 values of W_A to verify that the
            # re-generated random weight matrices during testing are the same as ones used
            # in training.

            # Save first 10 params of w_a
            wa[f"fc1_A_{i:03d}"] = blk.mlp.fc1.nola_fc.A.detach().data[0, :10, 0]

            # Save fc1 params
            alpha[f"fc1_alpha_{i:03d}"] = blk.mlp.fc1.nola_fc.nola_alpha.detach().data
            beta[f"fc1_beta_{i:03d}"] = blk.mlp.fc1.nola_fc.nola_beta.detach().data

            # Save fc2 params
            alpha[f"fc2_alpha_{i:03d}"] = blk.mlp.fc2.nola_fc.nola_alpha.detach().data
            beta[f"fc2_beta_{i:03d}"] = blk.mlp.fc2.nola_fc.nola_beta.detach().data

        _in = self.peft_vit.head.in_features
        _out = self.peft_vit.head.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.peft_vit.head.weight}

        merged_dict = {**alpha, **beta, **fc_tensors, **wa}
        save_file(merged_dict, filename)

    def load_nola_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load both peft and fc parameters.
        """

        assert filename.endswith(".safetensors")

        # Weight matrices (W_A, W_B) are not loaded since they are just random
        # matrices, can be easily re-generated using same seed. Only coefficients
        # alpha and beta are loaded. Load first 10 values of W_A to verify that the
        # re-generated random weight matrices during testing are the same as ones used
        # in training.
        with safe_open(filename, framework="pt") as f:
            for i, blk in enumerate(self.peft_vit.blocks):
                # Verify W_A and W_B matrices are correct
                saved_key = f"fc1_A_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                assert (blk.mlp.fc1.nola_fc.A[0, :10, 0] == saved_tensor.cuda()).all()

                # Load alpha and beta coefficients
                saved_key = f"fc1_alpha_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                blk.mlp.fc1.nola_fc.nola_alpha = Parameter(saved_tensor)

                saved_key = f"fc1_beta_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                blk.mlp.fc1.nola_fc.nola_beta = Parameter(saved_tensor)

                saved_key = f"fc2_alpha_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                blk.mlp.fc2.nola_fc.nola_alpha = Parameter(saved_tensor)

                saved_key = f"fc2_beta_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                blk.mlp.fc2.nola_fc.nola_beta = Parameter(saved_tensor)

            # Load classifier fc params
            _in = self.peft_vit.head.in_features
            _out = self.peft_vit.head.out_features
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.peft_vit.head.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def forward(self, x: Tensor) -> Tensor:
        return self.peft_vit(x)


if __name__ == "__main__":  # Debug
    img = torch.randn(2, 3, 224, 224)
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    peft_vit = LoRA_ViT_timm(vit_model=model, r=4, num_classes=10)
    pred = peft_vit(img)
    print(pred.shape)

    img = torch.randn(2*20, 3, 224, 224)
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    peft_vit = LoRA_ViT_timm(vit_model=model, r=4, num_classes=10)
    pred = peft_vit.forward3D(img)
    print(pred.shape)
