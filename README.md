# NOLA: Compressing LoRA using Linear Combination of Random Basis

Paper: NOLA: Compressing LoRA using Linear Combination of Random Basis (ICLR 2024)

Soroush Abbasi Koohpayegani*, Navaneet K L*, Parsa Nooralinejad, Soheil Kolouri, Hamed Pirsiavash

This Repository is an official implementation of NOLA: 
[arXiv](https://arxiv.org/abs/2310.02556), [ICLR 2024](https://openreview.net/forum?id=TjfXcDgvzk)

Our code is based on [LoRA](https://github.com/microsoft/LoRA/tree/main) and [QLoRA](https://github.com/artidoro/qlora). 


## Overview


NOLA is a novel approach for fine-tuning large models such as LLMs and Vision Transformers. Similar to LoRA, NOLA uses a low-rank decomposition of weight matrices for the fine-tuning step. However, LoRA face two primary limitations: 
- The parameter count is lower-bounded by the rank one decomposition
- The extent of reduction is heavily influenced by both the model architecture and the chosen rank.

We introduce NOLA, which brings parameter count felexiblity to LoRA. NOLA achieves this by re-parameterizing the low-rank matrices in LoRA using linear combinations of randomly generated matrices (basis) and optimizing the linear coefficients only. This approach allows us to decouple the number of trainable parameters from both the choice of rank and the network architecture.

For example, in LLaMA-2 70B, NOLA is almost 20 times more compact than the most compressed LoRA without degradation in accuracy. Remarkbly, We are able to finetune LLaMA-2 70B with only 0.6M parameters only. 



<div align="center">
  <a href="https://ucdvision.github.io/NOLA/">
    <img src="https://ucdvision.github.io/NOLA/assets/images/NOLA_gif.gif">
  </a>
</div>


## Simple Code of NOLA

Please use `GenerateParams` to generate parameters on thy fly with linear combinations of randomly generated matrices (basis). 

It's important to note that the random basis remains fixed during optimization, and we only need to compute gradients with respect to coefficients. As the random basis is static, we can discard it after computing the output of the forward function. This technique significantly reduces the GPU's memory footprint since there's no need to retain the basis. For the backward pass, we must again use the same seed as in the forward pass to compute the basis and determine the gradient for the coefficients. Although there is a slight overhead in calculating the dot product between the basis and coefficients, this overhead is negligible when the number and dimensions of the basis are small. For a more detailed analysis, please refer to our paper.

```python
class GenerateParams(torch.autograd.Function):
    # Generate parameters on the fly with random basis
    
    @staticmethod    
    def forward(ctx, coefficients, out_dim, in_dim, seed): 
        num_basis = coefficients.shape[0]
        Out = torch.zeros(out_dim, in_dim).to(coefficients.device)
        rand_seed = torch.randint(int(1e10), (1,))
        torch.manual_seed(seed)
        
        W = torch.zeros(num_basis, out_dim, in_dim, 
                        device=coefficients.device, dtype=coefficients.dtype)
        nn.init.uniform_(W, a=-1.0, b=1.0)
        Out = torch.einsum('b,boi->oi', coefficients, W)
        
        params = torch.autograd.Variable(torch.tensor([out_dim, in_dim, seed]))
        ctx.save_for_backward(coefficients, params)
        torch.manual_seed(rand_seed)
        return Out 
    
    @staticmethod
    def backward(ctx, grad_output):
        coefficients, params = ctx.saved_tensors
        num_basis = coefficients.shape[0]

        out_dim, in_dim, seed = params
        rand_seed = torch.randint(int(1e10), (1,))
        torch.manual_seed(seed)
        grad_coefficients = torch.empty(0).to(grad_output.device)

        W = torch.zeros(num_basis, out_dim, in_dim, 
                        device=coefficients.device, dtype=coefficients.dtype)
        nn.init.uniform_(W, a=-1.0, b=1.0) 
        W = W.permute(1,2,0).reshape(-1, num_basis)
        grad_coefficients = torch.einsum('d,dl->l', grad_output.flatten(), W)

        torch.manual_seed(rand_seed)    
        return grad_coefficients , None, None, None
```

Below example shows how to use this re-parametrization trick on a Linear layer and LoRA: 

```python
class NOLALinear(nn.Linear): 
    # NOLA Linear Layer 
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        num_basis = 256, 
        **kwargs):
        
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.num_basis = num_basis 
        self.generate_params = GenerateParams.apply
        self.seed = nn.Parameter(torch.randint(int(1e10), (1,)), requires_grad=False)
        self.coefficients = nn.Parameter(torch.zeros(num_basis), requires_grad=True)
        self.weight.requires_grad = False
        
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, num_basis={}'.format(
            self.in_features, self.out_features, self.num_basis)  

    def forward(self, x: torch.Tensor):
        W = self.generate_params(self.coefficients,
                          self.out_features,
                          self.in_features,
                          self.seed) + self.weight
        return x @ W.t()

 
class LoRA(nn.Module):
    # LoRA with NOLA, implementation of a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        use_nola = False,
        num_basis = 1024, 
        rank=16, 
        alpha = 1.0, 
        **kwargs
    ):
        super(LoRA, self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.num_basis_A = num_basis
        self.num_basis_B = num_basis
        self.rank = rank
        self.alpha = alpha 
        self.scale = self.alpha / self.rank
        self.generate_params = GenerateParams.apply
        self.use_nola = use_nola 
        
        if use_nola: 
            self.lora_A = NOLALinear(in_features, rank, num_basis=self.num_basis_A, bias=False)
            self.lora_B = NOLALinear(rank, out_features, num_basis=self.num_basis_B, bias=False)
        else:
            self.lora_A = nn.Linear(in_features, rank, bias=False)
            self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.reset_lora_parameters()
        
    def extra_repr(self) -> str:
        return 'NOLA: rank={}, in_features={}, out_features={}, num_basis_A={}, num_basis_B={}'.format(
            self.rank, self.in_features, self.out_features, self.num_basis_A, self.num_basis_B)

    def reset_lora_parameters(self):        
        if self.use_nola: 
            nn.init.zeros_(self.lora_A.coefficients)
            nn.init.zeros_(self.lora_B.coefficients)
            
        # initialize A the same way as the default for nn.Linear and B to zero
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor):
        return self.lora_B(self.lora_A(x)) * self.scale

```

## Requirements

All our experiments use the PyTorch library. Instructions for PyTorch installation can be found [here](https://pytorch.org/). We use GPT-2 and LLaMA 2 for our experiments on natural language generation tasks.

## Dataset

We use E2ENLG, DART and WebNLG datasets for our experiments on natural language generation with GPT-2. Moreover, we use Alpaca dataset for instruction fine-tuning with LLaMA-2.  

## Getting Started -- Large Language Model Finetuning (LLaMA-2)

We modified [QLoRA](https://github.com/artidoro/qlora) code and added NOLA to LoRA model. Please install dependencies in a virtual environment: 

```
 cd llama
 pip install -U -r requirements.txt
```

Please download LLaMA models from Hugging Face. For example, you can access the LLaMA-2 13B model from [here](https://huggingface.co/meta-llama/Llama-2-13b-hf). Alternatively, you can also use LLaMA-3. Please note that we have not conducted experiments with LLaMA-3 in our paper.

Next, please use the corresponding scripts to fine-tune LLaMA models with NOLA. Please change the `--model_name_or_path` to the path for LLaMA model. For instance, to fine-tune LLaMA-2 13B on the Alpaca dataset, execute the following script. 

```
bash scripts/finetune_llama2_guanaco_13b_nola.sh
```




## Getting Started -- Large Language Model Finetuning (GPT-2)

We modified [LoRA](https://github.com/microsoft/LoRA/tree/main) code and added NOLA to LoRA model. Please install dependencies in a virtual environment: 
 
 ```
 pip install -r requirement.txt
 bash download_pretrained_checkpoints.sh
 bash create_datasets.sh
 cd ./eval
 bash download_evalscript.sh
 cd ..
 ```

Next, please use the corresponding scripts to fine-tune LLaMA models with NOLA. For instance, to fine-tune GPT-2 Large on the E2E dataset, execute the following script.

```
bash scripts/finetune_gpt2l_e2e_qv_nola.sh
```






## Citation

If you find this project useful in your research, please consider cite:
```
@inproceedings{
koohpayegani2024nola,
title={{NOLA}: Compressing LoRA using Linear Combination of Random Basis},
author={Soroush Abbasi Koohpayegani and Navaneet K L and Parsa Nooralinejad and Soheil Kolouri and Hamed Pirsiavash},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=TjfXcDgvzk}
}
```

## License

This project is under the MIT license.
