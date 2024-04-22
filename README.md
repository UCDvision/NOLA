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

We introduce NOLA, which brings parameter count felexiblity to LoRA. NOLA achieves this by re-parameterizing the low-rank matrices in LoRA using linear combinations of randomly generated matrices (basis) and optimizing the linear mixture coefficients only. This approach allows us to decouple the number of trainable parameters from both the choice of rank and the network architecture.

For example, in LLaMA-2 70B, NOLA is almost 20 times more compact than the most compressed LoRA without degradation in accuracy. Remarkbly, We are able to finetune LLaMA-2 70B with only 0.6M parameters only. 

![Nola_teaser7](/NOLA/docs/assets/images/NOLA_gif.gif)

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

All our experiments use the PyTorch library. Instructions for PyTorch installation can be found [here](https://pytorch.org/). We primarily use GPT-2 and LLaMA 2 for our experiments on natural language generation tasks.

## Dataset

We use E2ENLG, DART and WebNLG datasets for our experiments on natural language generation with GPT-2. Moreover, we use Alpaca dataset for instruction fine-tuning with LLaMA-2.  

## Getting Started -- Large Language Model Finetuning (LLaMA)

We modify [QLoRA](https://github.com/artidoro/qlora) code and add NOLA to LoRA model. 


## Getting Started -- Large Language Model Finetuning (GPT-2)

Please install dependencies in a virtual environment: 
 
 ```
 pip install -r requirement.txt
 bash download_pretrained_checkpoints.sh
 bash create_datasets.sh
 cd ./eval
 bash download_evalscript.sh
 cd ..
 ```

## Finetuning with NOLA (float)

1. Finetune GPT-2 with NOLA: 
```
python -m torch.distributed.launch --nproc_per_node=1 src/gpt2_ft.py \
    --train_data ./data/e2e/train.jsonl \
    --valid_data ./data/e2e/valid.jsonl \
    --train_batch_size 8 \
    --grad_acc 1 \
    --valid_batch_size 4 \
    --seq_len 512 \
    --model_card gpt2.md \
    --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin \
    --platform local \
    --clip 0.0 \
    --lr 0.1 \
    --weight_decay 0.0 \
    --correct_bias \
    --adam_beta2 0.999 \
    --scheduler linear \
    --warmup_step 500 \
    --max_epoch 5 \
    --save_interval 1000 \
    --nola_rank 8 \
    --nola_c 1.0 \
    --nola_qv \
    --label_smooth 0.1 \
    --work_dir ./trained_models/GPT2_M/e2e \
    --random_seed 110
```

2. Use beam search to generate outputs from the finetuned model:
```
python -m torch.distributed.launch --nproc_per_node=1 src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M/e2e/model.26290.pt \
    --platform local \
    --nola_rank 8 \
    --nola_c 1.0 \
    --nola_qv \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M/e2e \
    --output_file predict.26290.b10p08r4.jsonl
```

3. Decode outputs:
```
python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./trained_models/GPT2_M/e2e/predict.26290.b10p08r4.jsonl \
    --input_file ./data/e2e/test_formatted.jsonl \
    --output_ref_file e2e_ref.txt \
    --output_pred_file e2e_pred.txt
```

4. Run evaluation on E2E test set

```
python eval/e2e/measure_scores.py e2e_ref.txt e2e_pred.txt -p
```

# Finetuning with NOLA (Quantized)

1. Finetune GPT-2 with NOLA (add `--qnola` flag and number of bits per parameter `--qbits 3`): 
```
python -m torch.distributed.launch --nproc_per_node=1 src/gpt2_ft.py \
    --train_data ./data/e2e/train.jsonl \
    --valid_data ./data/e2e/valid.jsonl \
    --train_batch_size 8 \
    --grad_acc 1 \
    --valid_batch_size 4 \
    --seq_len 512 \
    --model_card gpt2.md \
    --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin \
    --platform local \
    --clip 0.0 \
    --lr 0.1 \
    --weight_decay 0.0 \
    --correct_bias \
    --adam_beta2 0.999 \
    --scheduler linear \
    --warmup_step 500 \
    --max_epoch 5 \
    --save_interval 1000 \
    --nola_rank 8 \
    --nola_c 0.01 \
    --qnola \
    --qbits 3 \
    --nola_qv \
    --label_smooth 0.1 \
    --work_dir ./trained_models/GPT2_M/e2e \
    --random_seed 110
```

2. Use beam search to generate outputs from the finetuned model:
```
python -m torch.distributed.launch --nproc_per_node=1 src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M/e2e/model.26290.pt \
    --platform local \
    --nola_rank 8 \
    --nola_c 0.01 \
    --nola_qv \
    --qnola \
    --qbits 3 \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M/e2e \
    --output_file predict.26290.b10p08r4.jsonl
```

3. Decode outputs:
```
python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./trained_models/GPT2_M/e2e/predict.26290.b10p08r4.jsonl \
    --input_file ./data/e2e/test_formatted.jsonl \
    --output_ref_file e2e_ref.txt \
    --output_pred_file e2e_pred.txt
```

4. Run evaluation on E2E test set

```
python eval/e2e/measure_scores.py e2e_ref.txt e2e_pred.txt -p
```




## Citation

If you make use of the code, please cite the following work:
```
@inproceedings{koohpayegani2023nola,
 author = { Koohpayegani, Soroush Abbasi and Navaneet, K L and Nooralinejad, Parsa and Kolouri, Soheil and Pirsiavash, Hamed},
 title = {NOLA: Networks as Linear Combination of Low Rank Random Basis},
 year = {2023}
}
```

## License

This project is under the MIT license.
