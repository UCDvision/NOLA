

# NOLA -- Large Language Model Finetuning (LLaMA)

Our code is based on [QLoRA](https://github.com/artidoro/qlora). 

We modified [QLoRA](https://github.com/artidoro/qlora) code and added NOLA to LoRA model. Please install dependencies in a virtual environment: 

```
 cd llama
 pip install -U -r requirements.txt
```

Next, please use the corresponding scripts to fine-tune LLaMA models with NOLA. For instance, to fine-tune LLaMA-2 70B, execute the following script.

```
bash scripts/finetune_llama2_guanaco_70b_nola.sh
```

