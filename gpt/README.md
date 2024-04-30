# NOLA: Large Language Model Finetuning (GPT-2)
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

