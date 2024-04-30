

# NOLA: Large Language Model Finetuning (LLaMA)

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

