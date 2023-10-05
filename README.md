# NOLA: Networks as Linear Combination of Low Rank Random Basis

This Repository is an official implementation of NOLA.
Our code is based on [LoRA](https://github.com/microsoft/LoRA/tree/main). 

## Overview

NOLA is a novel approach for fine-tuning large models such as LLMs and Vision Transformers. Similar to LoRA, NOLA uses a low-rank decomposition of weight matrices for the fine-tuning step. However, instead of optmizing these matrices, we use a collection of such matrices with random initialization and learn just the mixture coefficients on the target task. This decouples the number of training parameters and the size of the weight matrices and provides a more fine-grained control on the number of training parameters. While LoRA is limited to rank one decomposition of the matrices to limit the training parameters, NOLA has no such limitations. Through experiments on both language and vision tasks, we show that NOLA outperforms LoRA at comparable parameters and achieves comparable performance with just half or one-third the parameters. The random basis matrices can be generated on the fly and can be shared across layers and tasks and thus provides training and inference memory efficiency. It can also be quantized better than LoRA and achieves comparable performance to full precision with 2-bit quantization of the NOLA parameters.

![](nola_teaser_2-1.png)

## Requirements

All our experiments use the PyTorch library. Instructions for PyTorch installation can be found [here](https://pytorch.org/). We primarily use GPT-2 for our experiments on natural language generation tasks.

## Dataset

We use E2ENLG, DART and WebNLG datasets for our experiments on natural language generation. 

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
