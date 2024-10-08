# IEEE BigData 2024 Cup: Detection of suicide risk on social media

A Pytorch implementation for our paper [Leveraging Large Language Models for Suicide Detection on Social Media with Limited Labels](https://arxiv.org/pdf/2410.04501).


If you find our work useful, please consider citing:
```
@article{nguyen2024leveraging,
title={Leveraging Large Language Models for Suicide Detection on Social Media with Limited Labels},
author={Vy Nguyen and Chau Pham},
journal={arXiv preprint arXiv:2410.04501},
year={2024}}
```

Given a user post (i.e., text) *T*, our goal is to train a model that takes *T* as input to classify the suicide risk levels. Our method involves using Large Language Models (LLMs) with few-shot Chain-of-Thought prompting, and classification fine-tuning, as shown in figure below.


![alt text](https://raw.githubusercontent.com/khanhvynguyen/Suicide_Detection_LLMs/main/assets/overview.png?token=GHSAT0AAAAAACQNCCFPLFBRYPQDKELWAGYSZQLNYNQ)


Our solution achieved the highest score on overall evaluation at [IEEE BigData 2024 Cup: Detection of suicide risk on social media Competition](https://competitionpolyu.github.io/), based on *model performance* (6th on the private board with an F1 score of 0.73), *approach innovation*, and *report quality*. 




## 1. Set up 
### 1.1. Dataset

In this competition, there are 500 labeled posts and 1500 unlabeled posts in the dataset provided by the organizers. We can't share the dataset here since it requires completion of the necessary data usage agreement. To request access to the dataset, please use [this link](https://github.com/AlexLee01/Suicide-Triggers-and-Risk-Dataset) to contact the organizers. For more details on the dataset, refer to [this paper](https://www.hkie.org.hk/hkietransactions/upload/2022-12-09/THIE-2022-0031.pdf).

 After obtaining the dataset, store it in `data/raw_data` folder.


### 1.2. Environment
The code was tested with Python 3.10 and PyTorch 2.2

To install required packages, run the following command:

```
conda create -n thedual python=3.10 -y
conda activate thedual
pip install -r requirements.txt
```

## 2. Download Huggingface checkpoints
Download the checkpoints from Huggingface model hub. The checkpoints and their corresponding Huggingface **MODEL IDs** are:

- LLama3-8B: meta-llama/Meta-Llama-3-8B
- LLama3.1-8B-1-Instruct: meta-llama/Meta-Llama-3.1-8B-Instruct
- Gemma2-9B: google/gemma-2-9b-it
- Qwen2-70B-Instruct: Qwen/Qwen2-72B-Instruct

The checkpoints should be downloaded from Huggingface model hub, and stored in the `hf_checkpoints` folder.

The checkpoints can be downloaded by running the following command:

```
cd hf_checkpoints
git lfs install
git clone git@hf.co:<MODEL ID> # example: git clone git@hf.co:meta-llama/Meta-Llama-3-8B
```

More options on downloading Hugging Face models can be found [here](https://huggingface.co/docs/hub/en/models-downloading).

**Note:** 
- You may need to set up an SSH key on Huggingface account to download the models. Also, some models such as LLaMA may require to fill out a request form for downloading it.
- Make sure the checkpoints are stored in the `hf_checkpoints` folder.
- Please make sure Qwen2-70B-Instruct is fully downloaded. It is a large model (271GB on our disk!) which takes a while to download. If the model is not fully downloaded, the code can still run but the model returns some nonsense characters.



## 3. Fine-tune the models

To fine-tune the model, such as `Llama3-8B` on a NVIDIA A6000 with 48GB RAM, run the following command:

```
python main.py -m -cn main model=llama3_8B optimizer=adamw ++optimizer.params.lr=0.00005 ++optimizer.params.weight_decay=0.1 ++trainer.num_epochs=20 ++trainer.run_name=demo ++trainer.loss_name=macro_double_soft_f1 ++trainer.train_batch_size=1 ++trainer.eval_batch_size=8 ++trainer.accumulate_grad_batches=16 ++model.lora.r=16
```

You can find the list of available models in `configs/models/`. 

## 4. Run inference
### 4.1. Modify config file
Modify `configs/inference.yaml`, only need to change `dataset_path` to point to a **CSV file**. The file should contain a column named `post`, which holds the user posts for classification.



### 4.2. Run inference
The final model is an ensemble of 5 models: one large model (*Qwen2-72B-Instruct*) and 4 smaller models (*LLama3-8B*(s), and *Gemma2-9B*).

You can find our LoRA checkpoints for the four fine-tuned models [here](https://drive.google.com/drive/folders/1RYWH1vgRl5DsvzZgjnLsLUJGBO1pzLxs?usp=sharing). Save the checkpoints in the `checkpoint_thedual` folder.

For **Qwen2-72B-Instruct** model (prompting), it will need to use around 160GB VRAM, eg., 2 NVIDIA A100(s) with 80GB VRAM each, or 4 A6000/A40(s) with 48GB each. 

For the other 4 models, it only need 1 GPU with 48GB VRAM. Folder `checkpoint_thedual` contains LoRA checkpoints of our fine-tuned models (5 checkpoints for each model, since we use 5-fold cross validation).

Thus, inference with 5 models requires 2 A100s (or 4 48GB-VRAM GPUs). 
To run inference, use the following command:

```
python inference.py
```

It will load each model (listed in `configs/inference.yaml`), predict the labels, and save the results in `results/detailed_outputs/` folder. After that, it will ensemble the results and save the final predictions in `results/final_ensemble.xlsx`.


Time estimate for inference: ~1.3 hours on 4 NVIDIA L40(s) (Qwen2-72B-Instruct takes about 40 minutes to run, while the other models take about 10 minutes each).

