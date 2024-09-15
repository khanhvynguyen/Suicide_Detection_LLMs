import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from omegaconf import OmegaConf

import random
from process_data import prepare_data_loader
from sklearn.metrics import confusion_matrix
from transformers import AutoModelForSequenceClassification
import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from transformers import AutoTokenizer
import torch
from metrics import weighted_f1
from sklearn.metrics import classification_report
from lightning.pytorch.loggers import WandbLogger
import wandb
from torch import Tensor
from lightning.pytorch import seed_everything
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from loss_fn import compute_loss
from utils import *


def store_checkpoint(model, epoch: Optional[int], save_path: str):
    state = model.state_dict()
    for name in list(state.keys()):
        if "lora" not in name and "modules_to_save" not in name:
            state.pop(name)

    ## save to file
    if epoch is None:
        torch.save(state, f"{save_path}/last.ckpt")
    else:
        torch.save(state, f"{save_path}/epoch_{epoch}.ckpt")


def get_checkpoint(criteria, checkpoint_path):
    if criteria == "best":
        ## list all files in the checkpoint_path start with "epoch"
        files = os.listdir(checkpoint_path)
        files = [f for f in files if f.startswith("epoch")]
        assert len(files) == 1
        return files[0]
    elif criteria == "last":
        return "last.ckpt"
    else:
        raise ValueError(f"Invalid criteria: {criteria}")


@torch.inference_mode()
def predict(model, data_loader):
    model.eval()
    y_pred_probs = []
    y_true = []
    for batch in data_loader:
        input_ids = batch["input_ids"]
        labels = batch["label"] if "label" in batch else None
        attention_mask = batch["attention_mask"]
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)["logits"]

        y_pred_probs.append(outputs)
        y_true.append(labels)
    y_pred_probs = torch.cat(y_pred_probs)
    y_pred_probs = torch.nn.functional.softmax(y_pred_probs, dim=1)
    if labels is not None:
        y_true = torch.cat(y_true)
    else:
        y_true = None
    return y_pred_probs, y_true


def compute_metric(y_pred_prob: Tensor, y_true: Tensor, logger):
    y_pred = torch.argmax(y_pred_prob, 1)
    f1 = weighted_f1(
        y_true=y_true.detach().cpu().numpy(),
        y_pred=y_pred.detach().cpu().numpy(),
        convert_to_int=False,
    )
    acc = (torch.sum(y_pred == y_true) / len(y_true)).item() * 100
    logger.print_and_log(confusion_matrix(y_true=y_true, y_pred=y_pred))
    logger.print_and_log(classification_report(y_true=y_true, y_pred=y_pred))
    return f1, acc


def evaluate(model, data_loader, logger):
    ## predict and then compute metrics
    y_pred_probs, y_true = predict(model, data_loader)
    f1, acc = compute_metric(y_pred_probs, y_true, logger)
    return f1, acc, y_pred_probs, y_true


def get_torch_model(config):
    ## model names: https://huggingface.co/docs/transformers/en/model_doc/auto
    if config.model.lora is not None:
        quan_config = BitsAndBytesConfig(
            load_in_4bit=True,  # enable 4-bit quantization
            bnb_4bit_quant_type="nf4",  # information theoretically optimal dtype for normally distributed weights
            bnb_4bit_use_double_quant=True,  # quantize quantized weights
            bnb_4bit_compute_dtype=torch.bfloat16,  # optimized fp format for ML
        )

        lora_config = LoraConfig(**config.model.lora)
        if "trainer" in OmegaConf.to_container(config):
            torch_dtype = torch.bfloat16 if config.trainer.precision == "16-mixed" else torch.float32
            num_labels = config.trainer.num_classes
            device_map = "auto"
        else:  ## inference, hardcode for now
            torch_dtype = torch.bfloat16
            num_labels = 4
            device_map = "cuda"
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model.model_name,
            num_labels=num_labels,
            torch_dtype=torch_dtype,
            quantization_config=quan_config,
            device_map=device_map,
        )

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        tokenizer = AutoTokenizer.from_pretrained(config.model.model_name, add_prefix_space=True)

        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        model.config.pretraining_tp = 1
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model.model_name,
            num_labels=config.trainer.num_classes,
            ignore_mismatched_sizes=True,
        )

    return model


def train_fold(
    fold_i,
    config,
    tokenizer,
    max_length,
    logger,
    watermark,
):
    train_path = config.data.train_path.format(fold_i=fold_i)
    val_path = config.data.val_path
    val_path = val_path.format(fold_i=fold_i) if val_path is not None else None
    checkpoint_path = config.trainer.checkpoint_path

    logger.print_and_log(f"train_path: {train_path}")
    logger.print_and_log(f"val_path: {val_path}")

    extra_data_paths = config.data.extra_data_paths
    extra_data_paths_new = []
    if extra_data_paths is not None:
        for path, truncate in extra_data_paths:
            if "data/augmentation/augment_clean/cv_{i}.csv" in path:
                all_other_folds = [i for i in range(1, 6) if i != fold_i]
                augment_paths = [
                    ["data/augmentation/augment_clean/cv_{i}.csv".format(i=i), "middle"]
                    for i in all_other_folds
                ]
                extra_data_paths_new.extend(augment_paths)
            else:
                extra_data_paths_new.append([path, truncate])
    extra_data_paths = extra_data_paths_new
    logger.print_and_log(f"--------+++++-------- extra_data_paths: {extra_data_paths}")
    model = get_torch_model(config)
    analyze_model(model)

    logger.print_and_log(model)
    preprocess_fn = None

    if config.model.layers_to_fine_tune == "last":
        for param in model.parameters():
            param.requires_grad = False
        if config.model.name in [
            "bert",
            "deproberta",
        ]:
            for param in model.classifier.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"model name '{config.model.name}' not implemented yet!")

    if checkpoint_path is not None:
        logger.print_and_log(f"fine_tune->train_fold(): loading {checkpoint_path}...")
        replace_cls_head = config.trainer.replace_cls_head
        model = load_torch_model(
            model_path=checkpoint_path,
            model=model,
            replace_cls_head=replace_cls_head,
        )

    extra_params = {}
    extra_params["train_path"] = train_path
    extra_params["extra_data_paths"] = extra_data_paths
    extra_params["tokenizer"] = tokenizer
    extra_params["max_length"] = max_length
    extra_params["sample_extra_data"] = config.trainer.sample_extra_data

    clear_gpu_mem(verbose=True)

    train_loader = prepare_data_loader(
        train_path,
        tokenizer=tokenizer,
        shuffle=True,
        batch_size=config.trainer.train_batch_size,
        workers=config.trainer.workers,
        max_length=max_length,
        truncate_position=config.model.truncate_position,
        preprocess_fn=preprocess_fn,
        is_train=True,
        extra_data_paths=extra_data_paths,
        sample_extra_data=config.trainer.sample_extra_data,
    )
    if val_path is not None:
        val_loader = prepare_data_loader(
            val_path,
            tokenizer=tokenizer,
            shuffle=False,
            batch_size=config.trainer.eval_batch_size,
            workers=config.trainer.workers,
            max_length=max_length,
            truncate_position=config.model.truncate_position,
            preprocess_fn=preprocess_fn,
            is_train=False,
            extra_data_paths=None,
        )
    else:
        val_loader = None

    if config.trainer.eval_only:
        logger.print_and_log(f"============== Skipped training.")
        if config.trainer.checkpoint_dir:
            checkpoint_name = get_checkpoint(
                config.trainer.checkpoint_criteria, f"{config.trainer.checkpoint_dir}/fold_{fold_i}"
            )
            path_i = os.path.join(config.trainer.checkpoint_dir, f"fold_{fold_i}/{checkpoint_name}")
            logger.print_and_log(f"===== loading model from {path_i}")
            model = load_torch_model(
                model_path=path_i,
                model=model,
                replace_cls_head=False,
            )
        val_f1, val_acc, val_y_pred_probs, val_y_true = evaluate(model, val_loader, logger)
    else:
        logger.print_and_log(f"training model for fold {fold_i}...")

        cfg = config.optimizer
        optimizer = make_my_optimizer(cfg.name, model.parameters(), cfg.params)
        num_epochs = config.trainer.num_epochs

        best_val_f1 = 0
        val_f1, val_acc = 0, 0

        for epoch in tqdm(range(num_epochs)):
            model.train()

            total_loss = 0
            train_y_true = []
            train_y_pred_probs = []
            for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):

                optimizer.zero_grad()
                input_ids = batch["input_ids"]
                labels = batch["label"]
                attention_mask = batch["attention_mask"]
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)["logits"]

                loss = compute_loss(
                    config.trainer.loss_name, outputs, labels, weight=config.trainer.class_weights
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                train_y_true.append(labels)
                train_y_pred_probs.append(outputs)

            train_loss = total_loss / len(train_loader)
            train_y_true = torch.cat(train_y_true)
            train_y_pred_probs = torch.cat(train_y_pred_probs)
            train_f1, train_acc = compute_metric(train_y_pred_probs, train_y_true, logger)

            ### evaluate on valid
            if val_loader is not None:
                val_f1, val_acc, val_y_pred_probs, val_y_true = evaluate(model, val_loader, logger)

            ## save checkpoint
            # Always save the last checkpoint
            save_path = f"checkpoint_thedual/{watermark}/fold_{fold_i}"
            os.makedirs(save_path, exist_ok=True)
            store_checkpoint(model=model, epoch=None, save_path=save_path)

            # And, save the best one as well
            if best_val_f1 < val_f1:
                best_val_f1 = val_f1
                ## remove the old checkpoint first
                # get all files in the folder, starting with epoch
                files = os.listdir(save_path)
                files = [f for f in files if f.startswith("epoch")]
                for f in files:
                    os.remove(os.path.join(save_path, f))

                store_checkpoint(model=model, epoch=epoch, save_path=save_path)
            msg = f"Fold={fold_i}: Epoch {epoch} |  train loss: {round(train_loss,3)} | train f1: {round(train_f1,4)} | train acc: {round(train_acc,2)}% | val f1: {round(val_f1,4)} | val acc: {round(val_acc,2)}%"
            logger.print_and_log(msg)

    res = {"f1": val_f1, "accuracy": val_acc}
    return res, val_y_pred_probs, val_y_true


@timer_func
def fine_tune_v2(config):
    if config.model.name.startswith("gemma2_9b"):
        torch.set_float32_matmul_precision("medium")  ## check

    jobid = config.trainer.jobid
    seed = default(config.trainer.get("seed"), random.randint(100, 1000000))
    config.trainer.seed = seed

    # sets seeds for numpy, torch and python.random.
    seed_everything(2024, workers=True)

    print("config:", config)
    config.trainer.jobid = default(config.trainer.jobid, "")

    # set_seeds(seed)
    cur_time = datetime.today().strftime("%B-%d-%Y-%H-%M-%S")
    if config.trainer.run_name is not None:
        watermark = f"{config.trainer.run_name}_{cur_time}--id_{jobid}--seed_{seed}"
    else:
        watermark = f"{cur_time}--id_{jobid}--seed_{seed}"
    get_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if config.trainer.use_wandb and get_rank == 0:
        logger = WandbLogger(
            project="the_dual_v4",
            entity="IEEE_Competition_2024",
            name=watermark,
            settings=wandb.Settings(start_method="fork"),
        )
        logger.print_and_log = print
    else:
        path = f"logs/use_cv-{config.trainer.use_cv}--{watermark}.log"
        logger = DummyLogger(path)
    ## flatten the config
    config_flatten = flatten(config)
    config_flatten["full_config"] = config
    logger.log_hyperparams(config_flatten)

    ## Get tokenizer
    if config.model.lora is not None:
        tokenizer = AutoTokenizer.from_pretrained(config.model.model_name, add_prefix_space=True)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    if config.model.model_max_length is not None:
        tokenizer.model_max_length = config.model.model_max_length

    max_length = tokenizer.model_max_length

    logger.print_and_log(f"Tokenizer input max length: {max_length}")
    logger.print_and_log(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
    logger.log_metrics({"model_max_length": max_length, "vocab_size": tokenizer.vocab_size})

    ##  cross-validation
    all_val_res = []
    all_y_val_preds = []
    all_y_val_true = []
    cur_date = datetime.today().strftime("%Y-%m-%d")
    os.makedirs(f"submission/{cur_date}", exist_ok=True)
    os.makedirs(f"val_preds/{cur_date}", exist_ok=True)
    os.makedirs(f"unlabeled_preds/{cur_date}", exist_ok=True)

    ## check if config.trainer.use_cv is int or bool
    if str(config.trainer.use_cv).isdigit():
        fold_list = [config.trainer.use_cv]
        num_folds = 1
    elif config.trainer.use_cv:
        logger.print_and_log("====== Set up for cv (5 folds) ...")
        num_folds = 5
        fold_list = range(1, num_folds + 1)
    else:
        logger.print_and_log("====== set up for 1 fold ...")
        num_folds = 1
        fold_list = [1]

    ## training
    for fold_i in fold_list:
        if not config.trainer.eval_only:
            logger.print_and_log(f"\n-------------- training fold {fold_i}/{num_folds}...")
        res, y_val_preds, y_val_true = train_fold(
            fold_i,
            config,
            tokenizer,
            max_length,
            logger,
            watermark,
        )
        all_val_res.append(res)
        all_y_val_preds.append(y_val_preds)
        all_y_val_true.append(y_val_true)
        for k, v in res.items():
            logger.log_metrics({f"{k}/fold_{fold_i}": v})

        clear_gpu_mem(verbose=True)

    logger.print_and_log("------------------=========================== Done training!!!!")
    ## save all val preds for error analysis
    all_y_val_preds_concat = torch.cat(all_y_val_preds)

    pred = torch.argmax(all_y_val_preds_concat, 1).detach().cpu().numpy()
    all_y_val_preds_concat = all_y_val_preds_concat.detach().cpu().numpy()
    all_y_val_true_concat = torch.cat(all_y_val_true).detach().cpu().numpy()
    all_y_val_preds_df = pd.DataFrame(all_y_val_preds_concat)
    all_y_val_preds_df.columns = [f"pred_{i}" for i in range(config.trainer.num_classes)]
    all_y_val_preds_df["label"] = all_y_val_true_concat
    all_y_val_preds_df["pred"] = pred
    all_y_val_preds_df.to_csv(f"val_preds/{cur_date}/{watermark}.csv", index=False)
    logger.print_and_log(f"saved val_preds/{cur_date}/{watermark}.csv")
    logger.print_and_log("--- result on all 500 labeled data:")
    logger.print_and_log(classification_report(y_true=all_y_val_true_concat, y_pred=pred))
    logger.print_and_log(f"{confusion_matrix(y_true=all_y_val_true_concat, y_pred=pred)}")
    val_f1_500 = weighted_f1(all_y_val_true_concat, pred)
    logger.log_metrics({"val_f1_500": val_f1_500})

    show_cv_result(all_val_res)

    if config.data.test_path is None:
        logger.print_and_log("No test data provided, skipping prediction...")
    else:
        ## predict: make prediction for test set
        test_loader = prepare_data_loader(
            config.data.test_path,
            tokenizer=tokenizer,
            shuffle=False,
            batch_size=config.trainer.eval_batch_size,
            workers=config.trainer.workers,
            max_length=max_length,
            truncate_position=config.model.truncate_position,
            preprocess_fn=make_preprocess_fn(config),
            is_train=False,
            extra_data_paths=None,
        )

        if config.trainer.get("skip_unlabeled_data", False):
            unlabeled_loader = None
        else:
            unlabeled_loader = prepare_data_loader(
                config.data.unlabeled_path,
                tokenizer=tokenizer,
                shuffle=False,
                batch_size=config.trainer.eval_batch_size,
                workers=config.trainer.workers,
                max_length=max_length,
                truncate_position=config.model.truncate_position,
                preprocess_fn=make_preprocess_fn(config),
                is_train=False,
                extra_data_paths=None,
            )

        preds_folds = []
        unlabeled_folds = []

        ### inference
        for fold_i in fold_list:
            ## Model
            model = get_torch_model(config)
            if config.trainer.eval_only:
                assert (
                    config.trainer.checkpoint_dir or config.trainer.checkpoint_path
                ), "checkpoint_dir or checkpoint_path must be provided for eval_only mode!"
                if config.trainer.checkpoint_dir:
                    checkpoint_name = get_checkpoint(
                        config.trainer.checkpoint_criteria, f"{config.trainer.checkpoint_dir}/fold_{fold_i}"
                    )
                    path_i = os.path.join(config.trainer.checkpoint_dir, f"fold_{fold_i}/{checkpoint_name}")
                else:
                    path_i = config.trainer.checkpoint_path
                logger.print_and_log(f"--------loading model from checkpoint for INFERENCE: {path_i}...")
            else:
                checkpoint_name = get_checkpoint(
                    config.trainer.checkpoint_criteria, f"checkpoint_thedual/{watermark}/fold_{fold_i}"
                )
                path_i = f"checkpoint_thedual/{watermark}/fold_{fold_i}/{checkpoint_name}"
                logger.print_and_log(f"--------loading the newly trained model for INFERENCE: {path_i}...")

            model = load_torch_model(model_path=path_i, model=model)
            logger.print_and_log("----- Done loading model for INFERENCE!!!")

            y_pred_probs, _ = predict(model, test_loader)
            preds_folds.append(y_pred_probs)

            if unlabeled_loader is not None:
                y_pred_probs_unlabeled, _ = predict(model, unlabeled_loader)
                unlabeled_folds.append(y_pred_probs_unlabeled)

            clear_gpu_mem(verbose=True)

        # if config.model.combine_folds == "avg":
        ## store final_pred_probs for test set
        final_pred_probs_all = torch.stack(preds_folds)  ## (5, num_samples, num_classes)

        ## store final_pred_probs for unlabeled set
        if unlabeled_loader is not None:
            final_pred_probs_unlabeled_all = torch.stack(unlabeled_folds)
            final_pred_probs_unlabeled_avgbyprob = final_pred_probs_unlabeled_all.mean(dim=0)
            final_pred_probs_unlabeled_vote = torch.argmax(final_pred_probs_unlabeled_all, dim=-1)
            final_pred_probs_unlabeled_vote = final_pred_probs_unlabeled_vote.mode(dim=0).values

        ## prepare to submit
        avg_f1 = np.mean([res["f1"] for res in all_val_res])
        std_f1 = np.std([res["f1"] for res in all_val_res])
        logger.log_metrics({"avg_f1": avg_f1, "std_f1": std_f1})

        file_name = f"TheDual-{round(avg_f1,4)}--std{round(std_f1,4)}--{watermark}.xlsx"
        file_path = os.path.join(f"submission/{cur_date}", file_name)

        final_pred_probs_all_np = final_pred_probs_all.detach().cpu().numpy()
        if unlabeled_loader is not None:
            final_pred_probs_unlabeled_all_np = final_pred_probs_unlabeled_all.detach().cpu().numpy()

        ## store npy file
        np.save(
            f"submission/np/final_pred_probs_all_{cur_date}_{round(avg_f1,4)}--std{round(std_f1,4)}--{watermark}.npy",
            final_pred_probs_all_np,
        )
        if unlabeled_loader is not None:
            np.save(
                f"submission/np/final_pred_probs_unlabeled_all_{cur_date}_{round(avg_f1,4)}--std{round(std_f1,4)}--{watermark}.npy",
                final_pred_probs_unlabeled_all_np,
            )

        ## save to csv file
        num_classes = config.trainer.num_classes
        if unlabeled_loader is not None:
            df_unlabeled = store_df(final_pred_probs_unlabeled_avgbyprob, num_classes, "UNLABELED_avgbyprob")
            # df_unlabeled_vote = store_df(final_pred_probs_unlabeled_vote, num_classes, "UNLABELED_vote")

            unlabeled_file_path = (
                f"unlabeled_preds/{cur_date}/pred_{round(avg_f1,4)}--std{round(std_f1,4)}--{watermark}.csv"
            )
            df_unlabeled.to_csv(unlabeled_file_path, index=False)
            logger.print_and_log(f"wrote df_unlabeled to {unlabeled_file_path}!")

        # final_pred_probs = convert_numer_to_label(final_pred_probs.detach().cpu().numpy())
        prepare_to_submit(final_pred_probs_all, file_path)

    if config.trainer.clean_up:
        os.system(f"rm -rf checkpoint_thedual/{watermark}")
        logger.print_and_log(f"cleaned up checkpoint_thedual/{watermark}")
    if config.trainer.use_wandb:
        wandb.finish()
