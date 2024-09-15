from typing import Union, Dict

import torch
import time
import logging
from datetime import datetime
import os
import pathlib

from omegaconf import OmegaConf

import wandb
from dotenv import load_dotenv

from utils import exists, default, get_machine_name


def init_wandb(args, job_id, project_name):
    # wandb.run.dir
    # https://docs.wandb.ai/guides/track/advanced/save-restore

    try:
        load_dotenv()
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        wandb.login(key=os.getenv("WANDB_API_KEY"))
    except Exception as e:
        print(f"--- was trying to log in Weights and Biases... e={e}")

    ## run_name for wandb's run
    machine_name = get_machine_name()

    watermark = "{}_{}_{}".format(machine_name, job_id, time.strftime("%I-%M%p-%B-%d-%Y"))
    wandb.init(
        project=project_name,
        entity="IEEE_Competition_2024",
        name=watermark,
        settings=wandb.Settings(start_method="fork"),
    )

    # if exists(model):
    ## TODO: fix later
    #     wandb.watch(model, log_freq=log_freq, log_graph=True, log="all")  # log="all" to log gradients and parameters
    return watermark


class MyLogging:
    def __init__(self, args, job_id, project_name):
        self.args = args
        # self.use_ddp = args.hardware.multi_gpus == "ddp"

        init_wandb(
            args,
            project_name=project_name,
            job_id=job_id,
        )

    def info(
        self,
        msg: Union[Dict, str],
        use_wandb=None,
        sep=", ",
        padding_space=False,
        pref_msg: str = "",
    ):

        if isinstance(msg, Dict):
            msg_str = (
                pref_msg
                + " "
                + sep.join(f"{k} {round(v, 4) if isinstance(v, int) else v}" for k, v in msg.items())
            )
            if padding_space:
                msg_str = sep + msg_str + " " + sep
            # if self.use_ddp:
            #     msg_str = f'[GPU{os.environ["LOCAL_RANK"]}]: {msg_str}'

            wandb.log(msg)

            print(msg_str)
        else:
            # if self.use_ddp:
            #     msg = f'[GPU{os.environ["LOCAL_RANK"]}]: {msg}'
            print(msg)

    def log_imgs(self, x, y, y_hat, classes, max_scores, name: str):
        columns = ["image", "pred", "label", "score", "correct"]
        data = []
        for j, image in enumerate(x, 0):
            # pil_image = Image.fromarray(image, mode="RGB")
            data.append(
                [
                    wandb.Image(image[:3]),
                    classes[y_hat[j].item()],
                    classes[y[j].item()],
                    max_scores[j].item(),
                    y_hat[j].item() == y[j].item(),
                ]
            )

        table = wandb.Table(data=data, columns=columns)
        wandb.log({name: table})

    def log_config(self, config):
        wandb.config.update(OmegaConf.to_container(config))  # , allow_val_change=True)

    def finish(
        self,
        use_wandb=None,
        msg_str: str = None,
        model=None,
        model_name: str = "",
        # dummy_batch_x=None,
    ):
        use_wandb = default(use_wandb, self.use_wandb)

        if exists(msg_str):
            if self.use_py_logger:
                self.py_logger.info(msg_str)
            else:
                print(msg_str)
        if use_wandb:
            ## skip for now, not necessary, and takes too much space
            # if model_name:
            #     wandb.save(model_name)
            #     print(f"saved pytorch model {model_name}!")

            # if exists(model):
            #     try:
            #         # https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb#scrollTo=j64Lu7pZcubd
            #         if self.args.hardware.multi_gpus == "DataParallel":
            #             model = model.module
            #         torch.onnx.export(model, dummy_batch_x, "model.onnx")
            #         wandb.save("model.onnx")
            #         print("saved to model.onnx!")
            #     except Exception as e:
            #         print(e)
            wandb.finish()
