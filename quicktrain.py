


model_name =                "TWPretrain-ReGAN"
save_every_epoch =          1
total_epoch =               75
pretrainG =                 None
pretrainD =                 None
gpus =                      "0"
batch_size =                40
sample_rate =               32000
save_only_latest =          False
save_every_weights =        True
cache_data_in_gpu =         False
overtraining_detector =     True
overtraining_threshold =    25
cleanup =                   False
vocoder =                   "RefineGAN"
checkpointing =             True

# remote model saver (optional) works sometimes, maybe
send_model = False
sender_ipaddr = "localhost:8000"
sender_keep_n_models = 9999
secret_code = "jg9438ghjhj9438jJs8ufh0NFYa7s8fnYA08F7Y"

# streaming data server (optional)
stream_data =              False
data_server_url =          "http://localhost:8000/"

import sys
import os

if os.name == "nt":

    print(
        f".\\env\\python.exe .\\rvc\\train\\train.py "
        f"{model_name} "
        f"{save_every_epoch} "
        f"{total_epoch} "
        f"{pretrainG} "
        f"{pretrainD} "
        f"{gpus} "
        f"{batch_size} "
        f"{sample_rate} "
        f"{save_only_latest} "
        f"{save_every_weights} "
        f"{cache_data_in_gpu} "
        f"{overtraining_detector} "
        f"{overtraining_threshold} "
        f"{cleanup} "
        f"{vocoder} "
        f"{checkpointing} "
        f"{send_model} "
        f"{sender_ipaddr} "
        f"{sender_keep_n_models} "
        f"{secret_code} "
        f"{stream_data} "
        f"{data_server_url}"
    )

elif os.name == "posix":

    print(
        f"source .venv/bin/activate\n"
        f"python3 ./rvc/train/train.py "
        f"{model_name} "
        f"{save_every_epoch} "
        f"{total_epoch} "
        f"{pretrainG} "
        f"{pretrainD} "
        f"{gpus} "
        f"{batch_size} "
        f"{sample_rate} "
        f"{save_only_latest} "
        f"{save_every_weights} "
        f"{cache_data_in_gpu} "
        f"{overtraining_detector} "
        f"{overtraining_threshold} "
        f"{cleanup} "
        f"{vocoder} "
        f"{checkpointing} "
        f"{send_model} "
        f"{sender_ipaddr} "
        f"{sender_keep_n_models} "
        f"{secret_code} "
        f"{stream_data} "
        f"{data_server_url}"
    )

input("\n\n> Press Enter to continue")
