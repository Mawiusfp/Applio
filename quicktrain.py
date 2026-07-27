



model_name =                "TWPretrain-HiFi-GAN"
save_every_epoch =          1
total_epoch =               1
pretrainG =                 None
pretrainD =                 None
gpus =                      "0"
batch_size =                8
sample_rate =               32000
save_only_latest =          False               # Enabling this setting will result in the G and D files saving only their most recent versions, effectively conserving storage space.
save_every_weights =        True                # This setting enables you to save the weights of the model at the conclusion of each epoch.
cache_data_in_gpu =         True
overtraining_detector =     True
overtraining_threshold =    25
cleanup =                   False
vocoder =                   "RefineGAN"
checkpointing =             False

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

args = (
    f"--model_name {model_name} "
    f"--save_every_epoch {save_every_epoch} "
    f"--total_epoch {total_epoch} "
    f"--pretrainG {pretrainG} "
    f"--pretrainD {pretrainD} "
    f"--gpus {gpus} "
    f"--batch_size {batch_size} "
    f"--sample_rate {sample_rate} "
    f"--save_only_latest {save_only_latest} "
    f"--save_every_weights {save_every_weights} "
    f"--cache_data_in_gpu {cache_data_in_gpu} "
    f"--overtraining_detector {overtraining_detector} "
    f"--overtraining_threshold {overtraining_threshold} "
    f"--cleanup {cleanup} "
    f"--vocoder {vocoder} "
    f"--checkpointing {checkpointing} "
    f"--send_model {send_model} "
    f"--sender_ipaddr {sender_ipaddr} "
    f"--sender_keep_n_models {sender_keep_n_models} "
    f"--secret_code {secret_code} "
    f"--stream_data {stream_data} "
    f"--data_server_url {data_server_url}"
)

if os.name == "nt":

    print(f".\\env\\python.exe .\\rvc\\train\\train.py {args}")

elif os.name == "posix":

    print(f"source .venv/bin/activate\npython3 ./rvc/train/train.py {args}")

input("\n\n> Press Enter to continue")
