import sys, torch, types, json, shutil, os
sys.path.insert(0, r"H:\Applio")
from rvc.train.process.extract_model import extract_model

ckpt = torch.load(r"H:\Applio\logs\TWPretrain\G_29088.pth", map_location="cpu")

with open(r"H:\Applio\logs\TWPretrain\config.json", "r") as f:
    cfg = json.load(f)

hps = types.SimpleNamespace(
    data=types.SimpleNamespace(**cfg["data"]),
    model=types.SimpleNamespace(**cfg["model"]),
    train=types.SimpleNamespace(**cfg["train"]),
)

# Copy model_info.json to output dir so extract_model can read it
out_dir = r"H:\Applio\rvc\models\pretraineds\custom"
shutil.copy(r"H:\Applio\logs\TWPretrain\model_info.json", os.path.join(out_dir, "model_info.json"))

extract_model(
    ckpt=ckpt["model"],
    sr=cfg["data"]["sample_rate"],
    name="TWPretrain",
    model_path=r"H:\Applio\rvc\models\pretraineds\custom\TWPretrain_epoch10.pth",
    epoch=12,
    step=29088,
    hps=hps,
    vocoder="RefineGAN",
    pitch_guidance=True,
    version="v3",
)