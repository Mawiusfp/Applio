import os
import shutil
import numpy as np
import torch
import torchaudio
from random import shuffle
from rvc.configs.config import Config
from rvc.lib.utils import load_embedding, get_embedder_dim
import json

config = Config()
current_directory = os.getcwd()


def generate_config(sample_rate: int, model_path: str):
    config_path = os.path.join("rvc", "configs", f"{sample_rate}.json")
    config_save_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_save_path):
        shutil.copyfile(config_path, config_save_path)


def generate_filelist(model_path: str, sample_rate: int, include_mutes: int = 2):
    gt_wavs_dir = os.path.join(model_path, "sliced_audios")
    feature_dir = os.path.join(model_path, f"extracted")

    f0_dir, f0nsf_dir = None, None
    f0_dir = os.path.join(model_path, "f0")
    f0nsf_dir = os.path.join(model_path, "f0_voiced")

    gt_wavs_files = set(name.split(".")[0] for name in os.listdir(gt_wavs_dir))
    feature_files = set(name.split(".")[0] for name in os.listdir(feature_dir))

    f0_files = set(name.split(".")[0] for name in os.listdir(f0_dir))
    f0nsf_files = set(name.split(".")[0] for name in os.listdir(f0nsf_dir))
    names = gt_wavs_files & feature_files & f0_files & f0nsf_files

    try:
        model_info_path = os.path.join(model_path, "model_info.json")
        with open(model_info_path, "r", encoding="utf-8") as f:
            model_info = json.load(f)
            embedder_name = model_info["embedder_model"]
    except:
        embedder_name = "contentvec"

    # Each embedder needs its own mute directory with correct feature dim
    mute_base_path = os.path.join(current_directory, "logs", f"mute_{embedder_name}")
    mute_feature_path = os.path.join(mute_base_path, "extracted", "mute.npy")
    if not os.path.isfile(mute_feature_path):
        print(f"Generating mute features for embedder '{embedder_name}'...")
        _generate_mute_features(embedder_name, mute_base_path, sample_rate)

    options = []
    sids = []
    for name in names:
        sid = name.split("_")[0]
        if sid not in sids:
            sids.append(sid)
        options.append(
            f"{os.path.join(gt_wavs_dir, name)}.wav|{os.path.join(feature_dir, name)}.npy|{os.path.join(f0_dir, name)}.wav.npy|{os.path.join(f0nsf_dir, name)}.wav.npy|{sid}"
        )

    if include_mutes > 0:
        mute_audio_path = os.path.join(
            mute_base_path, "sliced_audios", f"mute{sample_rate}.wav"
        )
        mute_feature_path = os.path.join(mute_base_path, f"extracted", "mute.npy")
        mute_f0_path = os.path.join(mute_base_path, "f0", "mute.wav.npy")
        mute_f0nsf_path = os.path.join(mute_base_path, "f0_voiced", "mute.wav.npy")

        # adding x files per sid
        for sid in sids * include_mutes:
            options.append(
                f"{mute_audio_path}|{mute_feature_path}|{mute_f0_path}|{mute_f0nsf_path}|{sid}"
            )

    file_path = os.path.join(model_path, "model_info.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
    data.update(
        {
            "speakers_id": len(sids),
        }
    )
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    shuffle(options)

    with open(os.path.join(model_path, "filelist.txt"), "w") as f:
        f.write("\n".join(options))


def _generate_mute_features(embedder_name: str, mute_base_path: str, sample_rate: int):
    """
    Generate mute features for the given embedder using the default mute audio.
    """
    default_mute = os.path.join(current_directory, "logs", "mute")
    src_wav = os.path.join(default_mute, "sliced_audios", f"mute{sample_rate}.wav")
    if not os.path.isfile(src_wav):
        print(f"  No mute audio for {sample_rate}Hz, skipping mute generation.")
        return

    os.makedirs(os.path.join(mute_base_path, "extracted"), exist_ok=True)
    os.makedirs(os.path.join(mute_base_path, "sliced_audios"), exist_ok=True)

    # copy the audio
    dst_wav = os.path.join(mute_base_path, "sliced_audios", f"mute{sample_rate}.wav")
    shutil.copy2(src_wav, dst_wav)

    # copy f0 files (embedder-independent)
    for sub in ("f0", "f0_voiced"):
        src_f0 = os.path.join(default_mute, sub, "mute.wav.npy")
        if os.path.isfile(src_f0):
            os.makedirs(os.path.join(mute_base_path, sub), exist_ok=True)
            shutil.copy2(src_f0, os.path.join(mute_base_path, sub, "mute.wav.npy"))

    # generate features for this embedder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_embedding(embedder_name).to(device)
    model.eval()

    audio, sr = torchaudio.load(dst_wav)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(audio)
    audio_16k = audio.mean(dim=0, keepdim=True)  # mono
    audio_16k = audio_16k.to(device)

    with torch.no_grad():
        feats = model(audio_16k)
        if hasattr(feats, "last_hidden_state"):
            feats = feats["last_hidden_state"]
        elif isinstance(feats, tuple):
            feats = feats[0]
        feats = feats.squeeze(0).float().cpu().numpy()

    np.save(os.path.join(mute_base_path, "extracted", "mute.npy"), feats)
    print(f"  Mute features saved: {feats.shape}")
