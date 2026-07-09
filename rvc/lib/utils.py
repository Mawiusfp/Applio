import os
import sys
import soxr
import librosa
import soundfile as sf
import numpy as np
import re
import unicodedata
import wget
from torch import nn

import logging
from transformers import (
    HubertModel,
    Wav2Vec2Model,
    WavLMModel,
    AutoConfig,
)
import warnings

# Remove this to see warnings about transformers models
warnings.filterwarnings("ignore")

logging.getLogger("fairseq").setLevel(logging.ERROR)
logging.getLogger("faiss.loader").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

now_dir = os.getcwd()
sys.path.append(now_dir)

base_path = os.path.join(now_dir, "rvc", "models", "formant", "stftpitchshift")
stft = base_path + ".exe" if sys.platform == "win32" else base_path


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        proj_size = getattr(config, "classifier_proj_size", 256)
        self.final_proj = nn.Linear(config.hidden_size, proj_size)


class EmbedderWithProj(nn.Module):
    """Wraps any HuggingFace speech model with final_proj for RVC compatibility."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config
        hidden_size = model.config.hidden_size
        proj_size = getattr(model.config, "classifier_proj_size", 256)
        self.final_proj = nn.Linear(hidden_size, proj_size)

    def forward(self, feats):
        return self.model(feats)


# Preset embedders: short name -> HF model ID (None = use Applio's download URLs)
EMBEDDER_PRESETS = {
    "contentvec": None,
    "spin": None,
    "spin-v2": None,
    "chinese-hubert-base": None,
    "japanese-hubert-base": None,
    "korean-hubert-base": None,
    "hubert-large": "facebook/hubert-large-ll60k",
    "hubert-xlarge": "facebook/hubert-xlarge-ll60k",
    "wavlm-base": "microsoft/wavlm-base",
    "wavlm-base-plus": "microsoft/wavlm-base-plus",
    "wavlm-large": "microsoft/wavlm-large",
}


def get_model_class(model_type):
    mapping = {
        "hubert": HubertModel,
        "wav2vec2": Wav2Vec2Model,
        "wavlm": WavLMModel,
    }
    cls = mapping.get(model_type)
    if cls is None:
        print(f"  Warning: unknown model_type '{model_type}', trying HubertModel")
        cls = HubertModel
    return cls


def get_embedder_dim(embedder_model_or_path, custom_embedder=None):
    """Get the hidden_size of an embedder without loading the full model."""
    embedder_root = os.path.join(now_dir, "rvc", "models", "embedders")
    embedding_list = {
        "contentvec": os.path.join(embedder_root, "contentvec"),
        "spin": os.path.join(embedder_root, "spin"),
        "spin-v2": os.path.join(embedder_root, "spin-v2"),
        "chinese-hubert-base": os.path.join(embedder_root, "chinese_hubert_base"),
        "japanese-hubert-base": os.path.join(embedder_root, "japanese_hubert_base"),
        "korean-hubert-base": os.path.join(embedder_root, "korean_hubert_base"),
    }

    if embedder_model_or_path == "custom" and custom_embedder:
        search_path = custom_embedder
    elif embedder_model_or_path in embedding_list:
        search_path = embedding_list[embedder_model_or_path]
    elif embedder_model_or_path in EMBEDDER_PRESETS:
        hf_id = EMBEDDER_PRESETS[embedder_model_or_path]
        if hf_id:
            cfg = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
            return cfg.hidden_size
        search_path = embedding_list.get(embedder_model_or_path)
    elif "/" in embedder_model_or_path:
        cfg = AutoConfig.from_pretrained(embedder_model_or_path, trust_remote_code=True)
        return cfg.hidden_size
    else:
        search_path = embedder_model_or_path

    if search_path and os.path.exists(os.path.join(search_path, "config.json")):
        cfg = AutoConfig.from_pretrained(search_path, trust_remote_code=True)
        return cfg.hidden_size
    return 768 


def load_audio_16k(file):
    try:
        audio, sr = librosa.load(file, sr=16000)
    except Exception as error:
        raise RuntimeError(f"An error occurred loading the audio: {error}")

    return audio.flatten()


def load_audio(file, sample_rate):
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        audio, sr = sf.read(file)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.T)
        if sr != sample_rate:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=sample_rate, res_type="soxr_vhq"
            )
    except Exception as error:
        raise RuntimeError(f"An error occurred loading the audio: {error}")

    return audio.flatten()


def load_audio_infer(
    file,
    sample_rate,
    **kwargs,
):
    formant_shifting = kwargs.get("formant_shifting", False)
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        if not os.path.isfile(file):
            raise FileNotFoundError(f"File not found: {file}")
        audio, sr = sf.read(file)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.T)
        if sr != sample_rate:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=sample_rate, res_type="soxr_vhq"
            )
        if formant_shifting:
            formant_qfrency = kwargs.get("formant_qfrency", 0.8)
            formant_timbre = kwargs.get("formant_timbre", 0.8)

            from stftpitchshift import StftPitchShift

            pitchshifter = StftPitchShift(1024, 32, sample_rate)
            audio = pitchshifter.shiftpitch(
                audio,
                factors=1,
                quefrency=formant_qfrency * 1e-3,
                distortion=formant_timbre,
            )
    except Exception as error:
        raise RuntimeError(f"An error occurred loading the audio: {error}")
    return np.array(audio).flatten()


def format_title(title):
    formatted_title = unicodedata.normalize("NFC", title)
    formatted_title = re.sub(r"[\u2500-\u257F]+", "", formatted_title)
    formatted_title = re.sub(r"[^\w\s.-]", "", formatted_title, flags=re.UNICODE)
    formatted_title = re.sub(r"\s+", "_", formatted_title)
    return formatted_title


def load_embedding(embedder_model, custom_embedder=None):
    embedder_root = os.path.join(now_dir, "rvc", "models", "embedders")
    embedding_list = {
        "contentvec": os.path.join(embedder_root, "contentvec"),
        "spin": os.path.join(embedder_root, "spin"),
        "spin-v2": os.path.join(embedder_root, "spin-v2"),
        "chinese-hubert-base": os.path.join(embedder_root, "chinese_hubert_base"),
        "japanese-hubert-base": os.path.join(embedder_root, "japanese_hubert_base"),
        "korean-hubert-base": os.path.join(embedder_root, "korean_hubert_base"),
    }

    online_embedders = {
        "contentvec": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/contentvec/pytorch_model.bin",
        "spin": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/spin/pytorch_model.bin",
        "spin-v2": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/spin-v2/pytorch_model.bin",
        "chinese-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/chinese_hubert_base/pytorch_model.bin",
        "japanese-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/japanese_hubert_base/pytorch_model.bin",
        "korean-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/korean_hubert_base/pytorch_model.bin",
    }

    config_files = {
        "contentvec": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/contentvec/config.json",
        "spin": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/spin/config.json",
        "spin-v2": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/spin-v2/config.json",
        "chinese-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/chinese_hubert_base/config.json",
        "japanese-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/japanese_hubert_base/config.json",
        "korean-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/korean_hubert_base/config.json",
    }

    # --- Resolve what to load ---
    hf_id = None
    if embedder_model == "custom":
        if custom_embedder and os.path.exists(custom_embedder):
            return _load_embedder_from_path(custom_embedder)
        print(f"Custom embedder not found: {custom_embedder}, using contentvec")
        embedder_model = "contentvec"

    # Check for HF model ID (contains "/" and not a local preset)
    if "/" in embedder_model and embedder_model not in embedding_list:
        return _load_embedder_from_hf(embedder_model)

    # Check for new presets with HF ID
    if embedder_model in EMBEDDER_PRESETS:
        hf_id = EMBEDDER_PRESETS[embedder_model]

    if hf_id is not None:
        return _load_embedder_from_hf(hf_id)

    # Local preset path
    if embedder_model in embedding_list:
        model_path = embedding_list[embedder_model]
    else:
        model_path = embedder_model  # may be a direct path

    # Download if needed
    if embedder_model in online_embedders:
        bin_file = os.path.join(model_path, "pytorch_model.bin")
        json_file = os.path.join(model_path, "config.json")
        os.makedirs(model_path, exist_ok=True)
        if not os.path.exists(bin_file):
            url = online_embedders[embedder_model]
            print(f"Downloading {url} to {model_path}...")
            wget.download(url, out=bin_file)
        if not os.path.exists(json_file):
            url = config_files[embedder_model]
            print(f"Downloading {url} to {model_path}...")
            wget.download(url, out=json_file)

    return _load_embedder_from_path(model_path)


def _load_embedder_from_path(model_path):
    """Load an embedder from a local path, auto-detecting model type."""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_cls = get_model_class(config.model_type)
    if model_cls is HubertModel:
        model = HubertModelWithFinalProj.from_pretrained(model_path)
    else:
        raw = model_cls.from_pretrained(model_path)
        model = EmbedderWithProj(raw)
    return model


def _load_embedder_from_hf(hf_model_id):
    """Load a HuggingFace speech model and wrap for RVC compatibility."""
    config = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=True)
    model_cls = get_model_class(config.model_type)
    if model_cls is HubertModel:
        model = HubertModelWithFinalProj.from_pretrained(
            hf_model_id, config=config, trust_remote_code=True
        )
    else:
        raw = model_cls.from_pretrained(
            hf_model_id, config=config, trust_remote_code=True
        )
        model = EmbedderWithProj(raw)
    return model
