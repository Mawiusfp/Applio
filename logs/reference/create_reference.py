import os
import numpy as np
import torch
import librosa
from rvc.lib.predictors.f0 import RMVPE
from rvc.lib.utils import load_embedding, EMBEDDER_PRESETS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def cf0(f0):
    f0_bin = 256
    f0_max = 1100.0
    f0_min = 50.0
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    """Convert F0 to coarse F0."""
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel = np.clip(
        (f0_mel - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1,
        1,
        f0_bin - 1,
    )
    return np.rint(f0_mel).astype(int)


ref = os.path.join(SCRIPT_DIR, "reference.wav")
audio, sr = librosa.load(ref, sr=16000)
trimmed_len = (len(audio) // 320) * 320
audio = audio[:trimmed_len]

print("audio", audio.shape)
rmvpe_model = RMVPE(device="cpu", sample_rate=16000, hop_size=160)
f0 = rmvpe_model.get_f0(audio, filter_radius=0.03)
print("f0", f0.shape)
f0c = cf0(f0)
print("f0c", f0c.shape)

feats = torch.from_numpy(audio).to(torch.float32).to("cpu")
feats = torch.nn.functional.pad(feats.unsqueeze(0), (40, 40), mode="reflect")
feats = feats.view(1, -1)

for name in EMBEDDER_PRESETS:
    out_dir = os.path.join(SCRIPT_DIR, name.replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "feats.npy")
    print(f"Generating reference for {name} -> {out_path}")
    model = load_embedding(name)
    with torch.no_grad():
        feats_out = model(feats)
        if hasattr(feats_out, "last_hidden_state"):
            feats_out = feats_out["last_hidden_state"]
        elif isinstance(feats_out, tuple):
            feats_out = feats_out[0]
        feats_out = feats_out.squeeze(0).float().cpu().numpy()
        print(f"  shape: {feats_out.shape}")
    np.save(out_path, feats_out)

np.save(os.path.join(SCRIPT_DIR, "pitch_coarse.npy"), f0c)
np.save(os.path.join(SCRIPT_DIR, "pitch_fine.npy"), f0)
