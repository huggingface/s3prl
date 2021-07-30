"""
This is just an example of what people would submit for
inference.
"""

from s3prl.downstream.runner import Runner
from typing import Dict
import torch
import os


class PreTrainedModel(Runner):
    def __init__(self, path=""):
        """
        Initialize downstream model.
        """
        ckp_file = os.path.join(path, "model.ckpt")
        ckp = torch.load(ckp_file, map_location='cpu')
        ckp["Args"].init_ckpt = ckp_file
        ckp["Args"].mode = "inference"
        ckp["Args"].device = "cpu" # Just to try in my computer
        ckp["Config"]["downstream_expert"]["datarc"]["dict_path"]=os.path.join(path,'char.dict')

        Runner.__init__(self, ckp["Args"], ckp["Config"])

    def __call__(self, inputs)-> Dict[str, str]:
        """
        Args:
            inputs (:obj:`np.array`):
                The raw waveform of audio received. By default at 16KHz.
        Return:
            A :obj:`dict`:. The object return should be liked {"text": "XXX"} containing
            the detected text from the input audio.
        """
        for entry in self.all_entries:
            entry.model.eval()

        inputs = [torch.FloatTensor(inputs)]

        with torch.no_grad():
            features = self.upstream.model(inputs)
            features = self.featurizer.model(inputs, features)
            preds = self.downstream.model.inference(features, [])
        return {"text": preds[0]}


"""
import subprocess
import numpy as np
from datasets import load_dataset
# This is already done in the Inference API
def ffmpeg_read(bpayload: bytes, sampling_rate: int) -> np.array:
    ar = f"{sampling_rate}"
    ac = "1"
    format_for_conversion = "f32le"
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        "pipe:0",
        "-ac",
        ac,
        "-ar",
        ar,
        "-f",
        format_for_conversion,
        "-hide_banner",
        "-loglevel",
        "quiet",
        "pipe:1",
    ]

    ffmpeg_process = subprocess.Popen(
        ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    output_stream = ffmpeg_process.communicate(bpayload)
    out_bytes = output_stream[0]

    audio = np.frombuffer(out_bytes, np.float32).copy()
    if audio.shape[0] == 0:
        raise ValueError("Malformed soundfile")
    return audio


model = PreTrainedModel()
ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
filename = ds[0]["file"]
with open(filename, "rb") as f:
    data = ffmpeg_read(f.read(), 16000)
    print(model(data)) 
"""