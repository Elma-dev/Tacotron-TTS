import configs
import json
import os
from pathlib import Path
import torchaudio
from tqdm import tqdm
from nemo_text_processing.text_normalization import Normalizer
from torchaudio import transforms as T
import numpy as np
import soundfile as sf
import importlib
import codecs
import pandas as pd
import logging
importlib.reload(configs)


logger= logging.getLogger(name="__file__")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)



class DataPreprocessing():
    def __init__(self):
        self.data_path = configs.DATA_PATH
        self.sr = configs.SAMPLING_RATE
        self.out_dir = configs.OUT_DIR
        if not Path(f"{self.out_dir}/wavs").exists():
          os.makedirs(f"{self.out_dir}/wavs")
        self.min_dur = configs.MIN_DUR
        self.max_dur = configs.MAX_DUR
        self.normalizer = Normalizer(input_case='cased', lang='en')
        self.audio_durations = []
        self.nbr_words = []
        self.audio_titles = []

    def read_json_file(self, file_path: str) -> list:
        with codecs.open(file_path, "r", "utf-8-sig") as jf:
            data = json.load(jf)
        return data

    def read_audio(self, audio_path: str):
        audio, sr = torchaudio.load(audio_path)
        if sr != self.sr:
            resample = T.Resample(sr, self.sr)
            audio = resample(audio)
        return audio, self.sr

    def normalize_file_name(self, file_name: str):
        path = Path(os.path.join(self.data_path, file_name + ".json"))
        if path.exists():
            file = os.path.splitext(file_name)[0]
            new_path = path.parent / (file + ".json")
            path.rename(new_path)

    def normalize_text(self, text: str):
        return self.normalizer.normalize(text).strip()

    def segment_audio(self, audio_path):
        base_path = os.path.splitext(audio_path)[0]
        audio, sr = self.read_audio(os.path.join(self.data_path, audio_path))
        words = self.read_json_file(os.path.join(self.data_path, base_path + ".json"))["words"]

        if audio.shape[0] > 1:
            audio = audio.mean(axis=0)

        entries = []
        cur_words, cur_start, cur_end = [], None, None

        for word in words:
            value, s, e = word["value"], float(word["startTime"]), float(word["endTime"])

            if cur_start is None:
                cur_start = s
            cur_end = e
            cur_words.append(value)

            duration = cur_end - cur_start

            if duration >= self.min_dur and (duration >= self.max_dur or value.endswith((".", "!", "?"))):
                start_idx = int(cur_start * sr)
                end_idx = int(cur_end * sr)
                seg_audio = audio[start_idx:end_idx]

                if len(seg_audio) > 1000:
                    seg_id = f"{base_path}_{int(cur_start*100)}"
                    wav_path = os.path.join(f"{self.out_dir}/wavs", seg_id+".wav")
                    sf.write(wav_path, seg_audio.numpy(), sr)
                    transcript = self.normalize_text(" ".join(cur_words))
                    entries.append((f"{seg_id}"," ".join(cur_words).strip(),transcript))

                cur_words, cur_start, cur_end = [], None, None

        return entries

    def info(self):
        return {
            "dataset_total_durations": sum(self.audio_durations),
            "dataset_nbr_words": sum(self.nbr_words),
            "sampling_rate": self.sr
        }

    def __call__(self):
        files = os.listdir(self.data_path)
        all_entries = []

        for file in tqdm(files):
            if file.endswith(".mp3"):
              self.normalize_file_name(file)
              if not Path(f"{self.data_path}/{os.path.splitext(file)[0]}.json").exists():
                continue
              entries = self.segment_audio(file)
              all_entries += entries
        df=pd.DataFrame(all_entries,columns=["path","text","normalized_text"])
        df.to_csv(f"{self.out_dir}/metadata.csv", sep="|", index=False, header=False)
        return df

if __name__=="__main__":
    logger.info("Start Data Preprocessing...")
    data_preprocessing=DataPreprocessing()
    df=data_preprocessing()
    logger.info("Data Preprocessing Finshed.")