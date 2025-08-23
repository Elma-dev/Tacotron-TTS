import yaml
from typing import Dict
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
from other.utils import logger

load_dotenv(dotenv_path="../.env")

def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__=="__main__":
    config_data = load_config("config.yaml")
    logger.info("Set Base Audio Configs...")
    audio_config = BaseAudioConfig(
        sample_rate=config_data["audio"]["sample_rate"],
        win_length=config_data["audio"]["win_length"],
        hop_length=config_data["audio"]["hop_length"],
        fft_size=config_data["audio"]["fft_size"],
        num_mels=config_data["audio"]["num_mels"],
        mel_fmin=config_data["audio"]["mel_fmin"],
        mel_fmax=config_data["audio"]["mel_fmax"],
    )
    logger.info("Set Base Data Configs...")
    dataset_config = BaseDatasetConfig(
        formatter=config_data["dataset"]["formatter"], # ljspeech-style: metadata.csv | wavs/
        meta_file_train=config_data["dataset"]["meta_file_train"],
        path=config_data["dataset"]["path"],
    )
    logger.info("Set Model Configs...")
    config_ = Tacotron2Config(
        audio=audio_config,
        run_name=config_data["model"]["run_name"],
        batch_size=config_data["model"]["batch_size"],
        eval_batch_size=config_data["model"]["eval_batch_size"],
        epochs=config_data["model"]["epochs"],
        print_step=config_data["model"]["print_step"],
        save_step = config_data["model"]["save_step"],
        num_loader_workers=config_data["model"]["num_loader_workers"],
        num_eval_loader_workers=config_data["model"]["num_eval_loader_workers"],
        compute_input_seq_cache=config_data["model"]["compute_input_seq_cache"],
        output_path=config_data["model"]["output_path"],
        datasets=[dataset_config],
        eval_split_size=config_data["model"]["eval_split_size"],
        dashboard_logger=config_data["model"]["dashboard_logger"],
        lr=config_data["model"]["learning_rate"]
    )
    logger.info("Set Processor/Tokenizer...")
    ap = AudioProcessor.init_from_config(config_)
    tokenizer, config_tok = TTSTokenizer.init_from_config(config_)
    logger.info("Load Data Samples...")
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config_tok.eval_split_max_size,
        eval_split_size=config_tok.eval_split_size,
    )
    logger.info("Set Model...")
    model = Tacotron2(config_, ap, tokenizer)
    logger.info("Set Trainer...")
    trainer = Trainer(
        TrainerArgs(),
        config_,
        config_data["model"]["output_path"],
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples
    )
    logger.info("Begin Training...")
    trainer.fit()
    logger.info("End Training")
