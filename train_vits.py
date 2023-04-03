import os
import warnings
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.config.shared_configs import BaseAudioConfig
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.models.vits import Vits
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from trainer import Trainer, TrainerArgs
warnings.filterwarnings('ignore')


def formatter(root_path, manifest_file, **kwargs):
    """custom formatter implementation"""
    pass


def text_cleaner(text):
    """Set of default text cleaners"""
    pass


output_path = 'out'
dataset_path = ''
run_name = "vits_fine_tune"
tpower = 1.3
tpreemphasis = 0.98
tdb = 20

dataset_config = BaseDatasetConfig(formatter="", meta_file_train="metadata.csv", path=dataset_path)

audio_config = BaseAudioConfig(
    sample_rate=22050,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    mel_fmin=0,
    mel_fmax=None,
    power=tpower,
    preemphasis=tpreemphasis,
    ref_level_db=tdb
)

config = VitsConfig(
    audio=audio_config,
    run_name=run_name,
    dashboard_logger="tensorboard",
    batch_size=32,
    eval_batch_size=16,
    batch_group_size=5,
    num_loader_workers=8,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="english_cleaners",  # pass text_cleaner function?
    use_phonemes=True,
    # phoneme_language="en-us",
    # phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=True,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
    cudnn_benchmark=False
)

train_samples, eval_samples = load_tts_samples(
    dataset_config,
    formatter=formatter,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)


ap = AudioProcessor.init_from_config(config)

tokenizer, config = TTSTokenizer.init_from_config(config)

model = Vits(config, ap, tokenizer, speaker_manager=None)

trainer = Trainer(TrainerArgs(), config, output_path,
                  model=model, train_samples=train_samples, eval_samples=eval_samples)

trainer.fit()

