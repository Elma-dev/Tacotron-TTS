# Trump TTS Project

## Project Overview

This project aims to create a Text-to-Speech (TTS) model capable of generating audio in the voice of Donald Trump. It leverages the Coqui TTS library, specifically fine-tuning a Tacotron2 model on a dataset of Donald Trump's speeches. The goal is to provide a robust and customizable TTS solution for generating realistic-sounding audio.

## Directory Structure

```
.
├── README.md
├── pyproject.toml
├── data_prep.ipynb
├── preprocessing/
│   └── preprocessing.py
│   └── configs.py
├── pretraining/
│   └── config.yaml
│   └── train.py
├── finetuning/
│   └── tactoron_ft.bash
├── inference/
│   └── inference.bash
│   └── inference.ipynb
│   └── wavs/
│       └── trump1.wav
│       └── trump2.wav
└── other/
    └── utils
```

## Thought Process and High-Level Approach

The primary thought process behind this project was to leverage existing, well-established TTS architectures for fine-tuning, rather than building a model from scratch. This approach significantly reduces development time and allows for faster iteration.

### High-Level Steps:

1.  **Data Acquisition and Preprocessing:** Using a dataset of Trump's speeches (specifically from Kaggle: [etaifour/trump-speeches-audio-and-word-transcription](https://www.kaggle.com/datasets/etaifour/trump-speeches-audio-and-word-transcription)) and preprocess the audio and text to be compatible with the Coqui TTS format (LJSpeech-style).
2.  **Configuration Management:** Centralize hyper-parameters and model configurations in a YAML file (`config.yaml`) for easy modification and experimentation.
3.  **Model Fine-tuning:** Utilize the `train.py` script to fine-tune a pre-trained Tacotron2 model on the prepared dataset.
4.  **Inference and Evaluation:** After training, use the fine-tuned model to generate new speech and evaluate its quality.

## Architectural Choices

-   **Tacotron2 Model:** Tacotron2 was chosen as the primary TTS model due to its widespread adoption, strong performance in generating natural-sounding speech, and its compatibility with the Coqui TTS framework. It's an attention-based sequence-to-sequence model that predicts a mel-spectrogram from input text, which is then converted into audio by a vocoder.

-   **Coqui TTS Library:** The Coqui TTS library provides a comprehensive and flexible framework for building and training TTS models. It offers pre-trained models, utility functions for data processing, and a well-defined training pipeline, which significantly accelerates development.

-   **YAML for Configuration:** Using a `config.yaml` file for hyper-parameters and other configurations enhances modularity and readability. It allows for easy adjustment of training parameters without modifying the code directly, facilitating experimentation and reproducibility.

-   **Data Preprocessing Notebook (`data_prep.ipynb`):** A Jupyter notebook is used for data preprocessing to provide an interactive and step-by-step approach to cleaning, segmenting, and formatting the audio-text pairs. This makes the data preparation process transparent and easier to debug.
    -   **`prerocessing.py`:** This script contains the `DataPreprocessing` class, which handles the core logic for audio segmentation, text normalization, and generating the `metadata.csv` file required by the Coqui TTS training pipeline. It reads raw audio and JSON transcription files, segments audio into shorter clips based on word timings, normalizes the accompanying text, and saves the processed audio and metadata. This ensures the data is in the correct format and meets the duration requirements for effective model training.

    ### Detailed Data Preprocessing Steps
    <img width="4502" height="436" alt="image" src="https://github.com/user-attachments/assets/537b4b2e-328c-4ff8-9c0b-bd6a529744d0" />


    The `DataPreprocessing` class orchestrates the entire data preparation pipeline. Here's a breakdown of its key methods:

    #### `__init__(self)`
    -   **Purpose:** Initializes the DataPreprocessing object, setting up paths, sampling rate, output directory, minimum/maximum audio durations, and the text normalizer.
    -   **Details:** It creates the output directory for processed WAV files if it doesn't already exist.

    #### `read_json_file(self, file_path: str) -> list`
    -   **Purpose:** Reads and parses a JSON file.
    -   **Details:** Used to load the word-level transcriptions and timings associated with each audio file.

    #### `read_audio(self, audio_path: str)`
    -   **Purpose:** Loads an audio file and resamples it if necessary.
    -   **Details:** Ensures all audio files conform to the project's specified sampling rate (`self.sr`). Handles multi-channel audio by converting it to mono.

    #### `normalize_file_name(self, file_name: str)`
    -   **Purpose:** Corrects file names by removing the `.mp3` extension when looking for corresponding JSON files.
    -   **Details:** This ensures that the `.json` file can be correctly matched with its `.mp3` counterpart, even if the `.mp3` extension was accidentally included in the JSON file name reference.

    #### `normalize_text(self, text: str)`
    -   **Purpose:** Normalizes the input text.
    -   **Details:** It uses the NeMo `Normalizer` to convert numbers, abbreviations, and symbols into their full textual representations, which is crucial for training high-quality TTS models.

    #### `segment_audio(self, audio_path)`
    -   **Purpose:** Segments a single audio file into smaller, utterance-level clips.
    -   **Details:** This is the core segmentation logic. It iterates through word timings from the JSON file, accumulating words until a segment reaches a minimum duration or a sentence-ending punctuation is encountered. These segments are then saved as WAV files, and their normalized transcripts are prepared for the `metadata.csv`.

    #### `info(self)`
    -   **Purpose:** Provides a summary of the processed dataset.
    -   **Details:** Returns a dictionary containing the total duration of audio processed, the total number of words, and the sampling rate.

    #### `__call__(self)`
    -   **Purpose:** Executes the main data preprocessing pipeline.
    -   **Details:** This method is invoked when the `DataPreprocessing` object is called as a function. It iterates through all MP3 files in the data directory, attempts to find their corresponding JSON transcription files, processes them using the `segment_audio` method, and finally compiles all entries into a Pandas DataFrame. This DataFrame is then saved as `metadata.csv`, which is the input for the TTS model training.

## Challenges and Solutions

-   **Training from Scratch vs. Fine-tuning:** Initial attempts to train the model from scratch using `trainer.py` on the custom dataset did not yield satisfactory results, demonstrating difficulty in achieving good convergence and natural-sounding speech. Consequently, the approach shifted directly to fine-tuning a pre-trained model, which significantly improved results and proved more effective for this task with limited domain-specific data.
-   **Data Quality and Quantity:** Obtaining a clean and sufficiently large dataset of Trump's speeches (1.5h) with accurate transcriptions was a primary challenge. Solutions involved using publicly available datasets and careful manual verification/correction of transcriptions.
-   **Computational Resources:** Training large TTS models like Tacotron2 can be computationally intensive. All experiments, including both the initial attempts at training from scratch and the subsequent fine-tuning processes, were conducted remotely via SSH on an RTX 4090 GPU with 24GB of VRAM. Each significant experiment typically required more than one day to complete. The solution involves leveraging such GPU resources and optimizing batch sizes and other training parameters to make efficient use of available hardware.
-   **Environment Setup:** Ensuring all dependencies and the Coqui TTS library are correctly installed can be tricky due to potential conflicts. Providing clear `pip install` instructions and recommending a virtual environment helps mitigate this.
-   **Fine-tuning Convergence:** Achieving good convergence and natural-sounding speech during fine-tuning can be challenging. Experimentation with learning rates, optimizer choices, and monitoring loss curves are crucial steps. The use of **WandB** for logging helps in tracking and comparing experiments.

## Setup and Running Instructions

Follow these steps to set up the environment and run the project:

### 1. Clone the Repository

```bash
git clone https://github.com/Elma-dev/Tacotron-TTS.git
cd trump_tts
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate 
```

### 3. Install Dependencies

```bash
uv pip install -r pyproject.toml
```

### 4. Prepare Data

Run the `preprocessing/preprocessing.py` on the audio and text data. This script will guide you through the steps to create the `metadata.csv` and processed audio files in the `./processed_data` directory.

```bash
uv run preprocessing/preprocessing.py
```

Before running dataset preprocessing, try to review `pretraining/config.yaml`, also set `.env` based on `.env.example`.

### 5. Configure Hyper-parameters

Review and adjust the hyper-parameters in `pretraining/config.yaml` as needed.

### 6. Train the Model from Scratch

Start the training process by running the `pretraining/trainer.py` script for full training from scratch:

```bash
uv run pretraining/trainer.py --config_path pretraining/config.yaml
```

### 7. Fine-tuning the Tacotron2 Model

For fine-tuning the Tacotron2 model using the CoquiTTS CLI, execute the `finetuning/tacotron_ft.bash` script. This script encapsulates the necessary steps, including model loading, configuration adjustments, GPU cache clearing, and initiating the training process. Please ensure that the `config.json` file path and `model.pth` restore path within the script are correctly pointing to your model's assets.

```bash
./finetuning/tacotron_ft.bash
```

### Hyperparameter Experimentation

Below is a table summarizing some hyperparameter experiments and their observed effects:

| Hyperparameter | Values Tested | Observed Experience |
|---|---|---|
| Learning Rate | 1e-4, 5e-5, 4e-5, 3e-5 | Lower rates (4e-5) led to better convergence and reduced overfitting. |
| Batch Size | 8, 16, 32 | Smaller batch sizes (16) provided more stable training, especially early on. |
| Number of Epochs | 10, 29, 100, 200 | 100 epochs generally yielded good quality, with diminishing returns beyond that. |

## Results

This section summarizes the outcomes of the training and fine-tuning experiments.

-   **WandB Logs:**
    -   Pretraining Logs: [Link to Pretraining WandB Logs](https://wandb.ai/th3elma2-enset-mohammedia/Tacotron-TTS-Pretraining)
    -   Fine-tuning Logs: [Link to Fine-tuning WandB Logs](https://wandb.ai/th3elma2-enset-mohammedia/Tacotron-TTS-FT?nw=nwuserth3elma2)

-   **Generated Audio:**
    -   Synthesized WAV files from inference can be found in the `/wav-results` directory.

-   **Visualizations:**
    -   Detailed loss plots, predicted spectrograms, and ground truth spectrograms are available within the respective WandB log pages, providing insights into model convergence and audio quality.

### 9. Run Inference

There are two primary methods for running inference:

1.  **Using the CoquiTTS CLI:**
    Execute the following command, replacing the `model_path` and `config_path` with the actual paths to your fine-tuned model and its configuration file:

    ```bash
    tts --model_path /workspace/Tacotron-TTS/ft_out/tacotron-ft-v4e-5-250e-August-21-2025_05+18PM-379b5b8/best_model_540.pth --config_path /workspace/Tacotron-TTS/ft_out/tacotron-ft-v4e-5-250e-August-21-2025_05+18PM-379b5b8/config.json --text "hello world"  --use_cuda
    ```

2.  **Using `inference.ipynb` (To be implemented):**
    An interactive inference example will be provided in `inference.ipynb` for a more detailed and step-by-step approach to generating speech.
