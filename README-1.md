# Trump TTS Project

## Project Overview

This project aims to create a Text-to-Speech (TTS) model capable of generating audio in the voice of Donald Trump. It leverages the Coqui TTS library, specifically fine-tuning a Tacotron2 model on a dataset of Donald Trump's speeches. The goal is to provide a robust and customizable TTS solution for generating realistic-sounding audio.

## Thought Process and High-Level Approach

The primary thought process behind this project was to leverage existing, well-established TTS architectures for fine-tuning, rather than building a model from scratch. This approach significantly reduces development time and allows for faster iteration.

### High-Level Steps:

1.  **Data Acquisition and Preprocessing:** Collect a diverse dataset of Trump's speeches (specifically from Kaggle: [etaifour/trump-speeches-audio-and-word-transcription](https://www.kaggle.com/datasets/etaifour/trump-speeches-audio-and-word-transcription)) and preprocess the audio and text to be compatible with the Coqui TTS format (LJSpeech-style).
2.  **Configuration Management:** Centralize hyper-parameters and model configurations in a YAML file (`config.yaml`) for easy modification and experimentation.
3.  **Model Fine-tuning:** Utilize the `trainer.py` script to fine-tune a pre-trained Tacotron2 model on the prepared dataset.
4.  **Inference and Evaluation:** After training, use the fine-tuned model to generate new speech and evaluate its quality.

## Architectural Choices

-   **Tacotron2 Model:** Tacotron2 was chosen as the primary TTS model due to its widespread adoption, strong performance in generating natural-sounding speech, and its compatibility with the Coqui TTS framework. It's an attention-based sequence-to-sequence model that predicts a mel-spectrogram from input text, which is then converted into audio by a vocoder.

-   **Coqui TTS Library:** The Coqui TTS library provides a comprehensive and flexible framework for building and training TTS models. It offers pre-trained models, utility functions for data processing, and a well-defined training pipeline, which significantly accelerates development.

-   **YAML for Configuration:** Using a `config.yaml` file for hyper-parameters and other configurations enhances modularity and readability. It allows for easy adjustment of training parameters without modifying the code directly, facilitating experimentation and reproducibility.

-   **Data Preprocessing Notebook (`data_prep.ipynb`):** A Jupyter notebook is used for data preprocessing to provide an interactive and step-by-step approach to cleaning, segmenting, and formatting the audio-text pairs. This makes the data preparation process transparent and easier to debug.
    -   **`prerocessing.py`:** This script contains the `DataPreprocessing` class, which handles the core logic for audio segmentation, text normalization, and generating the `metadata.csv` file required by the Coqui TTS training pipeline. It reads raw audio and JSON transcription files, segments audio into shorter clips based on word timings, normalizes the accompanying text, and saves the processed audio and metadata. This ensures the data is in the correct format and meets the duration requirements for effective model training.

    ### Detailed Data Preprocessing Steps

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

-   **Data Quality and Quantity:** Obtaining a clean and sufficiently large dataset of Trump's speeches with accurate transcriptions was a primary challenge. Solutions involved using publicly available datasets and careful manual verification/correction of transcriptions.

-   **Computational Resources:** Training large TTS models like Tacotron2 can be computationally intensive. The solution involves leveraging GPU resources and optimizing batch sizes and other training parameters to make efficient use of available hardware.

-   **Environment Setup:** Ensuring all dependencies and the Coqui TTS library are correctly installed can be tricky due to potential conflicts. Providing clear `pip install` instructions and recommending a virtual environment helps mitigate this.

-   **Fine-tuning Convergence:** Achieving good convergence and natural-sounding speech during fine-tuning can be challenging. Experimentation with learning rates, optimizer choices, and monitoring loss curves are crucial steps. The use of WandB for logging helps in tracking and comparing experiments.

## Setup and Running Instructions

Follow these steps to set up the environment and run the project:

### 1. Clone the Repository

```bash
git clone <repository_url>
cd trump_tts
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

### 3. Install Dependencies

```bash
uv pip install -r requirements.txt
# Or, if you need all Coqui TTS dependencies:
uv pip install 'TTS[all]' pyyaml python-dotenv
```

To generate `requirements.txt` if it doesn't exist, you can run:
```bash
uv pip freeze > requirements.txt
```

### 4. Prepare Data

Run the `data_prep.ipynb` Jupyter notebook to preprocess the audio and text data. This notebook will guide you through the steps to create the `metadata.csv` and processed audio files in the `./processed_data` directory.

```bash
jupyter notebook data_prep.ipynb
```

### 5. Configure Hyper-parameters

Review and adjust the hyper-parameters in `config.yaml` as needed.

### 6. Train the Model

Start the training process by running the `trainer.py` script:

```bash
python trainer.py
```

### 7. Run Inference (To be implemented)

Instructions for running inference and generating new audio will be added here once the inference script is developed.
