# Import packages.
# import os
# os.environ['HF_HUB_OFFLINE'] = '1'  # Set offline mode

import whisper
import srt
import datetime
import time
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

# Load dataset.
dataset_name = "kuanhuggingface/NTU-GenAI-2024-HW9"
dataset = load_dataset(dataset_name)

# Prepare audio.
input_audio = dataset["test"]["audio"][0]
input_audio_name = input_audio["path"]
input_audio_array = input_audio["array"].astype(np.float32)
sampling_rate = input_audio["sampling_rate"]

print(f"Now, we are going to transcribe the audio: 李琳山教授 信號與人生 (2023) ({input_audio_name}).")

def speech_recognition(model_name, input_audio, output_subtitle_path, decode_options, cache_dir="./"):
    '''
        (1) Objective:
            - This function aims to convert audio to subtitle.

        (2) Arguments:

            - model_name (str):
                The name of the model. There are five model sizes, including tiny, base, small, medium, large-v3.
                For example, you can use 'tiny', 'base', 'small', 'medium', 'large-v3' to specify the model name.
                You can see 'https://github.com/openai/whisper' for more details.

            - input_audio (Union[str, np.ndarray, torch.Tensor]):
                The path to the audio file to open, or the audio waveform
                - For example, if your input audio path is 'input.wav', you can use 'input.wav' to specify the input audio path.
                - For example, if your input audio array is 'audio_array', you can use 'audio_array' to specify the input audio array.

            - output_subtitle_path (str):
                The path of the output subtitle file.
                For example, if you want to save the subtitle file as 'output.srt', you can use 'output.srt' to specify the output subtitle path.

            - decode_options (dict):
                The options for decoding the audio file, including 'initial_prompt', 'prompt', 'prefix', 'temperature'.
                - initial_prompt (str):
                    Optional text to provide as a prompt for the first window. This can be used to provide, or
                    "prompt-engineer" a context for transcription, e.g. custom vocabularies or proper nouns
                    to make it more likely to predict those word correctly.
                    Default: None.

                You can see "https://github.com/openai/whisper/blob/main/whisper/decoding.py" and "https://github.com/openai/whisper/blob/main/whisper/transcribe.py"
                for more details.

                - temperature (float):
                    The temperature for sampling from the model. Higher values mean more randomness.
                    Default: 0.0

            - cache_dir (str):
                The path of the cache directory for saving the model.
                For example, if you want to save the cache files in 'cache' directory, you can use 'cache' to specify the cache directory.
                Default: './'

        (3) Example:

            - If you want to use the 'base' model to convert 'input.wav' to 'output.srt' and save the cache files in 'cache' directory,
            you can call this function as follows:

                speech_recognition(model_name='base', input_audio_path='input.wav', output_subtitle_path='output.srt', cache_dir='cache')
    '''

    # Record the start time.
    start_time = time.time()

    print(f"=============== Loading Whisper-{model_name} ===============")

    # Load the model.
    model = whisper.load_model(name=model_name, download_root=cache_dir)

    print(f"Begin to utilize Whisper-{model_name} to transcribe the audio.")

    # Transcribe the audio.
    transcription = model.transcribe(audio=input_audio, language=decode_options["language"], verbose=False,
                                     initial_prompt=decode_options["initial_prompt"], temperature=decode_options["temperature"])

    # Record the end time.
    end_time = time.time()

    print(f"The process of speech recognition costs {end_time - start_time} seconds.")

    subtitles = []
    # Convert the transcription to subtitle and iterate over the segments.
    for i, segment in tqdm(enumerate(transcription["segments"])):

        # Convert the start time to subtitle format.
        start_time = datetime.timedelta(seconds=segment["start"])

        # Convert the end time to subtitle format.
        end_time = datetime.timedelta(seconds=segment["end"])

        # Get the subtitle text.
        text = segment["text"]

        # Append the subtitle to the subtitle list.
        subtitles.append(srt.Subtitle(index=i, start=start_time, end=end_time, content=text))

    # Convert the subtitle list to subtitle content.
    srt_content = srt.compose(subtitles)

    print(f"\n=============== Saving the subtitle to {output_subtitle_path} ===============")

    # Save the subtitle content to the subtitle file.
    with open(output_subtitle_path, "w", encoding="utf-8") as file:
        file.write(srt_content)
        
# @title Parameter Setting of Whisper { run: "auto" }

''' In this block, you can modify your desired parameters and the path of input file. '''

# The name of the model you want to use.
# For example, you can use 'tiny', 'base', 'small', 'medium', 'large-v3' to specify the model name.
# @markdown **model_name**: The name of the model you want to use.
model_name = "medium" # @param ["tiny", "base", "small", "medium", "large-v3"]

# Define the suffix of the output file.
# @markdown **suffix**: The output file name is "output-{suffix}.* ", where .* is the file extention (.txt or .srt)
suffix = "信號與人生" # @param {type: "string"}

# Path to the output file.
output_subtitle_path = f"./output-{suffix}.srt"

# Path of the output raw text file from the SRT file.
output_raw_text_path = f"./output-{suffix}.txt"

# Path to the directory where the model and dataset will be cached.
cache_dir = "./"

# The language of the lecture video.
# @markdown **language**: The language of the lecture video.
language = "zh" # @param {type:"string"}

# Optional text to provide as a prompt for the first window.
# @markdown **initial_prompt**: Optional text to provide as a prompt for the first window.
initial_prompt = "請用繁體中文" #@param {type:"string"}

# The temperature for sampling from the model. Higher values mean more randomness.
# @markdown  **temperature**: The temperature for sampling from the model. Higher values mean more randomness.
temperature = 0 # @param {type:"slider", min:0, max:1, step:0.1}

# Construct DecodingOptions
decode_options = {
    "language": language,
    "initial_prompt": initial_prompt,
    "temperature": temperature
}
# print message.
message = "Transcribe 李琳山教授 信號與人生 (2023)"
print(f"Setting: (1) Model: whisper-{model_name} (2) Language: {language} (2) Initial Prompt: {initial_prompt} (3) Temperature: {temperature}")
print(message)

# Running ASR.
speech_recognition(model_name=model_name, input_audio=input_audio_array, output_subtitle_path=output_subtitle_path, decode_options=decode_options, cache_dir=cache_dir)

''' Open the SRT file and read its content.
The format of SRT is:

[Index]
[Begin time] (hour:minute:second) --> [End time] (hour:minute:second)
[Transcription]

'''

with open(output_subtitle_path, 'r', encoding='utf-8') as file:
    content = file.read()

print(content)
