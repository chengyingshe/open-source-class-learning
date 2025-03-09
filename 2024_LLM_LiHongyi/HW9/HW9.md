# HW9

## Task

- Step 1: Automatic Speech Recognition (ASR)
    > Utilize the OpenAI Whisper model for speech recognition to get the text transcription of the video.

- Step 2: Summarization
    > Design a prompt for large language models to summarize the text into 300 to 500 words, ensuring that the summary is in Traditional Chinese (繁體中文).

## Extra

本人已经将`HW9.ipynb`文件中的内容拆分为了两个python文件`step1_asr.py`和`step2_summarization.py`，分别单独运行两个文件即可完成作业要求。`Step 2`中默认只提供了`ChatGPT`, `Gemini`, `Claude`三种模型的API调用方式，本人额外添加了`Deepseek`的API调用，可以使用免费额度或少量充值的情况下完成作业