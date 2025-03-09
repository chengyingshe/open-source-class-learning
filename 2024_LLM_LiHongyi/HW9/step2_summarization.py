import re
import time
from openai import OpenAI
import textwrap

from opencc import OpenCC

def extract_and_save_text(srt_filename, output_filename):
    # Open the SRT file and read its content.
    with open(srt_filename, 'r', encoding='utf-8') as file:
        content = file.read()

    # Use regular expression to remove the timecode.
    pure_text = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n', '', content)

    # Remove the empty lines.
    pure_text = re.sub(r'\n\n+', '\n', pure_text)

    # Creating an instance of OpenCC for Simplified to Traditional Chinese conversion.
    cc = OpenCC('s2t')
    pure_text_conversion = cc.convert(pure_text)

    # Write the extracted text to a new file.
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        output_file.write(pure_text_conversion)

    print(f'Extracted text has been saved to {output_filename}.\n\n')

    return pure_text_conversion

def chunk_text(text, max_length):
    return textwrap.wrap(text, max_length)


# The name of the model you want to use.
model_name = "medium" # @param ["tiny", "base", "small", "medium", "large-v3"]

# Define the suffix of the output file.
suffix = "信號與人生" # @param {type: "string"}

# Path to the output file.
output_subtitle_path = f"./output-{suffix}.srt"

# Path of the output raw text file from the SRT file.
output_raw_text_path = f"./output-{suffix}.txt"

# Path to the directory where the model and dataset will be cached.
cache_dir = "./"

# The language of the lecture video.
language = "zh" # @param {type:"string"}

initial_prompt = "請用繁體中文" #@param {type:"string"}

temperature = 0 # @param {type:"slider", min:0, max:1, step:0.1}

chunk_length = 512
# Extracts the text from an SRT file and saves it to a new text file
pure_text = extract_and_save_text(srt_filename=output_subtitle_path, output_filename=output_raw_text_path)

# Split a long document into smaller chunks of a specified length
chunks = chunk_text(text=pure_text, max_length=512)

# You can see the number of words and contents in each paragraph.
print("Review the results of splitting the long text into several short texts.\n")
for index, chunk in enumerate(chunks):
    if index == 0:
        print(f"\n========== The {index + 1}-st segment of the split ({len(chunk)} words) ==========\n\n")
        for text in textwrap.wrap(chunk, 80):
            print(f"{text}\n")
    elif index == 1:
        print(f"\n========== The {index + 1}-nd segment of the split ({len(chunk)} words) ==========\n\n")
        for text in textwrap.wrap(chunk, 80):
            print(f"{text}\n")
    elif index == 2:
        print(f"\n========== The {index + 1}-rd segment of the split ({len(chunk)} words) ==========\n\n")
        for text in textwrap.wrap(chunk, 80):
            print(f"{text}\n")
    else:
        print(f"\n========== The {index + 1}-th segment of the split ({len(chunk)} words) ==========\n\n")
        for text in textwrap.wrap(chunk, 80):
            print(f"{text}\n")
# Your Deepseek API key.
deepseek_api_key = "sk-588fa9bef7544d0b8860cf3b4c9bc4d8" # @param {type:"string"}s

model_name = "deepseek-chat" # @param {type:"string"}

# @markdown **temperature**: Controls randomness in the response. Lower values make responses more deterministic.
temperature = 1 # @param {type:"slider", min:0, max:1, step:0.1}

# @markdown **top_p**: Controls diversity via nucleus sampling. Higher values lead to more diverse responses.
top_p = 1.0 # @param {type:"slider", min:0, max:1, step:0.1}


client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

def summarization(client, summarization_prompt, model_name="deepseek-chat", temperature=0.0, top_p=1.0, max_tokens=512):
    user_prompt = summarization_prompt

    while True:

        try:
            # Use the Claude API to summarize the text.
            response = client.chat.completions.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                stream=False
            )

            break

        except Exception as e:
            # If the API call fails, wait for 1 second and try again.
            print(f"Error: {e}, wait for 1s and try again.")
            time.sleep(1)

    return response.choices[0].message.content


# @markdown **max_tokens**: The maximum number of tokens to generate in the completion.
max_tokens = 350 # @param {type:"integer"}

# TODO: modify the summarization prompt and maximum number of tokens.(DO NOT modify the part of <text>)
summarization_prompt_template = "用 300 個字內寫出這段文字的摘要，其中包括要點和所有重要細節：<text>" # @param {type:"string"}

paragraph_summarizations = []

# First, we summarize each section that has been split up separately.
for index, chunk in enumerate(chunks):

    # Record the start time.
    start = time.time()

    # Construct summarization prompt.
    summarization_prompt = summarization_prompt_template.replace("<text>", chunk)
    
    # We summarize each section that has been split up separately.
    response = summarization(client=client, summarization_prompt=summarization_prompt, model_name=model_name, temperature=temperature, top_p=top_p, max_tokens=max_tokens)

    # Calculate the execution time and round it to 2 decimal places.
    cost_time = round(time.time() - start, 2)

    # Print the summary and its length.
    print(f"----------------------------Summary of Segment {index + 1}----------------------------\n")
    for text in textwrap.wrap(response, 80):
        print(f"{text}\n")
    print(f"Length of summary for segment {index + 1}: {len(response)}")
    print(f"Time taken to generate summary for segment {index + 1}: {cost_time} sec.\n")

    # Record the result.
    paragraph_summarizations.append(response)
    
# First, we collect all the summarizations obtained before and print them.

collected_summarization = ""
for index, paragraph_summarization in enumerate(paragraph_summarizations):
    collected_summarization += f"Summary of segment {index + 1}: {paragraph_summarization}\n\n"

print(collected_summarization)

max_tokens = 550 # @param {type:"integer"}

# @markdown ### Changing **summarization_prompt_template**
# @markdown You can modify the summarization prompt and maximum number of tokens. However, **DO NOT** modify the part of `<text>`.
summarization_prompt_template = "在 500 字以內寫出以下文字的簡潔摘要：<text>" # @param {type:"string"}

# Finally, we compile a final summary from the summaries of each section.

# Record the start time.
start = time.time()

summarization_prompt = summarization_prompt_template.replace("<text>", collected_summarization)

# Run final summarization.
final_summarization = summarization(client=client, summarization_prompt=summarization_prompt, model_name=model_name, temperature=temperature, top_p=top_p, max_tokens=max_tokens)

# Calculate the execution time and round it to 2 decimal places.
cost_time = round(time.time() - start, 2)

# Print the summary and its length.
print(f"----------------------------Final Summary----------------------------\n")
for text in textwrap.wrap(final_summarization, 80):
    print(f"{text}")
print(f"\nLength of final summary: {len(final_summarization)}")
print(f"Time taken to generate the final summary: {cost_time} sec.")

''' In this block, you can modify your desired output path of final summary. '''

output_path = f"./final-summary-{suffix}-deepseek-multi-stage.txt"

# If you need to convert Simplified Chinese to Traditional Chinese, please set this option to True; otherwise, set it to False.
convert_to_tradition_chinese = False

if convert_to_tradition_chinese == True:
    # Creating an instance of OpenCC for Simplified to Traditional Chinese conversion.
    cc = OpenCC('s2t')
    final_summarization = cc.convert(final_summarization)

# Output your final summary
with open(output_path, "w") as fp:
    fp.write(final_summarization)

# Show the result.
print(f"Final summary has been saved to {output_path}")
print(f"\n===== Below is the final summary ({len(final_summarization)} words) =====\n")
for text in textwrap.wrap(final_summarization, 64):
    print(text)