from modelscope import snapshot_download, AutoModel, AutoTokenizer

model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm2-chat-7b', revision='master')
model_dir = snapshot_download('jinaai/jina-embeddings-v2-base-zh', revision='master')