# -*- coding: utf-8 -*

from modelscope import snapshot_download

model_dir = snapshot_download('LLM-Research/Meta-Llama-3-8B-Instruct', cache_dir='/opt/models/')
