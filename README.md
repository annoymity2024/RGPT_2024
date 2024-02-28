# RGPT_2024
RGPT is based on LLaMA and generates a specialized text classification LLM by cyclically integrating a set of strong basic learners. The base learner is built by adjusting the distribution of training samples and using them to iteratively fine-tune the LLM.



# Requirements

 - `git clone https://github.com/facebookresearch/llama.git`
 - `bash download.sh`
 - Installation package：`pip install accelerate`
 - `pip install appdirs`
 - `pip install bitsandbytes`
 - `pip install datasets`
 - `pip install fire`
 - `pip install git+https://github.com/huggingface/peft.git`
 - `pip install git+https://github.com/huggingface/transformers.git`
 - `pip install torch`
 - `pip install sentencepiece`
 - `pip install tensorboardX`
 - `pip install gradio`
 - If you don't have enough memory, you need to convert the program：`wget https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/llama/convert_llama_weights_to_hf.py`
 - `python convert_llama_weights_to_hf.py --input_dir ./llama/llama-2-7b --model_size 7B --output_dir ./llama/models_hf/llama-2-7b`
