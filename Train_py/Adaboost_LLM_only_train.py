
# START 【生成任务训练，加入训练集与验证集】
#当你安装好所有环境后，开始数据集处理与模型训练（微调）
# 2. Dataset Check 数据集

from datetime import date, datetime
import imp
import time
import traceback

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from datasets import load_dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)
import torch
import torch.nn as nn
from torch.nn import DataParallel
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoConfig
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from sklearn import metrics
import pandas as pd
import json
from pandas import  DataFrame
import numpy as np
import random
def time_shower(main):
    def call_main():
        print("\033[1;34m.....Main Start.....\033[0m")
        start = time.time()
        main()
        end = time.time()
        print("\n\033[1;34m.....Main End.....")
        print(f"-----共耗费了：{(end-start):.4f} 秒-----\033[0m")
    return call_main

#     ### Speaker:
#     {data_point["Speaker"]}
# ### Speaker:
#             {data_point["Speaker"]}

def generate_prompt(data_point):
    # create prompts from the loaded dataset and tokenize them
    if data_point["input"]:
        return f"""Please perform Classification task. Given the sentence, assign the correct label. Return label only without any other text.
                    ### Instruction:
                    {data_point["instruction"]}

                    ### Input:
                    {data_point["input"]}

                    ### Response:
                    {data_point["output"]}"""


    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

        ### Instruction:
        {data_point["instruction"]}

        ### Response:
        {data_point["output"]}"""


def generate_test_prompt(data_point):
    # create prompts from the loaded dataset and tokenize them
    if data_point["input"]:
        return f"""Please perform Classification task. Given the sentence, assign the correct label. Return label only without any other text.

        ### Instruction:
        {data_point["instruction"]}

        ### Input:
        {data_point["input"]}

        ### Response:
        """

    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

        ### Instruction:
        {data_point["instruction"]}

        ### Response:
        {data_point["output"]}"""

def post_process(task, response: str):
    response = response.strip().lower()
    response = response.split(".")[0].strip()
    response = response.split("\n")[0].strip()
    print("R_split:",response)
    labels_dict = {"sentiment": ["positive", "neutral", "negative", "pos", "neg", "neu", "0", "1"],
                    "sarcasm": ["sarcastic", "sarcasm", "irony", "satire", "ironic", "non-sarcastic", "non-ironic"],
                    "humor": ["humor", "fun", "non-humor", "non-fun", "not fun", "not humor"],
                    "offensive": ["offensive", "offend", "non-offensive"],
                    "enthusiasm": ["monotonous", "normal", "enthusiastic"]}

    synonymy_dict = {"sentiment": { ("positive", "pos", "1") : 1, ("neutral", "neu") : 0, ("negative", "neg", "0") : 2 },
                      "sarcasm": { ("sarcastic", "sarcasm", "irony", "satire", "ironic") : 1, ("non-sarcastic", "non-ironic") : 0 },
                      "humor": { ("humor", "fun", "non-humor") : 1, ("non-fun", "not fun", "not humor") : 0 },
                      "offensive": { ("offensive", "offend") : 1, ("non-offensive") : 0 },
                      "enthusiasm": { ("monotonous") : 0, ("normal") : 1,  ("enthusiastic") : 2 },
                    }


    selected_task = labels_dict.get(task)
    for i in range(len(selected_task)):
        category = selected_task[i]
        if response.find(category) != -1:
            for key, value in synonymy_dict[task].items():
                if category in key:
                    # print(category)
                    return value
                    break

            break



def generate_true_labels(task, TEST_SET):
    true_labels = []
    i=0;
    with open(TEST_SET, "r", encoding="utf-8") as f:
        for test_set in json.load(f):
            # if i>800:
            #     break
            # i = i+1
            output = test_set["output"].lower()
            label = post_process(task, output)
            true_labels.append(int(label))

    return true_labels

def eval_performance(y_true, y_pred):
    # Precision
    print("Precision:\n\t", metrics.precision_score(y_true, y_pred, average='weighted'))

    # Recall
    print("Recall:\n\t", metrics.recall_score(y_true, y_pred, average='weighted'))

    # Accuracy
    print("Accuracy:\n\t", metrics.accuracy_score(y_true, y_pred))

    print("----------F1, Micro-F1, Macro-F1, Weighted-F1..----------------")
    print("----------**********************************----------------")

    # F1 Score
    print("F1 Score:\n\t", metrics.f1_score(y_true, y_pred, average='weighted'))

    # Micro-F1 Score
    print("Micro-F1 Score:\n\t", metrics.f1_score(y_true, y_pred, average='micro'))

    # Macro-F1 Score
    print("Macro-F1 Score:\n\t", metrics.f1_score(y_true, y_pred, average='macro'))

    # Weighted-F1 Score
    print("Weighted-F1 Score:\n\t", metrics.f1_score(y_true, y_pred, average='weighted'))

    print("------------------**********************************-------------------------")
    print("-------------------**********************************-------------------------")

    # ROC AUC Score
    # print("ROC AUC:\n\t", metrics.roc_auc_score(y_true, y_pred))

    # Confusion matrix
    print("Confusion Matrix:\n\t", metrics.confusion_matrix(y_true, y_pred))



#"decapoda-research/llama-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"
def tokenize(prompt, add_eos_token=True):

    #tokenizer.pad_token = tokenizer.eos_token
    # 特殊token [EOS]的位置。它被用来指示模型当前生成的句子已经结束

    CUTOFF_LEN = 256
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
        return_tensors=None,
    )
    if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < CUTOFF_LEN
            and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt




@time_shower
def main():

     # 0---超参数
    # Setting for A100 - For RTX 3090
    #超参数
    MICRO_BATCH_SIZE = 4  # change to 4 for 3090，默认是4
    BATCH_SIZE = 60  #为了加快训练，将默认的128加大为256，容易爆显存
    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
    #GRADIENT_ACCUMULATION_STEPS = 2
    EPOCHS = 9  # paper uses 3
    LEARNING_RATE = 3e-4  # from the original paper
    CUTOFF_LEN = 256  # 256 accounts for about 96% of the data 截断长度
    LORA_R = 8  #LORA中最重要的一个超参数，用于降维，LORA的秩
    LORA_ALPHA = 64 #alpha其实是个缩放参数，本质和learning rate相同，所以为了简化默认让alpha=rank
    LORA_DROPOUT = 0.05
    TRAIN_STEPS = 300


    # 1---分词器处理
    #设置基础模型与训练数据的路径
    TRAIN_DIR = "./train_dataset/STT2_train.json"
    DEV_DIR = "./train_dataset/STT2_dev.json"
    output_dir = "Adaboost_STT2_19"
    # BASE_MODEL = "decapoda-research/llama-7b-hf"
    BASE_MODEL = "meta-llama/Llama-2-7b-hf"


    # 2---开始进行数据加载与处理，可以替换为自己的数据集，前提是规范格式
    data_train = load_dataset("json", data_files=TRAIN_DIR)
    data_dev = load_dataset("json",data_files=DEV_DIR)
    #打乱数据，设置训练集,截断、最大长度是256，可以包容96的数据，也可以更大设置为512，但是会慢
    # train_val = data["train"].train_test_split(
    #     test_size=0, shuffle=True, seed=42
    # )
    train_data = (
        data_train["train"].shuffle().map(generate_and_tokenize_prompt)
    )
    val_data = (
        data_dev["train"].map(generate_and_tokenize_prompt)
    )
    print("-----------------------------------------")
    print("The dataset has been loaded.......Next load model and train model...")
    print("-----------------------------------------")


    # 3---开始模型与lora合并
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

    # LlamaForCausalLM，可用于文本生成;  LlamaForSequenceClassification，可用于文本分类
    #模型加载自huggingface，load_in_8bit=True将模型权重和激活值量化为8位整数,从而减少内存和计算开销；device_map模型多卡并行
    model = LlamaForCausalLM.from_pretrained(BASE_MODEL, load_in_8bit=True, device_map=device_map,)
    #prepare_model_for_int8_training 对在Lora微调中使用LLM.int8()进行了适配
    model = prepare_model_for_int8_training(model)

    #设置一下LORA的config，调用一下get_peft_model方法，就获得了在原模型基础上的PEFT模型
    config = LoraConfig(r=LORA_R,
                        lora_alpha=LORA_ALPHA,
                        target_modules=["q_proj", "v_proj"],
                        lora_dropout=LORA_DROPOUT,
                        bias="none",
                        task_type="CAUSAL_LM",)
    model = get_peft_model(model, config) #获得了在原模型基础上的PEFT模型
    print("peft...")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params


    # 4---多GPU模型训练


    print("now set the multi-GPU here...") #并行8个卡

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    #eval_dataset=val_data,eval_steps=20,load_best_model_at_end = True,
    trainer = transformers.Trainer(
        model=model,
        # train_dataset=data["train"], #已经建立好的训练集，如果没有需要自己提前分好训练集和测试集
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=100,
            #max_steps=TRAIN_STEPS,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=50,
            save_steps=50,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            ddp_find_unused_parameters=False if ddp else None,
            fp16=True,
            logging_steps=1,
            load_best_model_at_end = True,
            output_dir=output_dir, #模型训练完的输出路径
            save_total_limit=3,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
             tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
         )
        #DataCollatorForLanguageModeling 实现了一个对文本数据进行随机 mask 的data collator
        # data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False), #构建语言模型或者说是进行MLM任务时需要使用的数据收集器，该数据收集器会以一定概率（由参数mlm_probability控制）将序列中的Token替换成Mask标签

    )
    model.config.use_cache = False
    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))

    # model = torch.compile(model)
    print("Now training...")
    trainer.train(resume_from_checkpoint=False)	#resume_from_checkpoint 是否断点续训，开始训练
    print("....Model Has Been Trained....., Now, Save The Model....")
    model.save_pretrained(output_dir) #保存:它只包含 2 个文件: adapter_config.json 和 adapter_model.bin是保存经过训练的增量 PEFT 权重

    # TEST_SET = "./train_dataset/SST2_test.json"
    # TASK = "sentiment"
    # common = []
    # pred_labels = []
    # true_labels = generate_true_labels(TASK, TEST_SET)  # 先将测试集的 标签 数字化，方便后续直接评测计算。
    # output_csv = "./mid_result/SST2_1.csv"
    #  # 1---分词器处理

    #  # 加载LLAMA的分词器, from_pretrained:加载预训练模型
    # #tokenizer = LlamaTokenizer.from_pretrained(
    # #     BASE_MODEL)
    #  # 模型加载自huggingface，load_in_8bit=True将模型权重和激活值量化为8位整数,从而减少内存和计算开销；device_map模型多卡并行
    # model = LlamaForCausalLM.from_pretrained(
    #      BASE_MODEL, load_in_8bit=True, device_map="auto", )

    #  # 读取保存peft模型及相关配置，使用PeftModel.from_pretrained(model, peft_model_id)方法加载模型
    # model = PeftModel.from_pretrained(
    #      model, output_dir, adapter_name="my_alpaca")
    #  # 2---整个测试集测试
    #  # 如果是多个测试样本，可以参考如下：
    # with open(TEST_SET, "r", encoding="utf-8") as f:
    #      test_set = json.load(f)
    #      for i in range(len(test_set)):
    #          # if i > 800:
    #          #     break

    #          sample = test_set[i]
    #          # print(sample)
    #          PROMPT = generate_test_prompt(sample)
    #          inputs = tokenizer(PROMPT, return_tensors="pt", )
    #          input_ids = inputs["input_ids"].cuda()  # 将输入张量移动到 GPU 上
    #          # top_p 已知生成各个词的总概率是1（即默认是1.0）如果top_p小于1，则从高到低累加直到top_p，取这前N个词作为候选; repetition_penalty 重复处罚的参数
    #          generation_config = GenerationConfig(
    #              temperature=0.7,
    #              top_p=0.95,
    #              repetition_penalty=1.15,
    #              top_k=40,
    #              num_beams=4,
    #          )
    #          # temperature=0.1, top_p=0.95, repetition_penalty=1.15,
    #          print("Now we are Generating...\n")

    #          # 3--- 当处理生成任务时候，使用model = LlamaForCausalLM.from_pretrained。 生成任务而不是分类---以后情感对话生成！
    #          generation_output = model.generate(input_ids=input_ids, generation_config=generation_config,
    #                                             return_dict_in_generate=True, output_scores=True, max_new_tokens=256, )

    #          s = generation_output.sequences[0]
    #          output = tokenizer.decode(s)
    #          # output = output[2:output.find('"')]
    #          # print("output:", output)
    #          if "### Response:" in output:
    #              response = output.split("### Response:")[1].strip()
    #          else:
    #              response = "Neutral"
    #          print("Response:", response)
    #          print("-----------------------------------------")
    #          label = post_process(TASK, response)
    #          random_list = [0, 1]
    #          if (label == None):
    #              label = random.choice(random_list)
    #          pred_labels.append(label)
    #          if(true_labels[i]==pred_labels[i]):
    #              common.append("equal")
    #          else:
    #              common.append("no")
    #          print(label)
    #          print("***")
    #          print(i)
    #          print("..............Now Post-processing Respnese and Generating Labels..............")

    #  # 将标签存入csv
    # true_array = np.array(true_labels)[:, np.newaxis]
    # pre_array = np.array(pred_labels)[:, np.newaxis]
    # common_array = np.array(common)[:,np.newaxis]
    # concatenate_array = np.concatenate((true_array, pre_array,common_array), axis=1)
    #  # print("concatenate_array", concatenate_array)
    #  # print("concatenate_array", concatenate_array.shape)
    # data = DataFrame(concatenate_array, columns=["true_label", "pre_label", "common"])
    # data.to_csv(output_csv)

    #  # 4----开始模型评测计算
    # print(".........Now Computing P, R, F1 Scores..........")

    # eval_performance(true_labels, pred_labels)

    # 4---未来可以考虑将自己的模型上传
    #后期可以将自己训练好的模型上传到huggingface上，步骤是：
    # from huggingface_hub import notebook_login

    # notebook_login()

    # model.push_to_hub("affectLLM/Sentiment", use_auth_token=True) #这是我的账号




if __name__ == '__main__':
    main()
