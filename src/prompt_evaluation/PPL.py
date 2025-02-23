import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 加载预训练的 GPT-2 模型和 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 评估模式，不需要梯度计算
model.eval()


def calculate_ppl(sentence):
    # 将句子编码为 GPT-2 的输入格式
    inputs = tokenizer(sentence, return_tensors="pt")

    # 使用模型预测
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])

    # 计算损失（交叉熵损失）
    loss = outputs.loss

    # 使用损失值计算 perplexity
    ppl = torch.exp(loss)
    return ppl.item()


# load csv file
# 
# 测试句子
sentence = "The quick brown fox jumps over the lazy dog."
ppl = calculate_ppl(sentence)
print(f"Sentence: {sentence}")
print(f"Perplexity (PPL): {ppl}")
