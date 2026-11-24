"""实测 train_tokenizer.py：用项目自带的 README 造语料，完整走一遍训练 + 评测。

运行：
    D:\\conda_envs\\swift_env\\python.exe train_tokenizer_probe.py
"""
import json
import os
import sys

PROJECT = r"E:\learnnotes-vincent\notes\scrip\999-minimind-解读\minimind-master"
WORK = r"E:\learnnotes-vincent\notes\scrip\999-minimind-解读\_tmp_tok"
sys.path.insert(0, PROJECT)
sys.stdout.reconfigure(encoding="utf-8")

from trainer.train_tokenizer import train_tokenizer, eval_tokenizer, get_texts


def build_corpus(path: str) -> int:
    """把 README 的段落拼成交替的两种格式，顺便测 get_texts 的两条分支。"""
    texts = []
    for name in ("README.md", "README_en.md"):
        raw = open(os.path.join(PROJECT, name), encoding="utf-8", errors="ignore").read()
        for block in raw.split("\n\n"):
            b = block.strip()
            if len(b) >= 20:
                texts.append(b)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, t in enumerate(texts):
            if i % 5 == 0:      # 20% 走 conversations 分支
                f.write(json.dumps({
                    "conversations": [
                        {"role": "user", "content": t[:100]},
                        {"role": "assistant", "content": t},
                    ]
                }, ensure_ascii=False) + "\n")
            else:               # 80% 走 text 分支
                f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    return len(texts)


def main() -> None:
    data = os.path.join(WORK, "corpus.jsonl")
    out = os.path.join(WORK, "tok")
    n = build_corpus(data)
    print(f"语料段落数: {n}")
    print(f"get_texts 产出条数: {sum(1 for _ in get_texts(data))}")

    print("\n=== 训练 BPE ===")
    train_tokenizer(data, out, 6400)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(out)

    print("\n=== 词表信息 ===")
    print(f"len(tokenizer) = {len(tok)}")
    print(f"bos={tok.bos_token}({tok.bos_token_id})  eos={tok.eos_token}({tok.eos_token_id})  "
          f"pad={tok.pad_token}({tok.pad_token_id})  unk={tok.unk_token}({tok.unk_token_id})")
    print(f"vocab_size 属性 = {tok.vocab_size}")

    print("\n=== 特殊/非特殊标记 ===")
    for t in ["<|im_end|>", "<think>", "<tool_call>", "<|buffer1|>"]:
        i = tok.convert_tokens_to_ids(t)
        print(f"  {t:16s} id={i:<6} decode后保留={t in tok.decode([i], skip_special_tokens=True)}")

    print("\n=== 中文分词与解码 ===")
    s = "人工智能是计算机科学的一个分支"
    enc = tok(s)["input_ids"]
    print(f"  {s}  ({len(s)} 字) -> {len(enc)} tokens")
    print(f"  ids: {enc}")
    text = "<|im_start|>user\n你好<|im_end|>\n<think>\n想一想\n</think>\n你好呀"
    ids = tok(text)["input_ids"]
    print("  skip_special_tokens=False:", repr(tok.decode(ids, skip_special_tokens=False)))
    print("  skip_special_tokens=True :", repr(tok.decode(ids, skip_special_tokens=True)))

    print("\n=== 官方 eval_tokenizer 的输出 ===")
    eval_tokenizer(out)


if __name__ == "__main__":
    main()
