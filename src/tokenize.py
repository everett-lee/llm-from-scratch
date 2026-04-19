from pathlib import PurePath
import tiktoken
import re

p = PurePath(__file__).parent.parent


class SimpleTokenizerV2:
    def __init__(self, raw_text: str):
        preprocessed = self.process_text(raw_text)
        self.all_tokens = sorted(set(preprocessed))
        self.all_tokens.extend(["<|endoftext|>", "<|unk|>"])
        self.str_to_int: dict[str, int] = {
            token: integer for integer, token in enumerate(self.all_tokens)
        }
        self.int_to_str: dict[int, str] = {i: s for s, i in self.str_to_int.items()}

    def process_text(self, text: str) -> list[str]:
        processed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        processed = [item.strip() for item in processed if item.strip()]
        return processed

    def encode(self, text: str) -> list[int]:
        processed = self.process_text(text)
        processed = [
            item if item in self.str_to_int else "<|unk|>" for item in processed
        ]
        ids = [self.str_to_int[s] for s in processed]
        return ids

    def decode(self, ids: list[int]) -> str:
        text = " ".join([self.int_to_str[i] for i in ids])
        # Remove spaces before these punctuation symbols
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text


with open(f"{p}/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read().strip()
    print("Total number of character:", len(raw_text))

tokenizer = tiktoken.get_encoding("gpt2")
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))
enc_sample = enc_text[50:]
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1 : context_size + 1]
print(f"x: {x}")
print(f"y: {y}")
