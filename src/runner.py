import torch

from src.generate import text_to_token_ids, generate, token_ids_to_text
from src.gpt_model import GptConfig, GPTModel
from pathlib import PurePath
import tiktoken

GPT_CONFIG_124M = GptConfig(
    vocab_size=50257,
    context_length=1024,
    embedding_dim=768,
    n_heads=12,
    n_layers=12,
    dropout_rate=0.1,
    qkv_bias=True,
)

PWD = PurePath(__file__).parent
GPT_MODEL_PATH = PWD.parent / "model.pth"

gpt = GPTModel(GPT_CONFIG_124M)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt.load_state_dict(torch.load(GPT_MODEL_PATH, map_location=device))
gpt.eval()

tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("The meaning of life is", tokenizer).to(device),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M.context_length,
    top_k=50,
    temperature=1.5,
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
