from tokenizers import Tokenizer

from src.config import DataConfig
from src.data import get_dataloaders


def test_openwebtext_gpt2_pretokenized_loading():
    cfg = DataConfig()
    batch_size = 4

    train_loader, val_loader = get_dataloaders(cfg, batch_size, validate=True)
    train_batch = next(train_loader)
    val_batch = next(val_loader)
    print(train_batch.shape, val_batch.shape)

    tokenizer = Tokenizer.from_file(
        "/home/jsingh/projects/fastlm/tokenizer/better-gpt2/tokenizer.json"
    )
    print(tokenizer.get_vocab_size())


if __name__ == "__main__":
    test_openwebtext_gpt2_pretokenized_loading()
