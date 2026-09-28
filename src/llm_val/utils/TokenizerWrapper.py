import torch
from transformers import AutoTokenizer


class TokenizerWrapper:
    def __init__(self, model_path: str, allow_gpu: bool = True):
        self.allow_gpu = allow_gpu
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.__init_device()

    def __init_device(self):
        use_gpu = torch.cuda.is_available() and self.allow_gpu
        self.device = ('cpu', 'cuda')[use_gpu]

    def decode(self,
               batch,
               config):
        return self.tokenizer.batch_decode(batch.cpu().tolist(), **config)

    def encode(self,
               batch,
               config):
        return self.tokenizer.batch_encode_plus(batch,
                                                **config).to(self.device)
