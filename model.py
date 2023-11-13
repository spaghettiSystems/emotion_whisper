from transformers import WhisperProcessor, WhisperForConditionalGeneration

import torch
import torch.nn as nn
import torch.nn.functional as F

class WhisperEncoderWithHead(nn.Module):
    def __init__(self, model_name="openai/whisper-medium", num_labels=8, ):
        super(WhisperEncoderWithHead, self).__init__()
        

        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")#.to(torch.bfloat16)
        self.encoder = model.get_encoder()

        self.head = nn.Linear(1024, 8)

        self.logit_scaling = nn.Parameter(torch.tensor([[0.19170297, 0.09759999, 0.05292019, 0.06359768, 0.06397428, 0.11632104, 0.02482805, 0.3890558 ]]), requires_grad=False)

    def forward(self, x,):
        hidden = self.encoder(x).last_hidden_state
        return self.head(hidden)