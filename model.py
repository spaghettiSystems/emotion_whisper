from transformers import WhisperProcessor, WhisperForConditionalGeneration
from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class WhisperEncoderWithHead_old(nn.Module):
    def __init__(self, model_name="openai/whisper-medium", num_labels=8, ):
        super(WhisperEncoderWithHead, self).__init__()
        

        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")#.to(torch.bfloat16)
        self.encoder = model.get_encoder()

        self.head = nn.Linear(1024, 8)

        self.logit_scaling = nn.Parameter(torch.tensor([[0.19170297, 0.09759999, 0.05292019, 0.06359768, 0.06397428, 0.11632104, 0.02482805, 0.3890558 ]]), requires_grad=False)

    def forward(self, x,):
        hidden = self.encoder(x).last_hidden_state
        return self.head(hidden)
    
    
class MRL_Linear_Layer(nn.Module):
	def __init__(self, nesting_list: List, num_classes=1000, efficient=False, **kwargs):
		super(MRL_Linear_Layer, self).__init__()
		self.nesting_list = nesting_list
		self.num_classes = num_classes # Number of classes for classification
		self.efficient = efficient
		if self.efficient:
			setattr(self, f"nesting_classifier_{0}", nn.Linear(nesting_list[-1], self.num_classes, **kwargs))		
		else:	
			for i, num_feat in enumerate(self.nesting_list):
				setattr(self, f"nesting_classifier_{i}", nn.Linear(num_feat, self.num_classes, **kwargs))	

	def reset_parameters(self):
		if self.efficient:
			self.nesting_classifier_0.reset_parameters()
		else:
			for i in range(len(self.nesting_list)):
				getattr(self, f"nesting_classifier_{i}").reset_parameters()


	def forward(self, x):
		nesting_logits = ()
		for i, num_feat in enumerate(self.nesting_list):
			if self.efficient:
				if self.nesting_classifier_0.bias is None:
					nesting_logits += (torch.matmul(x[:,:, :num_feat], (self.nesting_classifier_0.weight[:, :num_feat]).t()), )
				else:
					nesting_logits += (torch.matmul(x[:,:, :num_feat], (self.nesting_classifier_0.weight[:, :num_feat]).t()) + self.nesting_classifier_0.bias, )
			else:
				nesting_logits +=  (getattr(self, f"nesting_classifier_{i}")(x[:, :, :num_feat]),)

		return nesting_logits

    
class WhisperEncoderWithHead(nn.Module):
    def __init__(self, model_name="openai/whisper-medium", do_logit_scaling=False, ):
        super(WhisperEncoderWithHead, self).__init__()
        self.model_name = model_name
        self.do_logit_scaling = do_logit_scaling
        self.logit_scaling = nn.Parameter(torch.tensor([[0.19170297, 0.09759999, 0.05292019, 0.06359768, 0.06397428, 0.11632104, 0.02482805, 0.3890558 ]]), requires_grad=False)
        
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.encoder = model.get_encoder()

        self.head = MRL_Linear_Layer([8, 16, 32, 64, 128, 256, 512, 1024], num_classes=8, efficient=False)

        self.global_head = MRL_Linear_Layer([8, 16, 32, 64, 128, 256, 512, 1024], num_classes=8, efficient=True)
        

    def forward(self, x):
        enc_output = self.encoder(x).last_hidden_state

        return self.head(enc_output)[-1] - (torch.log(self.logit_scaling**1 + 1e-12) if self.do_logit_scaling else 0)
    
