"""Frozen content encoder with preparation-compatible, differentiable input."""
import torch
from torch import nn
import torch.nn.functional as F
from scipy.signal import firwin


def resample_24_to_16(wave):
    """Match scipy.signal.resample_poly(up=2, down=3), including its delay."""
    if wave.ndim!=1 or not wave.numel() or not torch.isfinite(wave).all():
        raise ValueError('resampling requires finite nonempty mono audio')
    value=wave.float()
    # SciPy's default filter: half_len=10*max(up,down), Kaiser beta=5.
    kernel=torch.tensor(firwin(61,1/3,window=('kaiser',5.)),device=value.device,dtype=value.dtype)*2
    kernel=F.pad(kernel,(3,0))
    upsampled=F.pad(value[:,None],(0,1)).reshape(1,1,-1)
    filtered=F.conv1d(upsampled,kernel.flip(0)[None,None],stride=3,padding=len(kernel)-1).flatten()
    return filtered[11:11+(2*len(wave)+2)//3]


class FrozenCTC(nn.Module):
    """The pinned XLSR clock; no model gradients, no no_grad on the waveform."""
    def __init__(self,model):
        super().__init__()
        if (list(model.config.conv_kernel)!=[10,3,3,3,3,2,2]
                or list(model.config.conv_stride)!=[5,2,2,2,2,2,2]
                or any(p.dtype!=torch.float32 for p in model.parameters())):
            raise ValueError('content encoder differs from the pinned float32 XLSR clock')
        self.model=model.eval().requires_grad_(False)
        self.provenance=dict(kind='haru-frozen-content-v1',resample='scipy-polyphase-2-3-torch-v1',
            normalization='population-variance-eps1e-7',attention_mask='all-valid',
            model_config=model.config.to_dict())
        self.eval()

    def forward(self,wave):
        if self.training or self.model.training or any(p.requires_grad for p in self.parameters()):
            raise ValueError('content encoder must stay frozen and in eval mode')
        value=resample_24_to_16(wave)
        if len(value)<400:raise ValueError('content input is shorter than the receptive field')
        value=(value-value.mean())/torch.sqrt(value.var(unbiased=False)+1e-7)
        logits=self.model(input_values=value[None],attention_mask=torch.ones_like(value[None],dtype=torch.long)).logits[0]
        if logits.ndim!=2 or len(logits)!=(len(value)-400)//320+1 or not torch.isfinite(logits).all():
            raise ValueError('content encoder output differs from the expected clock')
        return logits
