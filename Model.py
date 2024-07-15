from torch import nn
from torch.nn import functional as F
import torch

from FeatureEncoding import FeatureEncoding
from Attention import MultiHeadAttention, Transformer

class MLPMulti(nn.Module):
    def __init__(self):
        super(MLPMulti, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(768 * 3, 200),
            nn.BatchNorm1d(200),
            nn.Dropout(0.3),
            
            nn.Linear(200, 40),
            nn.BatchNorm1d(40),
            nn.Dropout(0.3),

            nn.Linear(40, 20),
            nn.Linear(20, 2)
        )

    def forward(self, tensor):
        seq = self.mlp(tensor)
        output = F.softmax(seq, dim = -1)

        return output
    

class HICCAP(nn.Module):
    def __init__(self, dim_inp:int, dim_out:int, task:str, num_heads:int = 1, encoder:bool = False, num_layers:int = 1, prunning:bool = False, gmu:bool = False,\
                 parallel_ca:bool = False, dtype_parallel = 'Concat'):
        super(HICCAP, self).__init__()

        # Various
        self.task = task.lower()

        # Encoding for modalities
        self.encoding = FeatureEncoding()
        
        # Second Block
        if not encoder:
            # For the original model set num_heads to 1 and parallel_ca to False, the rest bool values to False
            self.txt = MultiHeadAttention(dim_inp, dim_out, num_heads, prunning, gmu, parallel_ca, dtype_parallel)
            self.img = MultiHeadAttention(dim_inp, dim_out, num_heads, prunning, gmu, parallel_ca, dtype_parallel)
            self.aud = MultiHeadAttention(dim_inp, dim_out, num_heads, prunning, gmu, parallel_ca, dtype_parallel)
        else:
            self.txt = Transformer(dim_inp, dim_out, num_heads, num_layers, prunning, gmu, parallel_ca, dtype_parallel)
            self.img = Transformer(dim_inp, dim_out, num_heads, num_layers, prunning, gmu, parallel_ca, dtype_parallel)
            self.aud = Transformer(dim_inp, dim_out, num_heads, num_layers, prunning, gmu, parallel_ca, dtype_parallel)

        # Classification Block
        if task.lower() == 'binary':
            self.mlp = MLPMulti()
        elif task.lower() == 'multi':
            self.mlps = nn.ModuleList(
                MLPMulti() for _ in range(4)
            )

    def forward(self, text, text_mask, img, img_mask, aud, aud_mask):

        # Feature Encoding
        txt_encoded, txt_msk, img_encoded, img_msk, aud_encoded, aud_msk = self.encoding(text, text_mask, img, img_mask, aud, aud_mask)

        # Contextualized Modals
        output_txt = self.txt(txt_encoded, txt_msk,
                              img_encoded, img_mask,
                              aud_encoded, aud_msk)
        output_img = self.img(img_encoded, img_msk,
                              aud_encoded, aud_msk,
                              txt_encoded, txt_msk)
        output_aud = self.aud(aud_encoded, aud_msk,
                              img_encoded, img_msk,
                              txt_encoded, txt_msk)
        
        # Final Representation
        txt_img_aud = torch.cat([output_txt, output_img, output_aud], dim = -1)

        # Classification
        if self.task.lower() == 'binary':
            out = self.mlp(txt_img_aud)
        else:
            out = [mlp(txt_img_aud) for mlp in self.mlps]

        return out

if __name__=='__main__':
    mlp = MLPMulti()

    ten1 = torch.rand(16, 768*3)
    ten2 = mlp(ten1)
    print(ten2.shape)