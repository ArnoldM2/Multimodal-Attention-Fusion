import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

run_on = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(run_on)

##### ========== GMU ========== #####
class GMU(nn.Module):
    def __init__(self, dim, no_modals = 3, return_zs = False):
        super(GMU, self).__init__()

        self.dim = dim
        self.modals = no_modals
        self.return_zs = return_zs

        self.Ws = [nn.Parameter(torch.Tensor(self.dim, self.dim), requires_grad =  True).to(device) for i in range(self.modals)]
        self.Wzs = [nn.Parameter(torch.Tensor(self.dim * self.modals, self.dim), requires_grad =  True).to(device) for j in range(self.modals)]

        self.normalize_weights(mean=0.0, std=0.05)

    def normalize_weights(self, mean=0.0, std=0.05):
        for w, wz in zip(self.Ws, self.Wzs):
            w.data.normal_(mean, std)
            wz.data.normal_(mean, std)

    def forward(self, modalities): # Recommended order: text, image, audio
        # Get hidden representation
        if len(modalities) != self.modals:
            raise ValueError(
                "You passed different number of modalities than you specified before\n"
                f"Modals you passed: {len(modalities)}\nModals you specified: {self.modals}")

        h_mod_i = [torch.tanh(torch.matmul(modal, weight)) for modal, weight in zip(modalities, self.Ws)]

        #Concatenate the hidden representations
        #x_concat = torch.cat((mod1, mod2, mod3), 1)
        x_concat = torch.cat(modalities, dim = -1)

        z_mod_i = [torch.sigmoid(torch.matmul(x_concat, wz)) for wz in self.Wzs] #torch.sigmoid(torch.matmul(x_concat, self.Wz))
        h_mult_z = [zi * hi for zi, hi in zip(z_mod_i, h_mod_i)]

        h = sum(h_mult_z)

        if self.return_zs:
            return h, z_mod_i
        else:
            return h
        
#### ====================================================================
#### ========================= Pruning Heads ============================
#### ====================================================================
class ConcreteGate(nn.Module):
    def __init__(self, num_heads = 12, temperature=0.33, stretch_limits=(-0.1, 1.1), eps=1e-6,
                 hard=False):
        super(ConcreteGate, self).__init__()
        self.temperature, self.stretch_limits, self.eps = temperature, stretch_limits, eps
        self.hard, self.num_heads = hard, num_heads
        shape = [1, self.num_heads, 1]

        self.log_a = nn.Parameter(torch.empty(shape))

    def forward(self, values, is_train=None, axis=None):
        """ applies gate to values, if is_train, adds regularizer to reg_collection """
        is_train = True
        gates = self.get_gates(is_train, shape=[1, self.num_heads, 1])
        #print(f'gates: {gates.shape}')
        #print(gates)
        out = values * gates

        return out, gates

    def get_gates(self, is_train, shape=None):
        """ samples gate activations in [0, 1] interval """
        low, high = self.stretch_limits

        if is_train:
            shape = shape
            noise = torch.rand(shape).to(self.log_a.device).clamp(self.eps, 1.0 - self.eps)
            concrete = F.sigmoid((torch.log(noise) - torch.log(1 - noise) + self.log_a) / self.temperature)
        else:
            concrete = F.sigmoid(self.log_a)

        stretched_concrete = concrete * (high - low) + low
        clipped_concrete = torch.clamp(stretched_concrete, 0, 1)

        if self.hard:
            hard_concrete = (clipped_concrete > 0.5).float()
            clipped_concrete = clipped_concrete + (hard_concrete - clipped_concrete).detach()

        return clipped_concrete

def combine_heads(x):
    """
    Inverse of split heads
    input: (batch_size * n_heads * ninp * (inp_dim/n_heads))
    out: (batch_size * ninp * inp_dim)
    """
    x = x.permute(0, 2, 1)
    ret = x.reshape(*x.shape[:-2], -1)

    return ret
        

class Attention(nn.Module):
    def __init__(self, dim_inp, dim_out, prunning = False):
        super(Attention, self).__init__()

        self.u = nn.Linear(dim_inp, dim_out)
        self.v = nn.Parameter(torch.rand(dim_out), requires_grad=True)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.epsilon = 1e-10
        
        ##### ========= Prunning Heads ========= #####
        self.prunning = prunning
        self.linear = nn.Linear(dim_inp, dim_out)

    def forward(self, h, mask):
       u_it = self.u(h)
       u_it = self.tanh(u_it)

       alpha = torch.exp(torch.matmul(u_it, self.v))
       alpha = mask * alpha + self.epsilon
       denominator_sum = torch.sum(alpha, dim=-1, keepdim=True)
       alpha = mask * (alpha / denominator_sum)
       output = h * alpha.unsqueeze(2)
       output = torch.sum(output, dim=1)

       ##### ========= Prunning Heads ========= #####
       if self.prunning:
           output = output.unsqueeze(1)

           return output

       else:
           return output

##### ========== Common Attention ========== #####
class AttentionHead(nn.Module):
    """
    AttentionHead class represents a single attention head in the multi-head attention layer.
    """

    def __init__(self, dim_inp, dim_out, prunning = False):
        """
        Initializes the AttentionHead class.

        Args:
            dim_inp (int): Input dimension.
            dim_out (int): Output dimension.
        """
        super(AttentionHead, self).__init__()

        self.dim_inp = dim_inp

        # Linear transformations for query, key, and value
        self.q = nn.Linear(dim_inp, dim_out)
        self.k = nn.Linear(dim_inp, dim_out)
        self.v = nn.Linear(dim_inp, dim_out)

        # Softmax module for attention scores
        self.softmax = nn.Softmax(dim=-1)

        ##### ========= Prunning Heads ========= #####
        self.prunning = prunning

    def forward(self, input_tensor, attention_mask = None):
        """
        Forward pass of the AttentionHead class.

        Args:
            input_tensor (torch.Tensor): Input tensor.
            attention_mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention computation.
        """
        query, key, value = self.q(input_tensor), self.k(input_tensor), self.v(input_tensor)

        # Compute attention scores
        scores = torch.bmm(query, key.transpose(1, 2))

        # Scale the scores
        scale = query.size(-1) ** 0.5  # square root of the dimension of the queries
        scores = scores / scale

        # Apply attention mask
        if attention_mask is not None:
            scores = scores + attention_mask.squeeze(1)
        
        # Apply softmax
        attn = self.softmax(scores)

        # Compute context vector
        context = torch.bmm(attn, value)

        ##### ========= Prunning Heads ========= #####
        if self.prunning:
            context = context.unsqueeze(1)

            return torch.sum(context, dim = 2)
        else:
            return torch.sum(context, dim = 1)

#### ====================================================================
#### ======================= Cross-Attention ============================
#### ====================================================================            
class CrossAtt(nn.Module):
    """
    AttentionHead class represents a single attention head in the multi-head attention layer.
    """

    def __init__(self, size, ctx_dim = None):
        """
        Initializes the AttentionHead class.

        Args:
            dim_inp (int): Input dimension.
            dim_out (int): Output dimension.
        """
        super(CrossAtt, self).__init__()

        self.size = size
        
        if ctx_dim is None:
            ctx_dim =size

        self.dropout = nn.Dropout(0.1)

        # Linear transformations for query, key, and value
        self.q = nn.Linear(size, size)
        self.k = nn.Linear(size, size)
        self.v = nn.Linear(size, size)

        # Softmax module for attention scores
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_tensor, context, attention_mask = None):
        """
        Forward pass of the AttentionHead class.

        Args:
            input_tensor (torch.Tensor): Input tensor.
            attention_mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention computation.
        """

        query, key, value = self.q(input_tensor), self.k(context), self.v(context)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(1, 2))

        # Scale the scores
        scale = query.size(-1) ** 0.5
        scores = scores / scale

        # Apply attention mask
        if attention_mask is not None:
            scores = scores + attention_mask.squeeze(1)

        # Apply softmax
        attn = self.softmax(scores)
        attn = self.dropout(attn)

        # Compute context vector
        cntext = torch.matmul(self.dropout(attn), value)

        return cntext # context: [batch, n_inps, size]
    
#### ====================================================================
#### ============== Hierarchical Cross-Attention (HCA) ==================
#### ====================================================================
class HCA(nn.Module):
    def __init__(self, dim_inp:int, dim_out:int, dp:float = 0.2, prunning:bool = False):
        super(HCA, self).__init__()
        """
        Hierarchical Cross-Attention Module

        Parameters:
        -----------
        dim_inp: int
            input dimension of the tensor
        dim_out: int
            output dimension of the tensor
        dp: float, optional
            dropout rate 
        pruning: bool, optional
            Wether to prun heads or not. Available only for Multi-Head model

        Returns:
        --------
        torch.Tensor
            The contextualized tensor for the main modality
        """
        # Attention module
        self.attention = Attention(dim_inp, dim_out, prunning = prunning)

        # Cross-attention modules
        self.ca1 = CrossAtt(dim_inp)
        self.ca2 = CrossAtt(dim_inp)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(dim_inp)
        self.norm2 = nn.LayerNorm(dim_inp)

        # Dropout
        self.dp1 = nn.Dropout(dp)
        self.dp2 = nn.Dropout(dp)

    def forward(self, modal1, mask1, modal2, mask2, modal3, mask3):
        # First Cross Attention
        ca1 = self.ca1(modal1, modal2, mask2)
        ca1 = self.norm1(modal1 + self.dp1(ca1))
        
        ca2 = self.ca2(ca1, modal3, mask3)
        ca2 = self.norm2(ca1 + self.dp2(ca2))

        masked = torch.tensor(np.array([1]*ca2.size()[1], dtype = bool)).to(device)

        out = self.attention(ca2, masked)
            
        return out
    
class ParCA(nn.Module):
    def __init__(self, dim_inp:int, dim_out:int, dtype_parallel:str = 'Concat', dp:float = 0.2, prunning:bool = False):
        super(ParCA, self).__init__()
        """
        Parallel Cross-Attention Module

        Parameters:
        -----------
        dim_inp: int
            input dimension of the tensor
        dim_out: int
            output dimension of the tensor
        dtype_parallel: str, optional
            type of combination for parallel attention: 'Concat', 'Sum' or 'GMU'
        dp: float, optional
            dropout rate 
        pruning: bool, optional
            Wether to prun heads or not. Available only for Multi-Head model

        Returns:
        --------
        torch.Tensor
            The contextualized tensor for the main modality
        """

        self.dim_inp = dim_inp
        self.dim_out = dim_out
        self.dtype_parallel = dtype_parallel

        # Attention modules
        self.attention1 = AttentionHead(dim_inp, dim_out, prunning = prunning)
        self.attention2 = AttentionHead(dim_inp, dim_out, prunning = prunning)

        # Cross-attention modules
        self.ca1 = CrossAtt(dim_inp)
        self.ca2 = CrossAtt(dim_inp)

        # Linear and GMU Layers
        self.linear = nn.Linear(dim_inp * 2, dim_out)
        self.gmu = GMU(dim_out, no_modals = 2)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(dim_inp)
        self.norm2 = nn.LayerNorm(dim_inp)

        # Dropout
        self.dp1 = nn.Dropout(dp)
        self.dp2 = nn.Dropout(dp)

    def forward(self, modal1, mask1, modal2, mask2, modal3, mask3):
        # Compute cross-attention for each modal
        
        # Cross attention mod1 and mod2
        ca_one1 = self.ca1(modal1, modal2, mask2)
        # Add and Norm
        ca_one = self.norm1(modal1 + self.dp1(ca_one1))
        masked1 = torch.tensor(np.array([1]*ca_one.size()[1], dtype = bool)).to(device)
        # Attention module
        att1 = self.attention1(ca_one, masked1)
        
        # Cross attention mod1 and mod3
        ca_two1 = self.ca2(modal1, modal3, mask3)
        # Add and Norm
        ca_two = self.norm2(modal1 + self.dp2(ca_two1))
        masked2 = torch.tensor(np.array([1]*ca_two.size()[1], dtype = bool)).to(device)
        # Attention Module
        att2 = self.attention2(ca_two, masked2)
        
        # Fusion-type outputs
        if self.dtype_parallel == 'Sum':
            out = att1 + att2

        elif self.dtype_parallel == 'Concat':
            out = self.linear(torch.cat([att1, att2], dim = -1))
        
        elif self.dtype_parallel == 'GMU':
            out = self.gmu([att1, att2])

        return out


#### ====================================================================
#### ======================== Multi-Head HCA ============================
#### ====================================================================
class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention class represents the multi-head attention layer.
    """

    def __init__(self, dim_inp, dim_out, num_heads, prunning = False, gmu = False, paralel_ca = False, dtype_parallel = None):
        """
        Initializes the MultiHeadAttention class.

        Args:
            num_heads (int): Number of attention heads.
            dim_inp (int): Input dimension.
            dim_out (int): Output dimension.
        """

        super(MultiHeadAttention, self).__init__()

        if prunning and gmu:
            raise ValueError(
                "Cannot prunning and apply GMU at the same time"
            )
        
        self.prunning = prunning
        self.gmu = gmu
        self.head = num_heads

        # List of Modules (HCA or ParCA)
        if paralel_ca:
            self.heads = nn.ModuleList(
                ParCA(dim_inp, dim_out, prunning = prunning, dtype_parallel = dtype_parallel)
            )
        else:
            self.heads = nn.ModuleList(
                HCA(dim_inp, dim_out, prunning = prunning)
            )

        # Linear transformation and layer normalization
        self.linear = nn.Linear(dim_out * num_heads, dim_inp)

        ##### ========= Prunning Heads ========= #####
        if self.prunning:
            self.prun = ConcreteGate(num_heads, hard = False)

        ##### ========= GMU ========= #####
        elif self.gmu:
            self.gate = GMU(dim_out, no_modals = num_heads)

    def forward(self, modal1, mask1, modal2, mask2, modal3, mask3):
        """
        Forward pass of the MultiHeadAttention class.

        Args:
            input_tensor (torch.Tensor): Input tensor.
            attention_mask (torch.Tensor): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after multi-head attention computation.
        """

        # Compute attention for each head
        # head: [batch, 1, n_input, size] if prunning = True
        # head: [batch, n_input, size] if prunning = False
        s = [module(modal1, mask1, modal2, mask2, modal3, mask3) for module in self.heads]

        ##### ========= Prunning Heads ========= #####
        if self.prunning:
            scores = torch.cat(s, dim = 1) # head: [batch, num_heads, n_input, size]
            scores, g_vals = self.prun(scores) # head: [batch, num_heads, n_input, size]
            scores = combine_heads(scores) # [batch, n_input, size * num_heads]
            
            if self.head > 1:
                scores = self.linear(scores)

            return scores

        ##### ========= GMU ========= #####
        elif self.gmu:
            scores = self.gate(s)

            return scores

        else:
            # Concatenate attention scores from all heads
            scores = torch.cat(s, dim = -1) # [batch, n_input, size * num_heads]
            if self.head > 1:
                # Apply linear transformation
                scores = self.linear(scores)

            return scores
        
class FeedForward(nn.Module):

    def __init__(self, d_inp, d_out, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(d_inp, d_out) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_out, d_inp) # w2 and b2
        self.relu = nn.ReLU()

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        n1 = self.linear_1(x)
        relu = self.relu(n1)
        dp = self.dropout(relu)
        n2 = self.linear_2(dp)
        
        return n2
    
class Encoder(nn.Module):
    def __init__(self, dim_inp, dim_out, num_heads, prunning = False, gmu = False, paralel_ca = False, dtype_paralel = None):
        super(Encoder, self).__init__()

        self.MultiHead = MultiHeadAttention(dim_inp, dim_out, num_heads, prunning, gmu, paralel_ca, dtype_paralel)
        self.FeedForward = FeedForward(dim_inp, dim_out, 0.1)

        self.norm1 = nn.LayerNorm(dim_inp)
        self.norm2 = nn.LayerNorm(dim_inp)
        

    def forward(self, modal1, mask1, modal2, mask2, modal3, mask3):
        if len(modal1.size()) != 3:
            modal1 = modal1.unsqueeze(1)
        
        mod1 = self.MultiHead(modal1, mask1, modal2, mask2, modal3, mask3)
        
        mod1 = self.norm1(torch.sum(modal1, dim = 1) + mod1)

        modal = self.FeedForward(mod1)
        modal = self.norm1(modal + mod1)
        
        return modal
    
class Transformer(nn.Module):
    def __init__(self, dim_inp, dim_out, num_heads, num_encs, prunning = False, gmu = False, paralel_ca = False, dtype_paralel = None, encoder = False):
        super(Transformer, self).__init__()

        self.encoder = nn.ModuleList([
            Encoder(dim_inp, dim_out, num_heads, prunning, gmu, paralel_ca, dtype_paralel) for _ in range(num_encs)
        ])

    def forward(self, modal1, mask1, modal2, mask2, modal3, mask3):
        for layer in self.encoder:
            modal1 = layer(modal1, mask1, modal2, mask2, modal3, mask3)

        return modal1