from torch import nn
from transformers import BertModel

class FeatureEncoding(nn.Module):
    def __init__(self):
        super(FeatureEncoding, self).__init__()

        self.rnn_units = 512
        self.embedding_dim = 768
        dropout = 0.2

        # Text -> BertModel from pretrained
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True,
                                              output_attention = True)
        
        # Auido -> dim_inp: 128 -> rnn_units: 512
        audio_size = 128
        self.rnn_audio = nn.LSTM(audio_size, self.rnn_units, num_layers=2, bidirectional=True, batch_first=True)
        self.rnn_audio_drop_norm = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.rnn_units*2, eps=1e-5),
        )

        self.sequential_audio = nn.Sequential(
            nn.Linear(self.rnn_units*2, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(0.3),
        )

        # Image -> dim_inp: 1024 from i3D -> rnn_units: 512
        image_size = 1024
        self.rnn_img = nn.LSTM(image_size, self.rnn_units, num_layers=2, bidirectional=True, batch_first=True)
        self.rnn_img_drop_norm = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.rnn_units*2, eps=1e-5),
        ) 
        
        self.sequential_image = nn.Sequential(
            nn.Linear(self.rnn_units*2, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(0.3),
        )

    def forward(self, text, text_mask, image, image_mask, audio, audio_mask):
        """
        Encoding the three different modalities with LSTM and Bert

        Parameters:
        -----------
        text : torch.Tensor
            Input tensor representing the tokenized text, with shape (batch_size, sequence_length).
        text_mask : torch.Tensor
            Attention mask for the text, with shape (batch_size, sequence_length). This mask differentiates between valid tokens (1) and padding tokens (0).
        image : torch.Tensor
            Input tensor representing the image data, with shape (batch_size, sequence_length, 1024). This data is processed by an LSTM.
        image_mask : torch.Tensor
            Attention mask for the image data, with shape (batch_size, sequence_length). This mask differentiates between valid image segments (1) and padding segments (0).
        audio : torch.Tensor
            Input tensor representing the audio data, with shape (batch_size, sequence_length, 128). This data is processed by an LSTM.
        audio_mask : torch.Tensor
            Attention mask for the audio data, with shape (batch_size, sequence_length). This mask differentiates between valid audio segments (1) and padding segments (0).

        """
        # Text Enconded
        hidden, _ = self.bert(text)[-2:]
        text_encoded = hidden[-1]
        
        # Image Encoded ->
        rnn_img_encoded, (hid, ct) = self.rnn_img(image)
        rnn_img_encoded = self.rnn_img_drop_norm(rnn_img_encoded)
        img_encoded = self.sequential_image(rnn_img_encoded)

        # Audio Encoded -> 
        rnn_audio_encoded, (hid_audio, ct_audio) = self.rnn_audio(audio)
        rnn_audio_encoded = self.rnn_audio_drop_norm(rnn_audio_encoded)
        audio_encoded = self.sequential_audio(rnn_audio_encoded)

        # Masks
        extended_text_attention_mask = text_mask.float().unsqueeze(1).unsqueeze(2)
        extended_text_attention_mask = extended_text_attention_mask.to(dtype = next(self.parameters()).dtype)
        extended_text_attention_mask = (1.0 - extended_text_attention_mask) * -10000.0

        extended_image_attention_mask = image_mask.float().unsqueeze(1).unsqueeze(2)
        extended_image_attention_mask = extended_image_attention_mask.to(dtype = next(self.parameters()).dtype)
        extended_image_attention_mask = (1.0 - extended_image_attention_mask) * -10000.0

        extended_audio_attention_mask = audio_mask.float().unsqueeze(1).unsqueeze(2)
        extended_audio_attention_mask = extended_audio_attention_mask.to(dtype = next(self.parameters()).dtype)
        extended_audio_attention_mask = (1.0 - extended_audio_attention_mask) * -10000.0

        return text_encoded, extended_text_attention_mask, img_encoded, extended_image_attention_mask, audio_encoded, extended_audio_attention_mask