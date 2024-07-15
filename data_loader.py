import os
import torch
import numpy as np
import json
from TorchHelper import pad_segment, mask_vector, pad_features

from torch import nn
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, data:str, text_pad_len:int = 500, img_pad_len:int = 36, aud_pad_len:int = 63):
        """
        Returns the data processed accordingly

        Returns `dict` with all the data and labels processed

        Parameters
        ----------
        data: str
            The partition you want, either train, val or test
        text_pad_len: int, optional
            The maximun length of padding text
        img_pad_len: int, optional
            The maximun length of padding image features
        aud_pad_len: int, optional
            The maximun length of padding audio features
        """
        
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(self.base_dir, f'processed_data/{data}_features_lrec_camera.json')

        self.data = json.load(open(json_path, 'r'))
        self.text_pad_length = text_pad_len
        self.img_pad_length = img_pad_len
        self.aud_pad_length = aud_pad_len

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        key = list(self.data.keys())[idx]
        data = self.data[key]

        # LOAD IMAGE FEATURES 
        img_rgb_ft = os.path.join(self.base_dir, f'Features/i3D_vecs/{key}_rgb.npy')
        img_flow_ft = os.path.join(self.base_dir, f'Features/i3D_vecs/{key}_flow.npy')

        try:
            rgb_ft = np.load(img_rgb_ft)
            flow_ft = np.load(img_flow_ft)

            img_ft = torch.tensor(rgb_ft + flow_ft)

            masked_img = mask_vector(self.img_pad_length, img_ft)
            image_vec = pad_segment(img_ft, self.img_pad_length)

        except:
            print('Image not found')
            image_vec = torch.zeros((self.img_pad_length, 1024))
            masked_img = torch.zeros(self.img_pad_length)

        # LOAD AUDIO FEATURES
        aud_ft = os.path.join(self.base_dir, f'Features/vgg_vecs/{key}_vggish.npy')
        try:
            audio_vec = np.load(aud_ft)
            audio_vec = torch.tensor(audio_vec)
        except FileNotFoundError:
            print('Audio not found')
            audio_vec = torch.zeros((1, 128))

        masked_aud = mask_vector(self.aud_pad_length, audio_vec)
        audio_vec = pad_segment(audio_vec, self.aud_pad_length)

        # OBTAIN TEXT FEATURES
        emb_idxs = torch.tensor(data['indexes'])
        mask = mask_vector(self.text_pad_length, emb_idxs)
        text = pad_features([emb_idxs], self.text_pad_length)[0]

        binary = torch.tensor(data['y']) 
        mature = torch.tensor(data["mature"])
        gory = torch.tensor(data["gory"])
        sarcasm = torch.tensor(data["sarcasm"])
        slapstick = torch.tensor(data["slapstick"])

        return {
            'text': text,
            'text_mask': mask,
            'image': image_vec.float(),
            'image_mask': masked_img,
            'audio': audio_vec.float(),
            'audio_mask': masked_aud,
            'binary': binary.float(),
            "mature": mature.float(),
            "gory": gory.float(),
            "sarcasm": sarcasm.float(),
            "slapstick": slapstick.float()
        }



if __name__=='__main__':
    dataset = Data("test")
    idx = 0
    for item in dataset:
        if idx == 2:
            break
        for key, value in item.items():
            print(key)
            print(value)
            print(value.shape)
            print()
        idx += 1
