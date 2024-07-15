import time
import numpy as np
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from TorchHelper import *
from data_loader import Data

from sklearn.metrics import f1_score, confusion_matrix, classification_report, hamming_loss, accuracy_score

run_on = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(run_on)

# Torch Helper
torch_helper = TorchHelper()

mature_w = 0.1
gory_w = 0.4
slap_w = 0.2
sarcasm_w = 0.2

lr_schedule_active = False
reduce_on_plateau_lr_schdlr = torch.optim.lr_scheduler.ReduceLROnPlateau

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    https://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        c = 0
        for j in range(len(y_true[i])):
            if y_true[i][j] == y_pred[i][j]:
               c += 1
        acc_list.append (c/4)
        #set_true = set( np.where(y_true[i])[0] )
        #set_pred = set( np.where(y_pred[i])[0] )
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        #tmp_a = None
        #if len(set_true) == 0 and len(set_pred) == 0:
        #    tmp_a = 1
        #else:
        #    tmp_a = len(set_true.intersection(set_pred))/\
        #            float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        #acc_list.append(tmp_a)
    return np.mean(acc_list)

# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------
def train(model, optimizer, task:str, path_res_out:str, num_epochs:int = 40, batch_size:int = 16,
          text_pad_length=500, img_pad_length=36, audio_pad_length=63):

    data = Data('train', text_pad_length, img_pad_length, audio_pad_length)
    dataset = DataLoader(data, batch_size = batch_size)
    lr_scheduler = reduce_on_plateau_lr_schdlr(optimizer, 'max', min_lr=1e-8, patience=2, factor=0.5)

    for epoch in range(num_epochs):
        print('[Epoch %d] / %d : %s' % (epoch + 1, num_epochs, task))
        f = open(path_res_out, "a")
        f.write('[Epoch %d] / %d : %s\n' % (epoch + 1, num_epochs, task))
        f.close()

        start_time = time.time()
        model.train()
        
        total_loss = 0

        len_data = len(dataset)
        total_len = 0

        #batch_iterator = tqdm(dataset, desc = f'Epoch: {epoch:02d}')
        for batch_idx, batch in enumerate(dataset):
            optimizer.zero_grad()

            batch_txt = batch['text'].to(device)
            batch_txt_mask = batch['text_mask'].to(device)
            batch_img = batch['image'].float().to(device)
            batch_mask_img = batch['image_mask'].to(device)
            batch_aud = batch['audio'].float().to(device)
            batch_mask_aud = batch['audio_mask'].to(device)

            model_out = model(batch_txt, batch_txt_mask, batch_img, batch_mask_img, batch_aud, batch_mask_aud)

            if task.lower() == 'binary':
                y_pred = model_out.cpu()

                loss = compute_l2_reg_val(model) + F.binary_cross_entropy(y_pred, batch['binary'])
                total_loss += loss.item()

                loss.backward()
                optimizer.step()

            else:
                mature_pred = model_out[0].cpu()
                gory_pred = model_out[1].cpu()
                slapstick_pred = model_out[2].cpu()
                sarcasm_pred = model_out[3].cpu()

                pred_mature = torch.argmax(mature_pred, -1).numpy()
                pred_gory = torch.argmax(gory_pred, -1).numpy()
                pred_slap = torch.argmax(slapstick_pred, -1).numpy()
                pred_sarcasm = torch.argmax(sarcasm_pred, -1).numpy()

                loss1 = F.binary_cross_entropy(mature_pred, batch['mature'])
                loss2 = F.binary_cross_entropy(gory_pred, batch['gory'])
                loss3 = F.binary_cross_entropy(slapstick_pred, batch['slapstick'])
                loss4 = F.binary_cross_entropy(sarcasm_pred, batch['sarcasm'])

                loss = mature_w*loss1 + gory_w*loss2 + slap_w*loss3 + sarcasm_w*loss4

                total_loss += loss.item()
                loss.backward()
                optimizer.step()

            torch_helper.show_progress(batch_idx + 1, len_data, start_time, round(total_loss / (batch_idx + 1), 4))

        val_pred, val_loss1, val_f1 = evaluate(model, 'val', path_res_out, task = task, batch_size = batch_size)

        current_lr = 0
        for pg in optimizer.param_groups:
            current_lr = pg['lr']

        if lr_schedule_active:
                lr_scheduler.step(val_f1)

        is_best = torch_helper.checkpoint_model(model, optimizer, f"Results/{task}/", val_f1, epoch + 1,
                                                        'max')

    return val_pred, val_loss1, val_f1

# ----------------------------------------------------------------------------
# Evaluate the model
# ----------------------------------------------------------------------------
def evaluate(model, partition:str, file_path:str, task:str, batch_size:int = 16,
          text_pad_length=500, img_pad_length=36, audio_pad_length=63):
    
    data = Data(partition, text_pad_length, img_pad_length, audio_pad_length)
    dataset = DataLoader(data, batch_size = batch_size)

    model.eval()

    total_loss1 = 0

    y_true = []
    predictions_mature, predictions_gory, predictions_slapstick, predictions_sarcasm = [], [], [], []

    batch_audio, batch_mask_audio = [],[]
    batch_mature, batch_gory, batch_sarcasm, batch_slapstick = [], [], [], []
    mature_true, gory_true, sarcasm_true, slapstick_true = [], [], [], []
    
    with torch.no_grad():
        for batch in dataset:
            batch_txt = batch['text'].to(device)
            batch_txt_mask = batch['text_mask'].to(device) # this is inverted for some reason, might be part of model code
            batch_img = batch['image'].float().to(device)
            batch_mask_img = batch['image_mask'].to(device)
            batch_aud = batch['audio'].float().to(device)
            batch_mask_aud = batch['audio_mask'].to(device)

            model_out = model(batch_txt, batch_txt_mask, batch_img, batch_mask_img, batch_aud, batch_mask_aud)

            if task.lower() == 'binary':
                y_pred = model_out.cpu()
                y_true.extend(list(torch.argmax(y_pred, -1).numpy()))

                loss = F.binary_cross_entropy(y_pred, batch['binary'])
                total_loss += loss.item()

            else:
                mature_pred = model_out[0].cpu()
                gory_pred = model_out[1].cpu()
                slapstick_pred = model_out[2].cpu()
                sarcasm_pred = model_out[3].cpu()

                pred_mature = torch.argmax(mature_pred, -1).numpy()
                pred_gory = torch.argmax(gory_pred, -1).numpy()
                pred_slap = torch.argmax(slapstick_pred, -1).numpy()
                pred_sarcasm = torch.argmax(sarcasm_pred, -1).numpy()

                predictions_mature.extend(list(pred_mature))
                predictions_gory.extend(list(pred_gory))
                predictions_slapstick.extend(list(pred_slap))
                predictions_sarcasm.extend(list(pred_sarcasm))

                loss = F.binary_cross_entropy(mature_pred, batch['mature'])

                total_loss += loss.item()

        
    true_values = []
    preds = []

    for i in range(len(mature_true)):
         true_values.append([mature_true[i], gory_true[i], slapstick_true[i], sarcasm_true[i]])
         preds.append([predictions_mature[i],predictions_gory[i],predictions_slapstick[i],predictions_sarcasm[i]])
         """if MT:
             preds.append([predictions_mature[i],predictions_gory[i],predictions_slapstick[i],predictions_sarcasm[i]])
         else:
             print(mout)
             preds.append(list(np.round(mout)[i]))"""
         #print ([mature_true[i], gory_true[i], slapstick_true[i], sarcasm_true[i]], [predictions_mature[i],predictions_gory[i],predictions_slapstick[i],predictions_sarcasm[i]])
         
    true_values = np.array(true_values)
    preds = np.array(preds)
    
    print ("acc_score ",accuracy_score(true_values, preds))
    print ("Hamin_score",hamming_score(np.array(true_values),np.array( preds)))
    print("Hamming_loss:", hamming_loss(true_values, preds))
    print (hamming_loss(true_values, preds) + hamming_score(np.array(true_values),np.array( preds)))

    print (classification_report(true_values, preds))
    micro1 = f1_score(mature_true, predictions_mature)
    micro2 = f1_score(gory_true, predictions_gory)
    micro3 = f1_score(slapstick_true, predictions_slapstick)
    micro4 = f1_score(sarcasm_true, predictions_sarcasm)
    
    bin_f1 = f1_score(true_values, preds, average = None)
    micro_f1 = f1_score(true_values, preds, average = 'micro')
    macro_f1 = f1_score(true_values, preds, average = 'macro')
    weighted_f1 = f1_score(true_values, preds, average = 'weighted')

    f = open(file_path, "a")

    print("Mature")
    print (confusion_matrix(mature_true, predictions_mature))
    print('weighted', f1_score(mature_true, predictions_mature, average='weighted'))
    print('micro', f1_score(mature_true, predictions_mature, average='micro')) # F1 de toda la vida
    print('f1-class 1', f1_score(mature_true, predictions_mature, pos_label = 1, average='binary')) # F1 de la clase 1
    print('f1-class 0', f1_score(mature_true, predictions_mature, pos_label = 0, average='binary')) # F1 de la clase 0 (en teor?a menor a f1-class 0)
    print('macro', f1_score(mature_true, predictions_mature, average='macro'))
    print("============================")

    f.write ("Mature\n")
    f.write('weighted: %f\n' % f1_score(mature_true, predictions_mature, average='weighted'))
    f.write('micro: %f\n' % f1_score(mature_true, predictions_mature, average='micro'))
    f.write('f1-class 1: %f\n' % f1_score(mature_true, predictions_mature, pos_label = 1, average='binary')) # F1 de la clase 1
    f.write('f1-class 0: %f\n' % f1_score(mature_true, predictions_mature, pos_label = 0, average='binary')) # F1 de la clase 0 (en teor?a menor a f1-class 0)
    f.write('macro: %f\n' % f1_score(mature_true, predictions_mature, average='macro'))
    f.write ("============================\n")

    print ("Gory")
    print (confusion_matrix(gory_true, predictions_gory))
    print('weighted', f1_score(gory_true, predictions_gory, average='weighted'))
    print('micro', f1_score(gory_true, predictions_gory, average='micro')) # F1 de toda la vida
    print('f1-class 1', f1_score(gory_true, predictions_gory, pos_label = 1, average='binary')) # F1 de la clase 1
    print('f1-class 0', f1_score(gory_true, predictions_gory, pos_label = 0, average='binary')) # F1 de la clase 0 (en teor?a menor a f1-class 0)
    print('macro', f1_score(gory_true, predictions_gory, average='macro'))
    print ("=============================")

    f.write ("Gory\n")
    f.write('weighted: %f\n' % f1_score(gory_true, predictions_gory, average='weighted'))
    f.write('micro: %f\n' % f1_score(gory_true, predictions_gory, average='micro'))
    f.write('f1-class 1: %f\n' % f1_score(gory_true, predictions_gory, pos_label = 1, average='binary')) # F1 de la clase 1
    f.write('f1-class 0: %f\n' % f1_score(gory_true, predictions_gory, pos_label = 0, average='binary')) # F1 de la clase 0 (en teor?a menor a f1-class 0)
    f.write('macro: %f\n' % f1_score(gory_true, predictions_gory, average='macro'))
    f.write ("============================\n")

    print ("Slapstick")
    print (confusion_matrix(slapstick_true, predictions_slapstick))
    print('weighted', f1_score(slapstick_true, predictions_slapstick, average='weighted'))
    print('micro', f1_score(slapstick_true, predictions_slapstick, average='micro'))  # F1 de toda la vida
    print('f1-class 1', f1_score(slapstick_true, predictions_slapstick, pos_label = 1, average='binary')) # F1 de la clase 1
    print('f1-class 0', f1_score(slapstick_true, predictions_slapstick, pos_label = 0, average='binary')) # F1 de la clase 0 (en teor?a menor a f1-class 0)
    print('macro', f1_score(slapstick_true, predictions_slapstick, average='macro'))
    print ("=============================")

    f.write ("Slapstick\n")
    f.write('weighted: %f\n' % f1_score(slapstick_true, predictions_slapstick, average='weighted'))
    f.write('micro: %f\n' % f1_score(slapstick_true, predictions_slapstick, average='micro'))
    f.write('f1-class 1: %f\n' % f1_score(slapstick_true, predictions_slapstick, pos_label = 1, average='binary')) # F1 de la clase 1
    f.write('f1-class 0: %f\n' % f1_score(slapstick_true, predictions_slapstick, pos_label = 0, average='binary')) # F1 de la clase 0 (en teor?a menor a f1-class 0)
    f.write('macro: %f\n' % f1_score(slapstick_true, predictions_slapstick, average='macro'))
    f.write ("============================\n")

    print ("Sarcasm")
    print (confusion_matrix(sarcasm_true, predictions_sarcasm))
    print('weighted', f1_score(sarcasm_true, predictions_sarcasm, average='weighted'))
    print('micro', f1_score(sarcasm_true, predictions_sarcasm, average='micro'))  # F1 de toda la vida
    print('f1-class 1', f1_score(sarcasm_true, predictions_sarcasm, pos_label = 1, average='binary')) # F1 de la clase 1
    print('f1-class 0', f1_score(sarcasm_true, predictions_sarcasm, pos_label = 0, average='binary')) # F1 de la clase 0 (en teor?a menor a f1-class 0)
    print('macro', f1_score(sarcasm_true, predictions_sarcasm, average='macro'))
    print ("=============================")

    f.write ("Sarcasm\n")
    f.write('weighted: %f\n' % f1_score(sarcasm_true, predictions_sarcasm, average='weighted'))
    f.write('micro: %f\n' % f1_score(sarcasm_true, predictions_sarcasm, average='micro'))
    f.write('f1-class 1: %f\n' % f1_score(sarcasm_true, predictions_sarcasm, pos_label = 1, average='binary')) # F1 de la clase 1
    f.write('f1-class 0: %f\n' % f1_score(sarcasm_true, predictions_sarcasm, pos_label = 0, average='binary')) # F1 de la clase 0 (en teor?a menor a f1-class 0)
    f.write('macro: %f\n' % f1_score(sarcasm_true, predictions_sarcasm, average='macro'))
    f.write ("============================\n")
    
    metric = (micro1 + micro2 + micro3 + micro4) / 4
    print("Average micro F1", metric)
    print("Micro F1", micro_f1)
    print("Macro F1", macro_f1)
    print("Weighted F1", weighted_f1)

    f.write('Micro F1: %f\n' % micro_f1)
    f.write('Macro_F1: %f\n'% macro_f1)
    f.write('Weighted_F1: %f\n'% weighted_f1)
    
    f.write('acc_score: %f\n' % accuracy_score(true_values, preds))
    f.write('Hamin_score: %f\n'% hamming_score(np.array(true_values), np.array( preds)))
    f.write('Hamming_loss: %f\n'% hamming_loss(true_values, preds))
    f.close()

    return y_true, \
           total_loss1 / len(dataset), \
           metric