from eval import train, evaluate
from Model import HICCAP
from TorchHelper import TorchHelper

import os
from torch import optim
import torch

seed = 7
torch.manual_seed(seed)
CUBLAS_WORKSPACE_CONFIG=":4096:8"
if torch.cuda. \
        is_available():
    torch.cuda.manual_seed_all(7)
    print(' ==================== CUDA ==================== ')
torch.backends.cudnn.enabled = True
torch.autograd.set_detect_anomaly(True)

run_on = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(run_on)

run_mode = 'run'
torch_helper = TorchHelper()

def create_model(dim_inp, dim_out, task, num_heads, encoder, num_layers, prunning, gmu, parallel_ca, dtype_parallel):

    model =  HICCAP(dim_inp, dim_out, task, num_heads, encoder, num_layers, prunning, gmu, parallel_ca, dtype_parallel)
    model.cuda()

    if run_mode == 'resume' or run_mode == 'test_resume' or run_mode == 'test':
        #model1 =  Bert_Model_CL()
        #model1 = Bert_Model_Pretraining()

        checkpoint_path = "Results/PreTraining/"#Pretraining_Kinetics
        #torch_helper.load_saved_model(model1, checkpoint_path + 'Matching_best.pth') #last_pretrain_All.pth
        print('model loaded')


        #model.features = model1.features
    model.to(device)

    return model

if __name__=='__main__':
    dim_inp = 768
    dim_out = 768
    num_epochs = 25
    batch_size = 16
    learning_rate = 0.00002

    task = 'Binary'
    
    num_heads = 1
    encoder = False
    num_layers = 1
    
    prunning = False
    gmu = False

    parallel_ca = True
    dtype_parallel = 'GMU'

    if parallel_ca:
        combine = 'ParCA'
    else:
        combine = 'HCA'

    output_dir_path = 'Results/'+ task # + '/'

    if prunning:
        MSAFusion = 'Prun'
    elif gmu:
        MSAFusion = 'GMU'
    else:
        MSAFusion = 'Concat'

    if parallel_ca:
        if run_mode == 'test':
            output_path = f"{output_dir_path}/PreTrained/Arnold/{MSAFusion}MSA_{combine}{dtype_parallel}"
            #path_res_out = f"{output_path}/{name}_MultiBench_{attention}_{MSAFusion}MSA_{combine}{paralel}_Heads{nhead}_Encs{lys}.out"
            path_res_out = f"{output_path}/{MSAFusion}MSA_{combine}{dtype_parallel}_Heads{num_heads}_Encs{num_layers}.out"
        else:
            #output_path = f"{output_dir_path}/{combine}_{MSAFusion}_{attention}/{paralel}/Encoders_{lys}/{exp}"
            #path_res_out = f"{output_path}/{run_name}_{attention}_{MSAFusion}MSA_{combine}{paralel}_{nhead}.out"
            #output_path = f"{output_dir_path}/Elaheh/{MSAFusion}MSA_{combine}{paralel}/{exp}"
            output_path = f"{output_dir_path}/Arnold/{MSAFusion}MSA_{combine}{dtype_parallel}"
            path_res_out = f"{output_path}/{MSAFusion}MSA_{combine}{dtype_parallel}_Heads{num_heads}_Encs{num_layers}.out"
    else:
        if run_mode == 'test':
            output_path = f"{output_dir_path}/PreTrained/HCA/{MSAFusion}MSA_HCA"
            #path_res_out = f"{output_path}/{name}_MultiBench_{attention}_Elaheh_Hierarchical_Heads{nhead}_Encs{lys}.out"
            path_res_out = f"{output_path}/Elaheh_Hierarchical_Heads{num_heads}_Encs{num_layers}.out"
        else:
            #output_path = f"{output_dir_path}/{MSAFusion}_{attention}/Encoders_{lys}/{exp}"
            output_path = f"{output_dir_path}/HCA/{MSAFusion}MSA_HCA"
            #path_res_out = f"{output_path}/{name}_{attention}_{MSAFusion}MSA_Hierarchical_{nhead}_{lys}.out"
            path_res_out = f"{output_path}/Elaheh_Hierarchical_Heads{num_heads}_Encs{num_layers}.out"

    if not os.path.exists(output_path):
            os.makedirs(output_path)
    
    f = open(path_res_out, "w")
    f.write('=========================\n')
    f.write('PARAMETERS:\n')
    f.write(f'Type of combination heads: Concatenation\n')
    f.write(f'Number of heads: {num_heads}\n')
    f.write(f'Number of encoders: {num_layers}\n')
    f.write(f'Prunning: {prunning}\n')
    f.write(f'GMU: {gmu}\n')
    f.write(f'Classification: {task}\n')
    f.write(f'Experiment seed: {seed}\n')
    f.write('=========================\n\n')
    f.close()
    
    print('=========================')
    print('PARAMETERS:')
    print(f'Type of combination heads: Concatenation')
    print(f'Number of heads: {num_heads}')
    print(f'Number of encoders: {num_layers}\n')
    print(f'Prunning: {prunning}')
    print(f'GMU: {gmu}')
    print(f'Classification: {task}\n')
    print(f'Experiment seed: {seed}')
    print('=========================\n')

    model = create_model(dim_inp, dim_out, task, num_heads, encoder, num_layers, prunning, gmu, parallel_ca, dtype_parallel)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    val_pred, val_loss1, val_f1 = train(model, optimizer, task, path_res_out, num_epochs, batch_size)

    val_pred, val_loss1, val_f1 = evaluate(model, 'test', path_res_out, task, batch_size)