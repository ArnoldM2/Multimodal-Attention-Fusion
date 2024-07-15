import argparse

def get_congfig():
    parser = argparse.ArgumentParser()

    # Data input setting
    parser.add_argument('--input_json_train', type = str,
                        help='path to the json file containing training set')
    parser.add_argument('--input_json_val', type = str,
                        help='path to the json file containing validation set')
    parser.add_argument('--input_json_test', type = str,
                        help='path to the json file containing test set')
    
    ## Mode
    parser.add_argument('--mode', type = str, default = 'train')
    

    