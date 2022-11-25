
# from ast import main
# from tkinter.tix import MAIN
# from unicodedata import name

import argparse
from train import model_train
from test import model_test
from plot import class_bar_plot
if __name__ == '__main__':

    print('run')
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type = int, default=1)
    parser.add_argument('-epochs', type = int, default=10)
    parser.add_argument('-lr', type = float, default=1e-6)
    parser.add_argument('-save_path', type=str, default='.')
    parser.add_argument('-model_name', type=str, default='./model_origin.bin')
    parser.add_argument('-doing', type=str, default='train')
    args = parser.parse_args()

    if args.doing == 'train':
        model_train(args)
    elif args.doing =='test':
        error_samples, acc, class_count, pred_list, label_list = model_test(args)
        class_bar_plot(class_count,'acc_by_class.png')