# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse

import pandas as pd

from dataset import Dataset
from trainer import Trainer
from tester import Tester
from params import Params

desc = 'Temporal KG Completion methods'
parser = argparse.ArgumentParser(description=desc)

parser.add_argument('-dataset', help='Dataset', type=str, default='icews14', choices = ['icews14', 'icews05-15', 'gdelt'])
parser.add_argument('-model', help='Model', type=str, default='DE_DistMult', choices = ['Conv_TA'])
parser.add_argument('-ne', help='Number of epochs', type=int, default=1000)
parser.add_argument('-bsize', help='Batch size', type=int, default=1024)
parser.add_argument('-lr', help='Learning rate', type=float, default=0.001)
parser.add_argument('-reg_lambda', help='L2 regularization parameter', type=float, default=0.0, choices = [0.0])
parser.add_argument('-emb_dim', help='Embedding dimension', type=int, default=200)
parser.add_argument('-neg_ratio', help='Negative ratio', type=int, default=500, choices = [500])
parser.add_argument('-dropout', help='Dropout probability', type=float, default=0.4)
parser.add_argument('-save_each', help='Save model and validate each K epochs', type=int, default=20)
parser.add_argument('-se_prop', help='Static embedding proportion', type=float, default=0.68)
parser.add_argument('--embedding-shape1', type=int, default=20, help='The first dimension of the reshaped 2D embedding. The second dimension is infered. Default: 20')
parser.add_argument('--hidden-drop', type=float, default=0.3, help='Dropout for the hidden layer. Default: 0.3.')
parser.add_argument('--input-drop', type=float, default=0.2, help='Dropout for the input embeddings. Default: 0.2.')
parser.add_argument('--feat-drop', type=float, default=0.2, help='Dropout for the convolutional features. Default: 0.2.')
parser.add_argument('--hidden-size', type=int, default=9728, help='The side of the hidden layer. The required size changes with the size of the embeddings. Default: 9728 (embedding size 200).')
parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing value to use. Default: 0.1')
parser.add_argument('--ker_sz',    	dest="ker_sz",        	default=9,     		type=int,       	help='Kernel size to use')
parser.add_argument('--k_w',	  	dest="k_w", 		default=10,   		type=int, 		help='Width of the reshaped matrix')
parser.add_argument('--k_h',	  	dest="k_h", 		default=20,   		type=int, 		help='Height of the reshaped matrix')
parser.add_argument('--perm',      	dest="perm",          	default=1,      	type=int,       	help='Number of Feature rearrangement to use')
parser.add_argument('--num_filt',  	dest="num_filt",      	default=96,     	type=int,       	help='Number of filters in convolution')

parser.add_argument('-beta', help='alphi', type=float, default=0.5)

parser.add_argument('--t_ratio',    	dest="t_ratio",        	default=0.8,     		type=float,       	help='t_ratio')
parser.add_argument('--m_ratio',    	dest="m_ratio",        	default=0.3,     		type=float,       	help='m_ratio')
parser.add_argument('--d_ratio',    	dest="d_ratio",        	default=0.4,     		type=float,       	help='d_ratio')
parser.add_argument('--num_heads',    	dest="num_heads",        	default=16,     		type=int,       	help='num_heads')
parser.add_argument('--time_embs_dim',    	dest="time_embs_dim",        	default=100,     		type=int,       	help='time_embs_dim')
parser.add_argument('--t_drop',    	dest="t_drop",        	default=0.2,     		type=float,       	help='t_drop')

args = parser.parse_args()




dataset = Dataset(args.dataset)

params = Params(
    ne=args.ne,
    bsize=args.bsize,
    lr=args.lr,
    reg_lambda=args.reg_lambda,
    emb_dim=args.emb_dim,
    neg_ratio=args.neg_ratio,
    dropout=args.dropout,
    save_each=args.save_each,
    se_prop=args.se_prop,

    embedding_shape1=args.embedding_shape1,
    hidden_drop=args.hidden_drop,
    input_drop=args.input_drop,
    feat_drop=args.feat_drop,
    hidden_size=args.hidden_size,
    label_smoothing=args.label_smoothing,

    ker_sz=args.ker_sz,
    k_w=args.k_w,
    k_h=args.k_h,
    perm=args.perm,
    num_filt=args.num_filt,
    alp=args.beta,

    t_ratio=args.t_ratio,
    m_ratio=args.m_ratio,
    d_ratio=args.d_ratio,
    num_heads=args.num_heads,
    time_embs_dim = args.time_embs_dim,
    t_drop = args.t_drop,
    model=args.model
)

trainer = Trainer(dataset, params, args.model)
trainer.train()

# validating the trained models. we seect the model that has the best validation performance as the fina model
validation_idx = [str(int(args.save_each * (i + 1))) for i in range(args.ne // args.save_each)]
# validation_idx = [str(int(args.save_each * (i + 1))) for i in range(190 // args.save_each)]
best_mrr = -1.0
best_index = '0'
model_prefix = "models/" + args.model + "/" + args.dataset + "/" + params.str_() + "_"

for idx in validation_idx:
    model_path = model_prefix + idx + ".chkpnt"
    tester = Tester(params,dataset, model_path, "valid")
    mrr = tester.test()
    if mrr > best_mrr:
        best_mrr = mrr
        best_index = idx

# testing the best chosen model on the test set
print("Best epoch: " + best_index)
model_path = model_prefix + best_index + ".chkpnt"
tester = Tester(params,dataset, model_path, valid_or_test="test")
tester.test(is_save=True)
with open(f'result/{args.model}_{dataset.name}_{params.ne}_{params.bsize}_{params.lr}_{params.ker_sz}_{params.perm}.txt','a',encoding='utf-8') as file:
    file.write(f'{args.model}, {model_path}\n')
    file.write("Best epoch: " + best_index)
    file.write("\n")

