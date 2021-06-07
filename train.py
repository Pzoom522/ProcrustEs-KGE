#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

torch.autograd.set_detect_anomaly(True)

import argparse
import logging
import os


from model_train import ProcrustEs

from experiment_impact_tracker.compute_tracker import ImpactTracker

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='ProcrustEs',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
    parser.add_argument('-td', '--total_dim', default=2000, type=int)
    parser.add_argument('-sd', '--sub_dim', default=20, type=int)
    parser.add_argument('--init_embedding', default=None, type=str)
    parser.add_argument('--use_scale', type=bool, default=True)
    parser.add_argument('--max_step', default=1000, type=int)
    parser.add_argument('--save_step', default=200, type=int)
    # parser.add_argument('--theta', default=0.005, type=float)
    parser.add_argument('-save', '--save_path', default="", type=str)

    parser.add_argument('--seed', default=999, type=int)  # fixed

    parser.add_argument('--eps', default=1e-7, type=float)
    parser.add_argument('--reg', default=0., type=float)
    parser.add_argument('--gamma', default=1, type=float)

    args = parser.parse_args(args)
    return args

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    rel_ent_dict = {}
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            rel_ent_dict.setdefault(relation2id[r], [])
            rel_ent_dict[relation2id[r]].append((entity2id[h], entity2id[t]))
    for rel_id in rel_ent_dict:
        rel_ent_dict[rel_id] = torch.LongTensor(rel_ent_dict[rel_id])
    return rel_ent_dict

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''
    log_file = os.path.join(args.save_path, 'train.log')
    

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO if not args.debug else logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO if not args.debug else logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def main(args):
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    set_logger(args)

    tracker = ImpactTracker(args.save_path)
    tracker.launch_impact_monitor()


    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    nentity = len(entity2id)
    nrelation = len(relation2id)

    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    
    rel_ent_dict = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(rel_ent_dict))
    
    model = ProcrustEs(rel_ent_dict, nentity, nrelation, args.total_dim, args.sub_dim, args.cuda, args.save_path, args.eps)
    optimizer = torch.optim.Adam(model.parameters(), 
        lr=args.learning_rate, 
        eps=args.eps, 
        weight_decay=args.reg,
        )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.save_step, gamma=args.gamma, last_epoch=-1)

    old_loss = torch.tensor(float("Inf"))
    
    if args.cuda:
        model = model.cuda()
        old_loss = old_loss.cuda()
    
    # training loop
    for epoch in range(args.max_step):
        info = tracker.get_latest_info_and_check_for_errors()
        model.normalise()
        save_flag = not ((epoch + 1) % args.save_step)
        loss = model(save=save_flag)
        logging.info(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        old_loss = loss
    model(save=True)
    

if __name__ == '__main__':
    main(parse_args())