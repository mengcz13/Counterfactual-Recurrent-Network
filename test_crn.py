# Copyright (c) 2020, Ioana Bica

import os
import argparse
import logging
import random
import tensorflow as tf
import numpy as np

from CRN_encoder_evaluate import test_CRN_encoder
from CRN_decoder_evaluate import test_CRN_decoder
from utils.cancer_simulation import get_cancer_sim_data


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chemo_coeff", default=2, type=int)
    parser.add_argument("--radio_coeff", default=2, type=int)
    parser.add_argument("--results_dir", default='results')
    parser.add_argument("--model_name", default="crn_test_2")
    parser.add_argument("--b_encoder_hyperparm_tuning", default=False)
    parser.add_argument("--b_decoder_hyperparm_tuning", default=False)
    parser.add_argument("--debug", action='store_true', help='debugging model, only train 1 epoch')
    parser.add_argument("--gr_off", action='store_true', help='switch off gradient reversal (lambda=0)')
    parser.add_argument("--seed", type=int, default=-1)
    return parser.parse_args()


if __name__ == '__main__':

    args = init_arg()

    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    pickle_map = get_cancer_sim_data(chemo_coeff=args.chemo_coeff, radio_coeff=args.radio_coeff, b_load=False,
                                          b_save=False, model_root=args.results_dir)


    models_dir = '{}/{}'.format(args.results_dir, args.model_name)
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    encoder_model_name = 'encoder_' + args.model_name
    encoder_hyperparams_file = '{}/{}_best_hyperparams.txt'.format(models_dir, encoder_model_name)

    if args.seed > 0:
        random.seed(args.seed)
        tf.set_random_seed(args.seed)
        np.random.seed(args.seed)

    rmse_encoder = test_CRN_encoder(pickle_map=pickle_map, models_dir=models_dir,
                                    encoder_model_name=encoder_model_name,
                                    encoder_hyperparams_file=encoder_hyperparams_file,
                                    b_encoder_hyperparm_tuning=args.b_encoder_hyperparm_tuning,
                                    b_debug=args.debug,
                                    b_gr_off=args.gr_off)


    decoder_model_name = 'decoder_' + args.model_name
    decoder_hyperparams_file = '{}/{}_best_hyperparams.txt'.format(models_dir, decoder_model_name)

    """
    The counterfactual test data for a sequence of treatments in the future was simulated for a 
    projection horizon of 5 timesteps. 
   
    """

    max_projection_horizon = 5
    projection_horizon = 5
    
    rmse_decoder = test_CRN_decoder(pickle_map=pickle_map, max_projection_horizon=max_projection_horizon,
                                    projection_horizon=projection_horizon,
                                    models_dir=models_dir,
                                    encoder_model_name=encoder_model_name,
                                    encoder_hyperparams_file=encoder_hyperparams_file,
                                    decoder_model_name=decoder_model_name,
                                    decoder_hyperparams_file=decoder_hyperparams_file,
                                    b_decoder_hyperparm_tuning=args.b_decoder_hyperparm_tuning,
                                    b_debug=args.debug,
                                    b_gr_off=args.gr_off)

    logging.info("Chemo coeff {} | Radio coeff {}".format(args.chemo_coeff, args.radio_coeff))
    print("RMSE for one-step-ahead prediction.")
    print(rmse_encoder)

    print("Results for 5-step-ahead prediction.")
    print(rmse_decoder)

    with open(os.path.join(models_dir, 'results.log'), 'w') as f:
        f.write("RMSE for one-step-ahead prediction: {}\n".format(rmse_encoder))
        f.write("Results for 5-step-ahead prediction: {}\n".format(rmse_decoder))
