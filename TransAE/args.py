# -*- coding: utf-8 -*-
import argparse

def read_options():
    parser = argparse.ArgumentParser()
    # Base settlement
    parser.add_argument("--dataset", default="FB15K-237", type=str)
    parser.add_argument("--seed", default="33", type=int)
    parser.add_argument('--saved_model_name', default='optimized_0.75_dist_tinymodel_fb15', type=str)
    parser.add_argument('--pretrained_model_name', default='epoch2_optimized_0.75_dist_tinymodel_fb15', type=str)
    parser.add_argument('--evaluate', action='store_true')
    # Multiprocessing info
    parser.add_argument('--nodes', default=1, type=int, metavar='N')
    parser.add_argument('--gpus', default=8, type=int,
                        help='number of gpus per node')
    parser.add_argument('--nr', default=0, type=int,
                        help='ranking within the nodes')
    # fusion modal Specification
    parser.add_argument("--nbatches", default="100", type=int)
    parser.add_argument('--emb_dim', default=200, type=int)

    parser.add_argument('--sample_size', default=3, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--dataloader_n_workers', default=4, type=int)
    parser.add_argument('--image_mask_ratio', default='0.75', type=float)
    parser.add_argument('--text_mask_ratio', default='0.75', type=float)
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--image_loss_weight', default=0.5, type=float)
    parser.add_argument('--text_loss_weight', default=0.5, type=float)
    parser.add_argument('--gcn_loss_weight', default=0.75, type=float)
    parser.add_argument('--unpaired_text_loss_weight', default=1.0, type=float)
    parser.add_argument('--image_all_token_loss', default=False, type=bool)
    parser.add_argument('--text_all_token_loss', default=False, type=bool)
    # Optimization Part
    parser.add_argument('--lr_warmup_epochs', default=5, type=int)
    parser.add_argument('--accumulate_grad_steps', default=1, type=int)
    parser.add_argument('--lr_minimum', default=0.0, type=float)
    parser.add_argument('--discretized_image', default=False, type=bool)

    
    # Wgan Specification
    parser.add_argument("--embed_model", default="TransM3AE", type=str)

    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes

    return args


if __name__ == "__main__":
    read_options()
