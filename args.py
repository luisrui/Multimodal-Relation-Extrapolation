# -*- coding: utf-8 -*-
import argparse

def read_options():
    parser = argparse.ArgumentParser()
    # Base settlement
    parser.add_argument("--dataset", default="FB15K-237-clear", type=str)
    parser.add_argument("--seed", default=210, type=int)
    parser.add_argument("--cuda", default=1, type=int)
    parser.add_argument('--model_type', default='small', type=str)
    parser.add_argument('--saved_model_name', default='nogcn_zsl_transe_small_FB15K-237-clear', type=str)
    #parser.add_argument('--saved_d_model_name', default='DistillModel_FB15K-237-clear', type=str)
    parser.add_argument('--pretrained_model_name', default='', type=str)
    #parser.add_argument('--pretrained_distill_model', default='', type=str)
    parser.add_argument('--evaluate', action='store_true')
    # Multiprocessing info
    parser.add_argument('--nodes', default=1, type=int, metavar='N')
    parser.add_argument('--gpus', default=2, type=int,
                        help='number of gpus per node')
    parser.add_argument('--nr', default=0, type=int,
                        help='ranking within the nodes')
    # fusion modal Specification
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--sample_size', default=5, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--save_epochs', default=10, type=int)
    parser.add_argument('--eval_epochs', default=10, type=int)
    parser.add_argument('--dataloader_n_workers', default=4, type=int)
    parser.add_argument('--image_mask_ratio', default='0.75', type=float)
    parser.add_argument('--text_mask_ratio', default='0.75', type=float)
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--image_loss_weight', default=0.7, type=float)
    parser.add_argument('--text_loss_weight', default=0.5, type=float)
    parser.add_argument('--gcn_loss_weight', default=0.7, type=float)
    parser.add_argument('--contrastive_loss_weight', default=0.5, type=float)
    parser.add_argument('--image_all_token_loss', default=False, type=bool)
    parser.add_argument('--text_all_token_loss', default=False, type=bool)
    # Optimization Part
    parser.add_argument('--lr_maximum', default=0.0001, type=float)
    parser.add_argument('--lr_minimum', default=0, type=float)
    parser.add_argument('--discretized_image', default=False, type=bool)
    parser.add_argument('--lr_warmup_epochs', default=5, type=int)
    parser.add_argument('--accumulate_grad_steps', default=1, type=int)

    ## GCN part
    parser.add_argument('--emb_dim', default=200, type=int)
    # # Distill Model Specification
    # parser.add_argument('--D_lr_maximum', default=0.0001, type=float)
    # parser.add_argument('--D_epoch', default=10, type=float)
    # parser.add_argument('--rel_batch_size', default=8, type=int)

    #WGAN generation part
    parser.add_argument("--test_sample", default=20, type=int)
    parser.add_argument("--no_meta", action="store_true")
    parser.add_argument("--max_neighbor", default=50, type=int)
    parser.add_argument("--noise_dim", default=15, type=int)

    parser.add_argument("--train_times", default=1000, type=int)
    parser.add_argument("--D_epoch", default=1, type=int)
    parser.add_argument("--G_epoch", default=5, type=int)
    parser.add_argument("--D_batch_size", default=256, type=int)
    parser.add_argument("--G_batch_size", default=256, type=int)
    parser.add_argument("--gan_batch_rela", default=2, type=int)

    parser.add_argument("--lr_D", default=0.0001, type=float)

    parser.add_argument("--lr_E", default=0.0001, type=float)
    parser.add_argument("--pretrain_times",default=10000, type=int, help="total training steps for pretraining")
    parser.add_argument("--pretrain_batch_size", default=64, type=int)
    parser.add_argument("--pretrain_few", default=20, type=int)
    parser.add_argument("--pretrain_subepoch", default=10, type=int)
    parser.add_argument("--pretrain_margin", default=5.0, type=float, help="pretraining margin loss")
    parser.add_argument("--pretrain_loss_every", default=500, type=int)

    parser.add_argument("--log_every", default=1000, type=int)
    parser.add_argument("--loss_every", default=50, type=int)
    parser.add_argument("--eval_every", default=500, type=int)

    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    args.save_path = f'./origin_data/{args.dataset}/Embed_used'

    return args


if __name__ == "__main__":
    read_options()
