# -*- coding: utf-8 -*-
import argparse

def read_options():
    parser = argparse.ArgumentParser()
    # Base settlement
    parser.add_argument("--dataset", default="FB15K-237", type=str)
    parser.add_argument("--seed", default=27, type=int)
    parser.add_argument("--cuda", default=5, type=int)
    parser.add_argument('--model_type', default='tiny', type=str)
    parser.add_argument('--saved_model_name', default='t1_357_dist_tinymodel_wn18', type=str)
    parser.add_argument('--pretrained_model_name', default='epoch60_t1_357_dist_tinymodel_wn18', type=str)
    parser.add_argument('--evaluate', action='store_true')
    # Multiprocessing info
    parser.add_argument('--nodes', default=1, type=int, metavar='N')
    parser.add_argument('--gpus', default=2, type=int,
                        help='number of gpus per node')
    parser.add_argument('--nr', default=0, type=int,
                        help='ranking within the nodes')
    # fusion modal Specification
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--sample_size', default=4, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--start_epoch', default=45, type=int)
    parser.add_argument('--eval_epochs', default=5, type=int)
    parser.add_argument('--save_epochs', default=5, type=int)
    parser.add_argument('--dataloader_n_workers', default=4, type=int)
    parser.add_argument('--image_mask_ratio', default='0.75', type=float)
    parser.add_argument('--text_mask_ratio', default='0.75', type=float)
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--image_loss_weight', default=0.3, type=float)
    parser.add_argument('--text_loss_weight', default=0.5, type=float)
    parser.add_argument('--gcn_loss_weight', default=0.7, type=float)
    parser.add_argument('--unpaired_text_loss_weight', default=1.0, type=float)
    parser.add_argument('--image_all_token_loss', default=False, type=bool)
    parser.add_argument('--text_all_token_loss', default=False, type=bool)
    # Optimization Part
    parser.add_argument('--lr_warmup_epochs', default=3, type=int)
    parser.add_argument('--accumulate_grad_steps', default=1, type=int)
    parser.add_argument('--lr_minimum', default=0.0, type=float)
    parser.add_argument('--discretized_image', default=False, type=bool)

    ## GCN part
    parser.add_argument('--emb_dim', default=200, type=int)
    
    # Wgan Specification
    parser.add_argument("--embed_model", default="TransM3AE", type=str)
    # parser.add_argument("--prefix", default="intial", type=str)

    # # embedding dimension
    # parser.add_argument(
    #     "--embed_dim", default=200, type=int, help="dimension of triple embedding"
    # )
    # parser.add_argument(
    #     "-w_dim", type=int, default=384, help="dimension of word embedding [50, 300]"
    # )
    # parser.add_argument(
    #     "--ep_dim", default=400, type=int, help="dimension of entity pair embedding"
    # )
    # parser.add_argument("--noise_dim", default=15, type=int)

    # # feature extractor pretraining related
    # parser.add_argument("--pretrain_batch_size", default=64, type=int)
    # parser.add_argument("--pretrain_few", default=20, type=int)
    # parser.add_argument("--pretrain_subepoch", default=10, type=int)
    # parser.add_argument(
    #     "--pretrain_margin", default=5.0, type=float, help="pretraining margin loss"
    # )
    # parser.add_argument(
    #     "--pretrain_times",
    #     default=10000,
    #     type=int,
    #     help="total training steps for pretraining",
    # )
    # parser.add_argument("--pretrain_loss_every", default=500, type=int)
    # # parser.add_argument("--pretrain_eval_every", default=2000, type=int)

    # # adversarial training related
    # # batch size
    # parser.add_argument("--D_batch_size", default=256, type=int)
    # parser.add_argument("--G_batch_size", default=256, type=int)
    # parser.add_argument("--gan_batch_rela", default=2, type=int)
    # # learning rate
    # parser.add_argument("--lr_G", default=0.0001, type=float)
    # parser.add_argument("--lr_D", default=0.0001, type=float)
    # parser.add_argument("--lr_E", default=0.0001, type=float)
    # # training times
    # parser.add_argument("--train_times", default=10000, type=int)
    # parser.add_argument("--D_epoch", default=1, type=int)
    # parser.add_argument("--G_epoch", default=5, type=int)
    # # log
    # parser.add_argument("--log_every", default=1000, type=int)
    # parser.add_argument("--loss_every", default=50, type=int)
    # parser.add_argument("--eval_every", default=500, type=int)
    # # hyper-parameter
    # parser.add_argument("--test_sample", default=20, type=int)
    # parser.add_argument("--dropout", default=0.5, type=float)
    # parser.add_argument("--REG_W", default=0.001, type=float)
    # parser.add_argument("--REG_Wz", default=0.0001, type=float)
    # parser.add_argument("--max_neighbor", default=50, type=int)
    # parser.add_argument("--grad_clip", default=5.0, type=float)
    # parser.add_argument("--weight_decay", default=0.0, type=float)

    # parser.add_argument("--fine_tune", action="store_true")
    # parser.add_argument("--aggregate", default="max", type=str)
    # parser.add_argument("--no_meta", action="store_true")

    # # switch
    # parser.add_argument("--generate_text_embedding", action="store_true")
    # parser.add_argument("--pretrain_feature_extractor", action="store_true")

    # parser.add_argument(
    #     "--device",
    #     type=int,
    #     default=0,
    #     help="device to use for iterate data, -1 means cpu [default: 0]",
    # )

    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes

    return args


if __name__ == "__main__":
    read_options()
