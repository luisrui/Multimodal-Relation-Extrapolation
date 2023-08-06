import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm.auto import tqdm, trange
from ml_collections import ConfigDict

from args import read_options
from utils import (
    WandBLogger, define_flags_with_default, get_user_flags,
    image_float2int, load_pickle, set_random_seed, create_log_images
)
from model import (
    MaskedMultimodalAutoencoder, extract_patches, patch_mse_loss, cross_entropy_loss_and_accuracy,
    mask_intersection, all_mask, mask_not
)
from data import (
    ImageTextDataset, TextDataset
)


FLAGS_DEF = define_flags_with_default(
    dataset_name='',
    seed=42,
    epochs=200,
    batch_size=2,
    accumulate_grad_steps=1,
    patch_size=16,
    discretized_image=False,
    image_tokenizer_type='maskgit',
    image_all_token_loss=False,
    text_all_token_loss=False,
    dataloader_n_workers=0,
    dataloader_shuffle=False,
    log_freq=50,
    plot_freq=1000,
    save_model_freq=100,
    image_loss_weight=1.0,
    text_loss_weight=0.1,
    unpaired_text_loss_weight=0.5,
    clip_gradient=1e9,
    lr_init_value=0.0,
    lr_end_value=0.0,
    lr_peak_value=1.5e-4,
    lr_warmup_epochs=0,
    weight_decay=0.05,
    load_checkpoint="./m3ae/checkpoints/m3ae_small.pkl",
    m3ae=MaskedMultimodalAutoencoder.get_default_config(),
    data=ImageTextDataset.get_default_config(),
    unpaired_text_data=TextDataset.get_default_config(),
    logging=WandBLogger.get_default_config(),
    log_all_worker=True,
)
def first_fusion_train(model, batch, args):
    image = batch['image']
    text = batch['text']
    text_padding_mask = batch['text_padding_mask']
    unpaired_text = batch['unpaired_text']
    unpaired_text_padding_mask = batch['unpaired_text_padding_mask']
    image_patches = extract_patches(image, args.patch_size)
    # Forward Propogation
    image_output, text_output, image_mask, text_mask = model(
        image_patches,
        text,
        text_padding_mask,
        deterministic=False,
    )
    _, unpaired_text_output, _, unpaired_text_mask = model(
        None,
        unpaired_text,
        unpaired_text_padding_mask,
        deterministic=False,
    )
    #Missing discretized image optimization
    image_loss = patch_mse_loss(
        image_output, image_patches,
        None if args.image_all_token_loss else image_mask
    )
    image_accuracy = 0.0

    text_loss, text_accuracy = cross_entropy_loss_and_accuracy(
        text_output, text,  
        mask_intersection(
            all_mask(text) if args.text_all_token_loss else text_mask,
            mask_not(text_padding_mask)
        )
    )
def main(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',                                   
    	world_size=args.world_size,                              
    	rank=rank                                               
    ) 
    torch.cuda.set_device(gpu)
    assert args.batch_size % args.gpus == 0
    process_index = dist.get_rank()
    process_batch_size = args.batch_size // args.world_size
    device_batch_size = process_batch_size // args.gpus
    lr_scale = args.batch_size / 256
    effective_lr = args.lr_peak_value * lr_scale
    #jax_devices = jax.local_devices()
    n_devices = torch.cuda.device_count()
    assert process_batch_size % n_devices == 0

    # logger = WandBLogger(
    #     config=FLAGS.logging,
    #     variant=variant,
    #     enable=FLAGS.log_all_worker or (process_index == 0),
    # )
    set_random_seed(args.seed * (process_index + 1))

    dataset = ImageTextDataset(ImageTextDataset.get_default_config(), process_index / args.world_size)
    image_text_sampler = torch.utils.data.distributed.DistributedSampler(
    	ImageTextDataset,
    	num_replicas=args.world_size,
    	rank=rank
    )
    paired_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=process_batch_size,
        drop_last=True,
        num_workers=args.dataloader_n_workers,
        prefetch_factor=2,
        persistent_workers=True,
        sampler=image_text_sampler
    )
    image_patch_dim = args.patch_size * args.patch_size * 3
    image_sequence_length = (dataset.config.image_size // args.patch_size) ** 2

    unpaired_text_dataset = TextDataset(TextDataset.get_default_config(), process_index / args.world_size)
    unpaired_text_sampler = torch.utils.data.distributed.DistributedSampler(
        unpaired_text_dataset,
        num_replicas=args.world_size,
        rank=rank
    )
    unpaired_text_dataloader = torch.utils.data.DataLoader(
        unpaired_text_dataset,
        batch_size=process_batch_size,
        drop_last=True,
        num_workers=args.dataloader_n_workers,
        prefetch_factor=2,
        persistent_workers=True,
        sampler=unpaired_text_sampler
    )

    steps_per_epoch = int(len(dataset) / args.batch_size)
    total_steps = steps_per_epoch * args.m3ae_epochs

    if args.discretized_image:
        tokenizer_params, encode_image, decode_image, image_vocab_size = get_image_tokenizer(
            args.image_tokenizer_type
        )
        image_output_dim = image_vocab_size
    else:
        tokenizer_params, encode_image, decode_image, image_vocab_size = (
            None, None, None, -1
        )
        image_output_dim = image_patch_dim

    model = MaskedMultimodalAutoencoder(
        text_vocab_size=dataset.vocab_size,
        image_output_dim=image_output_dim,
        config_updates=ConfigDict(dict(args.model_type)),
    )
    model = DDP(model, device_ids=[gpu])

    # learning_rate = optax.warmup_cosine_decay_schedule(
    #     init_value=FLAGS.lr_init_value * lr_scale,
    #     peak_value=FLAGS.lr_peak_value * lr_scale,
    #     warmup_steps=FLAGS.lr_warmup_epochs * steps_per_epoch // FLAGS.accumulate_grad_steps,
    #     decay_steps=total_steps // FLAGS.accumulate_grad_steps,
    #     end_value=FLAGS.lr_end_value * lr_scale,
    # )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.lr_warmup_epochs * steps_per_epoch // args.accumulate_grad_steps,
        T_mult=1,
        eta_min=args.lr_minimum
    )

    # def get_weight_decay_mask(params):
    #     flattened_params = flax.traverse_util.flatten_dict(
    #         flax.core.frozen_dict.unfreeze(params)
    #     )

    #     def decay(key):
    #         return all([k not in model.no_decay_list() for k in key])

    #     return flax.traverse_util.unflatten_dict(
    #         {key: decay(key) for key in flattened_params.keys()}
    #     )

    # if args.load_checkpoint != "":
    #     checkpoint_data = load_pickle(args.load_checkpoint)
    #     state = flax.jax_utils.replicate(checkpoint_data["state"], jax_devices)
    #     start_step = checkpoint_data["step"]
    #     del tokenizer_params
    # else:
    #     image = jnp.zeros((2, image_sequence_length, image_patch_dim), dtype=jnp.float32)
    #     text = jnp.zeros((2, dataset.config.tokenizer_max_length), dtype=jnp.int32)
    #     text_padding_mask = jnp.zeros((2, dataset.config.tokenizer_max_length))
    #     rngs = next_rng(keys=model.rng_keys())
    #     params = model.init(
    #         rngs, image, text, text_padding_mask, deterministic=False
    #     )

    #     state = flax.jax_utils.replicate(
    #         M3AETrainState.create(
    #             params=flax.core.frozen_dict.unfreeze(params),
    #             tokenizer_params=tokenizer_params,
    #             apply_fn=None,
    #             tx=optax.chain(
    #                 optax.clip_by_global_norm(FLAGS.clip_gradient),
    #                 optax.adamw(
    #                     learning_rate=learning_rate, weight_decay=FLAGS.weight_decay,
    #                     b1=0.9, b2=0.95, mask=get_weight_decay_mask,
    #                 ),
    #             ),
    #         ),
    #         jax_devices,
    #     )
    start_step = 0
    #     del params, tokenizer_params

    # train_step_fn, patch_predict_fn = create_train_step(
    #     model, learning_rate, encode_image, decode_image
    # )

    # def generate_batch():
    #     def infinite_iterator(iterator):
    #         while True:
    #             for batch in iterator:
    #                 yield tuple(
    #                     x.numpy().reshape(
    #                         n_devices, -1, *x.shape[1:]
    #                     ) for x in batch
    #                 )

    #     paired_iterator = infinite_iterator(dataloader)

    #     if FLAGS.unpaired_text_loss_weight > 0.0:
    #         unpaired_text_iterator = infinite_iterator(unpaired_text_dataloader)

    #     while True:
    #         batch = {}
    #         image, text, text_padding_mask = next(paired_iterator)
    #         batch['image'] = image.astype(np.float32)
    #         batch['text'] = text.astype(np.int32)
    #         batch['text_padding_mask'] = text_padding_mask.astype(np.float32)

    #         if FLAGS.unpaired_text_loss_weight > 0.0:
    #             unpaired_text, unpaired_text_padding_mask = next(unpaired_text_iterator)
    #             batch['unpaired_text'] = unpaired_text.astype(np.int32)
    #             batch['unpaired_text_padding_mask'] = unpaired_text_padding_mask.astype(np.float32)

    #         yield batch

    state = sync_state_across_devices(state)
    sharded_rng = jax.device_put_sharded(next_rng(n_devices), jax_devices)

    if FLAGS.accumulate_grad_steps > 1:
        accumulated_grads = flax.jax_utils.replicate(
            jax.tree_util.tree_map(jnp.zeros_like, flax.jax_utils.unreplicate(state).params),
            jax_devices
        )
        accumulated_steps = flax.jax_utils.replicate(jnp.array(0, jnp.int32), jax_devices)
    else:
        accumulated_grads = flax.jax_utils.replicate(jnp.array(0, jnp.int32), jax_devices)
        accumulated_steps = flax.jax_utils.replicate(jnp.array(0, jnp.int32), jax_devices)

    data_iterator = prefetch_to_device(generate_batch(), 2, jax_devices)
    step_counter = trange(start_step, total_steps, ncols=0)

    for step, (image, text, text_padding_mask), (unpaired_text, unpaired_text_padding_mask) in zip(step_counter, paired_dataloader, unpaired_text_dataloader):
        batch = {}
        batch['image'] = image.astype(torch.float32)
        batch['text'] = text.astype(torch.int32)
        batch['text_padding_mask'] = text_padding_mask.astype(torch.float32)
        batch['unpaired_text'] = unpaired_text.astype(torch.int32)
        batch['unpaired_text_padding_mask'] = unpaired_text_padding_mask.astype(torch.float32)
        loss = first_fusion_train(model, batch, args)
        # if FLAGS.discretized_image:
        #     encoded_image = encode_image(state.tokenizer_params, image)

if __name__ == "__main__":
    args = read_options()
    os.environ['MASTER_ADDR'] = 'localhost'             
    os.environ['MASTER_PORT'] = '1113'
    args.world_size = args.gpus * args.nodes
    # torch.multiprocessing.set_start_method("spawn")
    mp.spawn(main, args=(args,), nprocs=args.world_size)