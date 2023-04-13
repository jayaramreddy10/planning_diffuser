import diffuser.utils as utils
import pdb


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'

args = Parser().parse_args('diffusion')


#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
)

# [ utils/preprocessing ] Segmented maze2d-large-v1 | 1061 paths | min length: 67 | max length: 30470
# [ datasets/buffer ] Finalized replay buffer | 1062 episodes
# [ datasets/buffer ] Fields:
#     actions: (1062, 40000, 2)
#     infos/goal: (1062, 40000, 2)
#     infos/qpos: (1062, 40000, 2)
#     infos/qvel: (1062, 40000, 2)
#     observations: (1062, 40000, 4)
#     rewards: (1062, 40000, 1)
#     terminals: (1062, 40000, 1)
#     timeouts: (1062, 40000, 1)
#     next_observations: (1062, 40000, 4)
#     normed_observations: (1062, 40000, 4)
#     normed_actions: (1062, 40000, 2)
# [ utils/config ] Imported diffuser.models:TemporalUnet

render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),
    env=args.dataset,
)

dataset = dataset_config()
# len(dataset): 3697722

	# dataset[i][0].shape: (384, 6) 
	# dataset[i][1]: 2 elements in dict: (0, 383) as initial, final states are fixed in n_horizon
	# {0: array([ 0.04206157, ...e=float32), 383: array([ 0.31135416, ...e=float32)}   , each with dim = (4, )
    
renderer = render_config()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim
# dataset.n_episodes = 1062
# dataset.path_lengths.shape = (1062, ) , it has path len for every episode

#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    device=args.device,
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=args.horizon,   #384
    observation_dim=observation_dim,   #4
    action_dim=action_dim,   #2
    n_timesteps=args.n_diffusion_steps,   #256
    loss_type=args.loss_type,   #l2
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    ## loss weighting
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    device=args.device,
)

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size,   #32
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),    #400000
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
    n_samples=args.n_samples,    #10
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

model = model_config()

diffusion = diffusion_config(model)

trainer = trainer_config(diffusion, dataset, renderer)


#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

utils.report_parameters(model)

print('Testing forward...', end=' ', flush=True)
batch = utils.batchify(dataset[0])
	# batch[0].shape: (384, 6)
	# batch[0] len : 384 (horizon)

	# batch[1]: 2 elements in dict: (0, 383) as initial, final states are fixed in n_horizon
	# {0: array([ 0.04206157, ...e=float32), 383: array([ 0.31135416, ...e=float32)}   , each with dim = (4, )

loss, _ = diffusion.loss(*batch)     #forward pass
loss.backward()                      #backward pass
print('✓')


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)    #n_steps_per_epoch: 10000, n_epochs: 200

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train(n_train_steps=args.n_steps_per_epoch)

