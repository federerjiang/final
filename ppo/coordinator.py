import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np 

import sys
sys.path.insert(0,'..')
# from model_fft import ActorCritic
from model import ActorCritic

# coordinator is only for updating model's weights based on coolected experiences from exp_queues
def update(args, optimizer, model, states, returns, actions, values, old_action_log_probs, advantages, clip_param):
	ba_size = states.shape[0]
	logits, vpreds = model(states, batch_size=ba_size)
	probs = F.softmax(logits, dim=1)
	log_probs = F.log_softmax(logits, dim=1)
	action_probs = probs.gather(1, actions)
	action_log_probs = log_probs.gather(1, actions)
	entropies = -(action_probs * action_log_probs)
	dist_entropy = entropies.mean()

	ratio = torch.exp(action_log_probs - old_action_log_probs)
	surr1 = ratio * advantages
	surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
	action_loss = -torch.min(surr1, surr2).mean()

	if args.use_clipped_value_loss:
		value_pred_clipped = values + (vpreds - values).clamp(-clip_param, clip_param)
		values_losses = (vpreds - returns).pow(2)
		values_losses_clipped = (value_pred_clipped - returns).pow(2)
		value_loss = 0.5 * torch.max(values_losses, values_losses_clipped).mean()
	else:
		value_loss = 0.5 * (returns - vpreds).pow(2).mean()

	optimizer.zero_grad()
	(value_loss * args.value_loss_coef + action_loss - dist_entropy * args.entropy_coef).backward()
	torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
	optimizer.step()


def generator(args, states_batch, returns_batch, actions_batch, values_batch, old_action_log_probs_batch, advantages_batch):
	# number of samples from all agents
	batch_size = states_batch.shape[0]
	assert batch_size >= args.num_mini_batch
	mini_batch_size = batch_size // args.num_mini_batch
	sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
	for indices in sampler:
		states = states_batch[indices]
		returns = returns_batch[indices]
		actions = actions_batch[indices]
		values = values_batch[indices]
		old_action_log_probs = old_action_log_probs_batch[indices]
		advantages = advantages_batch[indices]

		yield states, returns, actions, values, old_action_log_probs, advantages


def coordinator(rank, args, share_model, exp_queues, model_params):
	assert len(exp_queues) == args.num_processes

	model = ActorCritic()
	model.train()
	# model.load_state_dict(share_model.state_dict())

	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
	entropy_coef = args.entropy_coef

	count = 0
	while True:
		count += 1

		for i in range(args.num_processes):
			model_params[i].put(model.state_dict())

		states_batch = torch.ones((1, 7, 16))
		returns_batch = torch.ones((1, 1))
		actions_batch = torch.ones((1, 1), dtype=torch.long)
		values_batch = torch.ones((1, 1))
		old_action_log_probs_batch = torch.ones((1, 1))
		advantages_batch = torch.ones((1, 1))
		# assemble experiences from the agents
		for i in range(args.num_processes):
			states, returns, actions, values, action_log_probs, advantages = exp_queues[i].get()
			states_batch = torch.cat((states_batch, states), 0)
			returns_batch = torch.cat((returns_batch, returns), 0)
			actions_batch = torch.cat((actions_batch, actions), 0)
			values_batch = torch.cat((values_batch, values), 0)
			old_action_log_probs_batch = torch.cat((old_action_log_probs_batch, action_log_probs), 0)
			advantages_batch = torch.cat((advantages_batch, advantages), 0)

		states_batch = states_batch[1:]
		returns_batch = returns_batch[1:]
		actions_batch = actions_batch[1:]
		values_batch = values_batch[1:]
		old_action_log_probs_batch = old_action_log_probs_batch[1:]
		advantages_batch = advantages_batch[1:]

		print(states_batch.shape[0])
		# print(returns_batch.shape)
		# print(states_batch)

		data_generator = generator(args, states_batch, returns_batch, actions_batch, values_batch, old_action_log_probs_batch, advantages_batch)
		clip_param = args.clip_param * (1 - count / 5e4)
		for sample in data_generator:
			states, returns, actions, values, old_action_log_probs, advantages = sample
					# print(states.shape, returns.shape, actions.shape, values.shape, old_action_log_probs.shape, advantages.shape)

			update(args, optimizer, model, states, returns, actions, values, old_action_log_probs, advantages, clip_param)
			# 

		del states_batch
		del returns_batch
		del actions_batch
		del values_batch
		del old_action_log_probs_batch
		del advantages_batch

		print('update model parameters ', count)
		share_model.load_state_dict(model.state_dict())
		# break
	