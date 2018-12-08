import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 

import sys
sys.path.insert(0,'..')
# from model_fft import ActorCritic
from model import ActorCritic

# coordinator is only for updating model's weights based on coolected experiences from exp_queues
def compute_loss(args, s_batch, a_batch, r_batch, done, model, entropy_coef):
	assert len(s_batch) == len(a_batch)
	assert len(s_batch) == len(r_batch)

	ba_size = len(s_batch)
	s_batch = torch.FloatTensor(s_batch).view(-1, args.s_gop_info, args.s_gop_len)
	logits, v_batch = model(s_batch, batch_size=ba_size)
	r_batch = torch.FloatTensor(r_batch).view(ba_size, -1)
	R_batch = torch.zeros(r_batch.shape)
	a_batch = torch.LongTensor(a_batch).view(ba_size, -1)

	if done:
		R_batch[-1, 0] = 0
	else:
		R_batch[-1, 0] = v_batch[-1, 0]

	if args.use_gae:
		gae = 0
		for t in reversed(range(ba_size - 1)):
			delta = r_batch[t, 0] + args.gamma * v_batch[t+1, 0] - v_batch[t, 0]
			gae = delta + args.gamma * args.tau * gae
			R_batch[t, 0] = gae + v_batch[t, 0]
	else:	
		for t in reversed(range(ba_size - 1)):
			R_batch[t, 0] = r_batch[t] + args.gamma * R_batch[t+1, 0]

	adv_batch = R_batch - v_batch
	
	probs = F.softmax(logits, dim=1)
	log_probs = F.log_softmax(logits, dim=1)

	action_probs = probs.gather(1, a_batch)
	action_log_probs = log_probs.gather(1, a_batch)
	entropies = -(action_probs * action_log_probs)

	value_loss = adv_batch.pow(2).mean()
	policy_loss = -(adv_batch.detach() * action_log_probs).mean()
	dist_entropy = entropies.mean()

	loss = value_loss * args.value_loss_coef + policy_loss - dist_entropy * entropy_coef
	return loss


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
		if count >=10:
			entropy_coef = 1
		if count >= 20:
			entropy_coef = 0.5
		if count >= 25:
			entropy_coef = 0.1

		for i in range(args.num_processes):
			model_params[i].put(model.state_dict())
		# assemble experiences from the agents
		for i in range(args.num_processes):
			s_batch, a_batch, r_batch, done = exp_queues[i].get()
			loss = compute_loss(args, s_batch, a_batch, r_batch, done, model, entropy_coef)
			optimizer.zero_grad()
			loss.backward(retain_graph=True)
			torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
			optimizer.step()
		print('update model parameters ', count)

		share_model.load_state_dict(model.state_dict())
	