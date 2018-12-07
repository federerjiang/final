import torch
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np

import sys
sys.path.insert(0,'..')
from env_wrap import EnvWrap
from model import ActorCritic


def ensure_shared_grads(model, shared_model):
	for param, shared_param in zip(model.parameters(), shared_model.parameters()):
		shared_param.grad = param.grad 


def compute_loss(args, s_batch, a_batch, r_batch, done, model, entropy_coef):
	assert len(s_batch) == len(a_batch)
	assert len(s_batch) == len(r_batch)

	ba_size = len(s_batch)
	# print(ba_size)
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


def agent(rank, args, share_model):
	torch.manual_seed(args.seed+rank)
	video_file_id = rank % 3
	env = EnvWrap(video_file_id)

	model = ActorCritic()
	model.load_state_dict(share_model.state_dict())

	lr = args.lr
	entropy_coef = args.entropy_coef
	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
	model.train()

	state = env.reset()
	action = 0

	s_batch = [state]
	a_batch = [action]
	r_batch = []

	count = 0
	while True:
		model.load_state_dict(share_model.state_dict())

		reward_gop = 0
		while True:
				_, end_of_video, decision_flag = env.step_gop(action)
				if decision_flag or end_of_video:
					reward_gop = env.get_reward_gop()
					state = env.get_state_gop()
					break
				else:
					continue

		r_batch.append(reward_gop)

		logit, value = model(torch.FloatTensor(state).view(-1, args.s_gop_info, args.s_gop_len))
		prob = F.softmax(logit, dim=1)
		action = prob.multinomial(1).data.numpy()[0][0]
		done = end_of_video
		
		if len(r_batch) >= args.max_update_step or done:
			loss = compute_loss(args, s_batch, a_batch, r_batch, done, model, entropy_coef)
			if count >= 50000:
				optimizer = optim.Adam(model.parameters(), lr= 1e-5, weight_decay=1e-5)
				entropy_coef = 1
			if count >= 70000:
				entropy_coef = 0.5
			if count >= 100000:
				entropy_coef = 0.3
			if count >= 150000:
				entropy_coef = 0.1

			optimizer.zero_grad()
			loss.backward(retain_graph=True)
			torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
			# ensure_shared_grads(model, share_model)
			optimizer.step()
			model.zero_grad()			
			share_model.load_state_dict(model.state_dict())
			print('update model')

			del s_batch[:]
			del a_batch[:]
			del r_batch[:]

		if done:
			state = env.reset()
			action = 0

		s_batch.append(state)
		a_batch.append(action) 
