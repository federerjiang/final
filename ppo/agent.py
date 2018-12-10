import torch
import torch.nn.functional as F 
# import torch.optim as optim
import numpy as np


import sys
sys.path.insert(0,'..')
from env_wrap import EnvWrap
from model import ActorCritic
# from env_wrap_fft import EnvWrap
# from model_fft import ActorCritic

def collect_info(args, s_batch, a_batch, r_batch, done, model):
	assert len(s_batch) == len(a_batch)
	assert len(s_batch) == len(r_batch)

	ba_size = len(s_batch)
	s_batch = torch.FloatTensor(s_batch).view(-1, args.s_gop_info, args.s_gop_len)
	with torch.no_grad():
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

	log_probs = F.log_softmax(logits, dim=1)
	action_log_probs = log_probs.gather(1, a_batch)

	adv_batch = R_batch - v_batch

	return s_batch, R_batch, a_batch, v_batch, action_log_probs, adv_batch

def agent(rank, args, exp_queue, model_param):
	video_file_id = rank % 3
	# env = EnvWrap(video_file_id, bw_trace='low')
	env = EnvWrap(video_file_id) # bw_trace default is mix

	model = ActorCritic()
	model.load_state_dict(model_param.get())
	model.eval()

	state = env.reset()
	action = 0

	s_batch = [state]
	a_batch = [action]
	r_batch = []

	while True:
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

		with torch.no_grad():
			logit, value = model(torch.FloatTensor(state).view(-1, args.s_gop_info, args.s_gop_len))
			prob = F.softmax(logit, dim=1)
			action = prob.multinomial(1).data.numpy()[0][0]
		
		done = end_of_video 

		if len(r_batch) >= args.max_update_step or done:
			if len(s_batch) >= 5:
				states, returns, actions, values, action_log_probs, advantages = collect_info(args, s_batch[1:], a_batch[1:], r_batch[1:], done, model)
				exp_queue.put([states, returns, actions, values, action_log_probs, advantages])
				model.load_state_dict(model_param.get())
				del states
				del returns
				del actions
				del values
				del action_log_probs
				del advantages

			del s_batch[:]
			del a_batch[:]
			del r_batch[:]
			# print('agent finish work')

		if done:
			state = env.reset()
			action = 0
			
		s_batch.append(state)
		a_batch.append(action)