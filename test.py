# based on online.py

import time
import visdom
import numpy as np 

import torch
import torch.nn.functional as F 

from fixed_env_wrap import FixedEnvWrap 
from model import ActorCritic
# from fixed_env_wrap_fft import FixedEnvWrap 
# from model_fft import ActorCritic


def _set_action_map():
	"""
	Map action to (bitrate, target_buffer)
	
	Exp as below:
	action 		==>  	(bitrate, target_buffer)
	0					(0, 1)
	1					(0, 2)
	2					(1, 0)
	3					(1, 1)
	4					(2, 0)
	5					(2, 1)
	6					(3, 0)
	7					(3, 1)
	"""
	bit_rate_levels = [0, 1, 2, 3]
	target_buffer_levels = [0, 1]
	action_map = []
	for bitrate_idx in range(len(bit_rate_levels)):
		for target_buffer_idx in range(len(target_buffer_levels)):
			action_map.append((bit_rate_levels[bitrate_idx], target_buffer_levels[target_buffer_idx]))
	return action_map


def test(args, shared_model, alg, video_file_id=1):
	action_map = _set_action_map()

	env = FixedEnvWrap(video_file_id)

	# time.sleep(10)
	model = ActorCritic()
	model.load_state_dict(shared_model.state_dict())
	model.eval()

	state = env.reset()

	training_time = 0
	vis = visdom.Visdom(env='test')
	line_plot = vis.line(Y=np.array([0]), opts=dict(
						xlabel='testing count',
						ylabel='average reward',
						title=alg+'-cooked-v1'))

	start = time.time()
	vis_count = 0
	while True:
		video_count = 0
		reward_all_sum = 0
		reward_all = 0
		reward_all_ave = 0
		reward_gop = 0
		action = 0
		last_action = 0
		# update model before testing all trace files
		# time.sleep(5)
		# print('load updated model')
		model.load_state_dict(shared_model.state_dict())
		while True:
			# get the reward for one gop
			while True:
				_, done, decision_flag = env.step_gop(action)
				if decision_flag or done:
					reward_gop = env.get_reward_gop()
					state = env.get_state_gop()
					break
				else:
					continue
			# print('testing')
			# get action from model
			last_action = action
			with torch.no_grad():
				state = torch.FloatTensor(state)
				logit, _ = model(state.view(-1, args.s_gop_info, args.s_gop_len))
				prob = F.softmax(logit, dim=1)
				_, action = torch.max(prob, 1)
				action = action.data.numpy()[0]

			bitrate, target_buffer = action_map[last_action]
			# print('bitrate: %d, target_buffer: %d, reward is %s' % (bitrate, target_buffer, reward_gop))
			if done:
				# print("video count %d, reward is %.5f" % (video_count, reward_all))
				# reward_all_sum += reward_all / 100
				reward_all_sum += reward_all
				video_count += 1
				if reward_all < 0:
					# print('bad model ! just break this loop')
					reward_all_ave = 0
					break 
				if video_count >= env.traces_len:
					reward_all_ave = reward_all_sum / video_count
					break
				action = 0
				last_action = 0
				reward_all = 0

			reward_all += reward_gop

		# update the figure of average reward of all testing files
		vis_count += 1
		reward_all_ave = max(reward_all_ave, 0)
		vis.line(Y=np.array([reward_all_ave]), X=np.array([vis_count]), win=line_plot, update='append')
		path = 'result-cooked-v1/actor.pt-' + str(vis_count)
		torch.save(model.state_dict(), path)

		end = time.time()
		hours, rem = divmod(end-start, 3600)
		minutes, seconds = divmod(rem, 60)
		# if count > 20:
			# print9
		# print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
		# print("average reward of traces are: ", reward_all_ave)
		# print('saved one model in epoch:', vis_count)

# result-v3 
# 4375: 30.12764; 4203: 29.76445; 2463: 29.75238; 1907: 29.444
# 4376: 29.8; 2450: 29.84

# score:
# v1: 4375: 50.1903
# v2: 2450: 

# result-v2
# 773: 29.7444



