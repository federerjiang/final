# this file is for evaluating trained models
# based on online.py

import time
import numpy as np 

import torch
import torch.nn.functional as F 

from fixed_env_wrap import FixedEnvWrap 
from model import ActorCritic
from env_args import EnvArgs


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


def test(args, model, bw_trace, video_file_id=3):
	action_map = _set_action_map()

	env = FixedEnvWrap(video_file_id, bw_trace)

	model.eval()

	state = env.reset()

	start = time.time()
	while True:
		video_count = 0
		reward_all_sum = 0
		reward_all = 0
		reward_all_ave = 0
		reward_gop = 0
		action = 0
		last_action = 0

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
				
			state = torch.FloatTensor(state)
			logit, _ = model(state.view(-1, args.s_gop_info, args.s_gop_len))
			prob = F.softmax(logit, dim=1)
			_, action = torch.max(prob, 1)
			action = action.data.numpy()[0]

			bitrate, target_buffer = action_map[last_action]
			# print('bitrate: %d, target_buffer: %d, reward is %s' % (bitrate, target_buffer, reward_gop))
			if done:
				print("video count %d, reward is %.5f" % (video_count, reward_all))
				reward_all_sum += reward_all / 100
				video_count += 1
				# if reward_all < 0:
					# print('bad model ! just break this loop')
					# reward_all_ave = 0
					# break 
				if video_count >= env.traces_len:
					reward_all_ave = reward_all_sum / video_count
					break
				action = 0
				last_action = 0
				reward_all = 0

			reward_all += reward_gop

		end = time.time()
		hours, rem = divmod(end-start, 3600)
		minutes, seconds = divmod(rem, 60)
		print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
		print("average reward of traces are: ", reward_all_ave)


if __name__ == '__main__':
	model = ActorCritic()
	model.load_state_dict(torch.load('/Users/federerjiang/research-project/aitrans-competition/final/a2c/seletec_result/actor.pt-1516'))
	args = EnvArgs()
	test(args, model, 'eval-test')





