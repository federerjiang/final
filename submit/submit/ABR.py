import numpy as np 
import torch
import torch.nn as nn 
import torch.nn.functional as F 

NN_MODEL = "/home/team/上山打老虎/submit/model/actor.pt" # model path settings
NN_MODEL_2 = "/home/team/上山打老虎/submit/model/actor.pt-xlow" # model path settings

class Algorithm:

	def __init__(self):
		
		self.model = ActorCritic()
		self.model.load_state_dict(torch.load(NN_MODEL))

		self.model_2 = ActorCritic()
		self.model_2.load_state_dict(torch.load(NN_MODEL_2))

		self.action_map = self._set_action_map()

		# state info for last gop
		self.state_gop = np.zeros((7, 16))
		self.gop_size = 0
		self.next_gop_sizes = 0
		self.gop_delay = 0
		self.time = 0
		self.buffer_size = 0
		self.buffer_flag = False
		self.cdn_flag = False
		self.frame_thps = [0] * 16
		self.cdn_flags = 0

		self.gop_count = 0
		self.thps = 0

		self.last_bit_rate = 0
		self.bitrates = [500.0, 850.0, 1200.0, 1850.0]

		# info of the simulator
		self.frame_len = 0.04
		self.gop_len = 50


	@staticmethod
	def _set_action_map():
		bit_rate_levels = [0, 1, 2, 3]
		target_buffer_levels = [0, 1]
		action_map = []
		for bitrate_idx in range(len(bit_rate_levels)):
			for target_buffer_idx in range(len(target_buffer_levels)):
				action_map.append((bit_rate_levels[bitrate_idx], target_buffer_levels[target_buffer_idx]))
		return action_map

		
	# predict next gop sizes based on past gop sizes
	def _predict_gop_sizes(self):
		if self.last_bit_rate == 0:
			return self.gop_size * np.array([1, 1.7, 1.7*1.4, 1.7*1.4*1.5])
		elif self.last_bit_rate == 1:
			return self.gop_size * np.array([1/1.7, 1, 1.4, 1.4*1.5])
		elif self.last_bit_rate == 2:
			return self.gop_size * np.array([1/(1.7*1.4), 1/1.4, 1, 1.5])
		else:
			return self.gop_size * np.array([1/(1.7*1.4*1.5), 1/(1.4*1.5), 1/1.5, 1])
	

	def _get_next_gop_sizes(self, cdn_has_frame):
		gop_flags = cdn_has_frame[4]

		# if there are no frames on cdn, predict future gop sizes
		len_cdn = len(cdn_has_frame[4])
		if len_cdn == 0:
			return self._predict_gop_sizes()

		# if there are more than one gop on cdn
		total_gop_flag = False
		gop_start_id = 0
		gop_end_id = 0
		for i in range(0, len(gop_flags)):
			if gop_flags[i] == 1:
				gop_start_id = i
				break
		for i in range(gop_start_id+1, len(gop_flags)):
			if gop_flags[i] == 1: 
				total_gop_flag = True
				gop_end_id = i
				break

		# if there is a whole gop on cdn, or otherwise
		if total_gop_flag == True:
			next_frame_sizes = np.array(cdn_has_frame)[:4, gop_start_id:gop_end_id]
			next_gop_sizes = np.sum(next_frame_sizes, axis=1)
			return next_gop_sizes
		else:
			next_I_frame_sizes = np.array(cdn_has_frame)[:4, gop_start_id:gop_start_id+1].reshape(4)
			next_P_frame_average = np.mean(np.array(cdn_has_frame)[:4, gop_start_id:-1], axis=1)
			if len(cdn_has_frame[4]) == 1:
				next_P_frame_average = np.array([5000, 5000*1.7, 5000*1.7*1.4, 5000*1.7*1.4*1.5])
			next_gop_sizes = next_I_frame_sizes + next_P_frame_average * 49
			return next_gop_sizes


	def _update_last_gop_info(self, time, S_time_interval, \
							S_send_data_size, \
							S_chunk_len, \
							S_rebuf, \
							S_buffer_size, \
							S_play_time_len, \
							S_end_delay, \
							S_decision_flag, \
							S_buffer_flag, \
							S_cdn_flag):

		assert S_decision_flag[-1] == True

		if S_decision_flag.count(True) <= 1:
			self.gop_len = 50
		else:
			S_decision_flag[-1] = False
			count = 0
			for flag in reversed(S_decision_flag):
				count += 1
				if flag == True:
					break
			self.gop_len = count - 1
			# print('gop len is: ', self.gop_len)

		self.gop_delay = np.array(S_end_delay[-50:]).sum()
		self.gop_size = np.array(S_send_data_size[-50:]).sum()
		self.cdn_flags = S_cdn_flag[-16:]
		for i in range(16):
			if self.cdn_flags == 0:
				self.cdn_flags[i] = 0
			else:
				self.cdn_flags[i] = 1
		# self.gop_buffer_cnt = S_buffer_flag[-self.gop_len:].count(True)
		# self.cdn_cnt = S_cdn_flag[-self.gop_len:].count(True)

		# update history throughputs, every 0.5s
		thp_count = 0
		count = 0
		last_frame_thp = 0
		for cdn_flag, send_data_size, time_interval in zip(reversed(S_cdn_flag), reversed(S_send_data_size), reversed(S_time_interval)):
			if not cdn_flag and time_interval > 0:
				frame_thp = send_data_size / time_interval
				if frame_thp != last_frame_thp:
					self.frame_thps.insert(0, frame_thp / 1000000)
					del self.frame_thps[-1]
					last_frame_thp = frame_thp
					thp_count += 1
					if thp_count >= 16:
						# print('update frame thps')
						# print(self.frame_thps)
						break
			count += 1
			if count >= 200:
				break
		# self.frame_thps = self.frame_thps[0:16]
		# self.thps = []
		self.time = time
		self.buffer_size = S_buffer_size[-1]
		self.buffer_flag = S_buffer_flag[-1]
		self.cdn_flag = S_cdn_flag[-1]


	 # Intial 
	def Initial(self):
		IntialVars = []
		return IntialVars


	#Define your algorithm
	def run(self, time, S_time_interval, S_send_data_size, S_chunk_len, S_rebuf, S_buffer_size, S_play_time_len, S_end_delay, S_decision_flag, S_buffer_flag, S_cdn_flag, end_of_video, cdn_newest_id, download_id, cdn_has_frame, IntialVars):

		if end_of_video:
			self.state_gop = np.zeros((7, 16))
			self.gop_count = 0
			return 0, 0
		else:
			self.gop_count += 1
			self._update_last_gop_info(time, S_time_interval, S_send_data_size, \
							S_chunk_len, S_rebuf, S_buffer_size, S_play_time_len, \
							S_end_delay, S_decision_flag, S_buffer_flag, S_cdn_flag)

			self.next_gop_sizes = self._get_next_gop_sizes(cdn_has_frame)

			# update gop state info
			self.state_gop = np.roll(self.state_gop, -1, axis=1)
			self.state_gop[0, -1] = self.buffer_size # current buffer size [0, 10] [fc]
			self.state_gop[1, -1] = self.bitrates[self.last_bit_rate] / 1000 # last bitrate [0, 2] [fc]
			# self.state_gop[2, -1] = self.gop_size / 1000000 / max(self.gop_time_interval, 1e-6) # last throughput Mbps [0, 10] [conv]
			self.state_gop[2, :] = self.frame_thps # last throughput Mbps [0, 10] [conv]
			self.state_gop[3, -1] = self.gop_delay / 100 # gop delay (100ms) [conv]
			self.state_gop[4, -1] = (1 if self.buffer_flag else 0) # if True, no buffering content, should choose target buffer as 0. [fc]
			# self.state_gop[5, -1] = (1 if self.cdn_flag else 0) # if True, cdn has no content. [fc]
			self.state_gop[5, :] = self.cdn_flags
			self.state_gop[6, :4] = self.next_gop_sizes / 1000000 # gop size (Mb) [0, 10] [conv]

			# print(self.state_gop)
		if np.mean(self.frame_thps) < 0.7 and np.std(self.frame_thps) < 0.1:
			# print('detec extreme low')
			logit, _ = self.model_2(torch.FloatTensor(self.state_gop).view(-1, 7, 16))
		else:
			logit, _ = self.model(torch.FloatTensor(self.state_gop).view(-1, 7, 16))
		# logit, _ = self.model(torch.FloatTensor(self.state_gop).view(-1, 7, 16))
		prob = F.softmax(logit, dim=1)
		_, action = torch.max(prob, 1)

		bitrate, target_buffer = self.action_map[action]

		self.last_bit_rate = bitrate

		return bitrate, target_buffer


class ActorCritic(nn.Module):
	def __init__(self, a_dim=8):
		super(ActorCritic, self).__init__()
		self.a_dim = a_dim
		
		# actor model
		self.a_fc0 = nn.Linear(1, 128) # buffer size 1
		self.a_fc1 = nn.Linear(1, 128) # last bit rate 1
		self.a_conv2 = nn.Conv1d(1, 128 ,4) # throughput 16
		self.a_conv3 = nn.Conv1d(1, 128, 4) # delay 16
		self.a_fc4 = nn.Linear(1, 128) # client rebuffer flag 1
		# self.a_fc5 = nn.Linear(1, 128) # cdn rebuffer flag 1
		self.a_conv5 = nn.Conv1d(1, 128, 4) # cdn rebuffer flag
		self.a_conv6 = nn.Conv1d(1, 128, 3) # next gop sizes 4
		# self.a_fc = nn.Linear(32*128, 128)
		self.a_fc = nn.Linear(44 * 128, 128)
		self.a_actor_linear = nn.Linear(128, self.a_dim)

		# critic model
		self.c_fc0 = nn.Linear(1, 128) # buffer size 1
		self.c_fc1 = nn.Linear(1, 128) # last bit rate 1
		self.c_conv2 = nn.Conv1d(1, 128, 4)	# throughput 16
		self.c_conv3 = nn.Conv1d(1, 128, 4) # delay 16
		self.c_fc4 = nn.Linear(1, 128) # client rebuffer flag 1
		# self.c_fc5 = nn.Linear(1, 128) # cdn rebuffer flag 1
		self.c_conv5 = nn.Conv1d(1, 128, 4) # cdn rebuffer flag
		self.c_conv6 = nn.Conv1d(1, 128, 3) # next gop sizes 4
		# self.c_fc = nn.Linear(32*128, 128)
		self.c_fc = nn.Linear(44 * 128, 128)
		self.c_critic_linear = nn.Linear(128, 1)


	def forward(self, inputs, batch_size=1):
		# actor
		split_0 = F.relu(self.a_fc0(inputs[:, 0:1, -1]))
		split_1 = F.relu(self.a_fc1(inputs[:, 1:2, -1]))
		split_2 = F.relu(self.a_conv2(inputs[:, 2:3, 0:16])).view(batch_size, -1)
		split_3 = F.relu(self.a_conv3(inputs[:, 3:4, 0:16])).view(batch_size, -1)
		split_4 = F.relu(self.a_fc4(inputs[:, 4:5, -1]))
		# split_5 = F.relu(self.a_fc5(inputs[:, 5:6, -1]))
		split_5 = F.relu(self.a_conv5(inputs[:, 5:6, 0:16])).view(batch_size, -1)
		split_6 = F.relu(self.a_conv6(inputs[:, 6:7, :4])).view(batch_size, -1)

		merge = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5, split_6), 1)
		merge = merge.view(batch_size, -1)
		fc_out = F.relu(self.a_fc(merge))
		logit = self.a_actor_linear(fc_out)

		# critic
		split_0 = F.relu(self.c_fc0(inputs[:, 0:1, -1]))
		split_1 = F.relu(self.c_fc1(inputs[:, 1:2, -1]))
		split_2 = F.relu(self.c_conv2(inputs[:, 2:3, 0:16])).view(batch_size, -1)
		split_3 = F.relu(self.c_conv3(inputs[:, 3:4, 0:16])).view(batch_size, -1)
		split_4 = F.relu(self.c_fc4(inputs[:, 4:5, -1]))
		# split_5 = F.relu(self.c_fc5(inputs[:, 5:6, -1]))
		split_5 = F.relu(self.c_conv5(inputs[:, 5:6, 0:16])).view(batch_size, -1)
		split_6 = F.relu(self.c_conv6(inputs[:, 6:7, 0:4])).view(batch_size, -1)

		merge = torch.cat((split_0, split_1, split_2, split_3, split_4, split_5, split_6), 1)
		merge = merge.view(batch_size, -1)
		fc_out = F.relu(self.c_fc(merge))
		v = self.c_critic_linear(fc_out)

		return logit, v