import LiveStreamingEnv.env as env
import LiveStreamingEnv.load_trace as load_trace
import numpy as np

from env_args import EnvArgs 

class EnvWrap(env.Environment):
	
	def __init__(self, video_file_id):
		self.args = EnvArgs()
		all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(self.args.train_bw_trace)
		super().__init__(all_cooked_time=all_cooked_time,
						 all_cooked_bw=all_cooked_bw,
						 random_seed=self.args.random_seed,
						 VIDEO_SIZE_FILE =self.args.train_video_size_files[video_file_id], 
						 logfile_path='./log/', 
						 Debug=False)
		
		# self.state = np.zeros((args.s_info, args.s_len)) # state info for past frames
		self.state_gop = np.zeros((self.args.s_gop_info, self.args.s_gop_len)) # state info for past gops
		self.last_bit_rate = 0
		self.reward_gop = 0
		self.last_reward_gop = 0
		self.action_map = self._set_action_map()

		# state info for gop
		self.gop_time_interval = 0
		self.gop_size = 0
		self.next_gop_sizes = 0
		self.gop_delay = 0

		# info for traces
		self.traces_len = len(all_file_names)


	@staticmethod
	def _set_action_map():
		bit_rate_levels = [0, 1, 2, 3]
		target_buffer_levels = [0, 1]
		action_map = []
		for bitrate_idx in range(len(bit_rate_levels)):
			for target_buffer_idx in range(len(target_buffer_levels)):
				action_map.append((bit_rate_levels[bitrate_idx], target_buffer_levels[target_buffer_idx]))
		return action_map


	# @staticmethod
	def _predict_gop_sizes(self):
		# predict next gop sizes based on past gop sizes
		if self.last_bit_rate == 0:
			return self.gop_size * np.array([1, 1.7, 1.7*1.4, 1.7*1.4*1.5])
		elif self.last_bit_rate == 1:
			return self.gop_size * np.array([1/1.7, 1, 1.4, 1.4*1.5])
		elif self.last_bit_rate == 2:
			return self.gop_size * np.array([1/(1.7*1.4), 1/1.4, 1, 1.5])
		else:
			return self.gop_size * np.array([1/(1.7*1.4*1.5), 1/(1.4*1.5), 1/1.5, 1])

	# @staticmethod
	def _get_next_gop_sizes(self, cdn_has_frame):
		gop_flags = cdn_has_frame[4]
		# print(gop_flags)
		# if there are no frames on cdn, predict future gop sizes
		len_cdn = len(cdn_has_frame[4])
		if len_cdn == 0:
			# print('cdn has no frames')
			return self._predict_gop_sizes()

		# if there are more than one gop on cdn
		# assert gop_flags[0] == 1
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
		# print("gop start id is : %d" % (gop_start_id))
		# if total_gop_flag == False:
			# gop_end_id = len(gop_flags)
		# print("gop end id is : %d" % (gop_end_id))
		if total_gop_flag == True:
			next_frame_sizes = np.array(cdn_has_frame)[:4, gop_start_id:gop_end_id]
			next_gop_sizes = np.sum(next_frame_sizes, axis=1)
			# print('there is a whole gop on cdn')
			return next_gop_sizes
		else:
			next_I_frame_sizes = np.array(cdn_has_frame)[:4, gop_start_id:gop_start_id+1].reshape(4)
			next_P_frame_average = np.mean(np.array(cdn_has_frame)[:4, gop_start_id:-1], axis=1)
			# print(next_I_frame_sizes)
			# print(next_P_frame_average)
			if len(cdn_has_frame[4]) == 1:
				next_P_frame_average = np.array([5000, 5000*1.7, 5000*1.7*1.4, 5000*1.7*1.4*1.5])
			next_gop_sizes = next_I_frame_sizes + next_P_frame_average * 49
			# print('there is part of gop on cdn')
			# print(next_I_frame_sizes)
			# print(next_P_frame_average)
			# print(next_gop_sizes)
			return next_gop_sizes


	# return gop state
	def step_gop(self, action):
		bit_rate, target_buffer = self.action_map[action]
		time, time_interval, send_data_size, frame_len, rebuf,\
		buffer_size, play_time_len,end_delay, cdn_newest_id,\
		download_id, cdn_has_frame, decision_flag, buffer_flag,cdn_flag, end_of_video =\
		self.get_video_frame(bit_rate, target_buffer)

		# reward setting
		if not cdn_flag:
			reward_frame = self.args.frame_time_len * float(self.args.bitrate[bit_rate]) / 1000 \
							- self.args.rebuf_penalty * rebuf \
							- self.args.latency_penalty * end_delay
		else:
			reward_frame = -(self.args.rebuf_penalty * rebuf)

		if not decision_flag:

			self.reward_gop += reward_frame

			# collect frames info in last gop
			self.gop_time_interval += time_interval
			self.gop_size += send_data_size
			self.gop_delay += end_delay


		if decision_flag or end_of_video:
			reward_frame += -1 * self.args.smooth_penalty * (abs(self.args.bitrate[bit_rate] - self.args.bitrate[self.last_bit_rate]) / 1000)
			self.last_bit_rate = bit_rate
			self.last_reward_gop = self.reward_gop
			self.reward_gop = reward_frame
			
			# calculate next gop sizes for 4 bitrate levels [500k, 800k, 1200k, 1800k]
			self.next_gop_sizes = self._get_next_gop_sizes(cdn_has_frame)

			# collect gop state info
			self.state_gop = np.roll(self.state_gop, -1, axis=1)
			self.state_gop[0, -1] = buffer_size # current buffer size [0, 10] [fc]
			self.state_gop[1, -1] = self.args.bitrate[bit_rate] / 1000 # last bitrate [0, 2] [fc]
			self.state_gop[2, -1] = self.gop_size / 1000000 / max(self.gop_time_interval, 1e-6) # last throughput Mbps [0, 10] [conv]
			self.state_gop[3, -1] = self.gop_delay / 100 # gop delay (100ms) [conv]
			self.state_gop[4, -1] = (1 if buffer_flag else 0) # if True, no buffering content, should choose target buffer as 0. [fc]
			self.state_gop[5, -1] = (1 if cdn_flag else 0) # if True, cdn has no content. [fc]
			self.state_gop[6, :4] = self.next_gop_sizes / 1000000 # gop size (Mb) [0, 10] [conv]

			# reset gop info
			self.gop_time_interval = time_interval
			self.gop_size = send_data_size
			self.gop_delay = end_delay

		return reward_frame, end_of_video, decision_flag

	def get_reward_gop(self):
		return self.last_reward_gop

	def get_state_gop(self):
		return self.state_gop

	def reset(self):
		self.state_gop = np.zeros((self.args.s_gop_info, self.args.s_gop_len))

		self.last_bit_rate = 0
		self.reward_gop = 0
		self.last_reward_gop = 0
		self.gop_time_interval = 0
		self.gop_size = 0
		self.next_gop_sizes = 0
		self.gop_delay = 0

		return self.state_gop
