class EnvArgs:
	def __init__(self):
		# env settings
		self.s_gop_len = 16 # gop based info
		self.s_gop_info = 7 # or 7
		self.a_dim = 8
		self.random_seed = 50
		self.bitrate_levels = 4
		self.bitrate = [500.0, 850.0, 1200.0, 1850.0] # kbps
		self.target_buffer_levels = 2
		self.target_buffer = [2.0, 3.0] # seconds

		self.frame_time_len = 0.04
		self.smooth_penalty = 0.02
		self.rebuf_penalty = 1.5
		self.latency_penalty = 0.005

		self.bw_trace = {'mix-hml': '../trace/network_trace/mix_trace_h-m-l/',
						'mix-mlx': '../trace/network_trace/mix_trace_m-l-xl/',
						'mix-lx': '../trace/network_trace/mix_trace_l-xl/',
						'cooked': '../trace/cooked_trace/',
						'new': '../trace/new_network_trace/',
						'high': '../trace/network_trace/high/',
						'middle': '../trace/network_trace/middle/',
						'low': '../trace/network_trace/low/',
						'xlow': '../trace/network_trace/xlow/',
						'test-hml': '../trace/test-hml/',
						'test-mlx': '../trace/test-mlx/',
						'test-lx': '../trace/test-lx/',
						'test-xlow': '../trace/test-xlow/',
						'eval-high': 'trace/network_trace/high/',
						'eval-middle': 'trace/network_trace/middle/',
						'eval-low': 'trace/network_trace/low/',
						'eval-test': 'trace/test/'}

		self.video_size_files_room = \
		['../trace/video_trace/room/Fengtimo_2018_11_3/frame_trace_',
		'../trace/video_trace/room/room_1/frame_trace_',
		'../trace/video_trace/room/room_2/frame_trace_',
		'../trace/video_trace/room/room_3/frame_trace_',
		'../trace/video_trace/room/room_4/frame_trace_',
		'../trace/video_trace/room/room_5/frame_trace_',]

		self.video_size_files_sports = \
		['../trace/video_trace/sports/AsianCup_China_Uzbekistan/frame_trace_',
		'../trace/video_trace/sports/sports_1/frame_trace_',
		'../trace/video_trace/sports/sports_2/frame_trace_',
		'../trace/video_trace/sports/sports_3/frame_trace_',
		'../trace/video_trace/sports/sports_4/frame_trace_',
		'../trace/video_trace/sports/sports_5/frame_trace_',]

		self.video_size_files_game = \
		['../trace/video_trace/game/YYF_2018_08_12/frame_trace_',
		'../trace/video_trace/game/game_1/frame_trace_',
		'../trace/video_trace/game/game_2/frame_trace_',
		'../trace/video_trace/game/game_3/frame_trace_',
		'../trace/video_trace/game/game_4/frame_trace_',
		'../trace/video_trace/game/game_5/frame_trace_',]

		# ['../trace/video_trace/sports/AsianCup_China_Uzbekistan/frame_trace_',
		# '../trace/video_trace/room/Fengtimo_2018_11_3/frame_trace_',
		# '../trace/video_trace/game/YYF_2018_08_12/frame_trace_',
		# 'trace/video_trace/Fengtimo_2018_11_3/frame_trace_',
		# 'trace/video_trace/AsianCup_China_Uzbekistan/frame_trace_',
		# 'trace/video_trace/YYF_2018_08_12/frame_trace_']