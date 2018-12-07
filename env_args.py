class EnvArgs:
	def __init__(self):
		# env settings
		self.s_gop_len = 16 # gop based info
		self.s_gop_info = 7
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

		self.train_bw_trace = '../trace/network_trace/mix_trace/'
		self.test_bw_trace = '../trace/test/'
		self.video_size_files = \
		['../trace/video_trace/AsianCup_China_Uzbekistan/frame_trace_',
		'../trace/video_trace/Fengtimo_2018_11_3/frame_trace_',
		'../trace/video_trace/YYF_2018_08_12/frame_trace_']