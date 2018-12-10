class Args:
	def __init__(self):
		# model training parameters
		self.max_update_step = 21
		self.lr = 1e-4 # 
		self.use_gae = True
		self.gae = 0.95
		self.gamma = 0.99 #
		self.tau = 1.00 #
		self.num_mini_batch = 24
		self.clip_param = 0.2
		self.use_clipped_value_loss = True
		# self.entropy_coef = 0.5 # pretrain start from 1
		self.entropy_coef = 0.01 # begin from 3, then gradually reduce to 0.1
		self.value_loss_coef = 1 #
		self.max_grad_norm = 0.1 #
		self.num_processes = 12
		self.seed = 30 #
		self.s_gop_info = 7 # or 7
		self.s_gop_len = 16


