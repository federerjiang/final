class Args:
	def __init__(self):
		# model training parameters
		self.max_update_step = 20
		self.lr = 1e-4 # 
		self.use_gae = True
		self.gae = 0.95
		self.gamma = 0.99 #
		self.tau = 1.00 #
		self.entropy_coef = 3 #
		self.value_loss_coef = 0.5 #
		self.max_grad_norm = 0.1 #
		self.num_processes = 9
		self.seed = 30 #
		self.s_gop_info = 7
		self.s_gop_len = 16


