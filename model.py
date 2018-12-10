import torch
import torch.nn as nn
import torch.nn.functional as F 

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