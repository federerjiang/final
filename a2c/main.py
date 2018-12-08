import torch
import torch.multiprocessing as mp
# import multiprocessing as mp 
import os

from args import Args 
from agent import agent
from coordinator import coordinator

import sys
sys.path.insert(0,'..')
from model import ActorCritic
from test import test


if __name__ == '__main__':
	os.environ['OMP_NUM_THREADS'] = '1'
	torch.set_num_threads(1)

	args = Args()
	torch.manual_seed(args.seed)
	
	model = ActorCritic()
	model.load_state_dict(torch.load('result/actor.pt-1516'))
	model.share_memory()

	# inter-process communication queues
	exp_queues = []
	model_params = []
	for i in range(args.num_processes):
		exp_queues.append(mp.Queue(1))
		model_params.append(mp.Queue(1))

	p = mp.Process(target=test, args=(args, model, 'a2c', 1))
	p.start()

	# creat a process for coordinator
	coordinator = mp.Process(target=coordinator, args=(args.num_processes, args, model, exp_queues, model_params))
	coordinator.start()

	# create processes for multiple agents
	for rank in range(0, args.num_processes):
		p = mp.Process(target=agent, args=(rank, args, exp_queues[rank], model_params[rank]))
		p.start()

	# wait until training is done
	coordinator.join()
	