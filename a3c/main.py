import torch
import torch.multiprocessing as mp 
import os

from args import Args 
from agent import agent

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
	model.share_memory()

	processes = []

	# p = mp.Process(target=test, args=(env_args, model, 'A3C', 1))
	# p.start()
	# processes.append(p)

	for rank in range(0, args.num_processes):
		p = mp.Process(target=agent, args=(rank, args, model))
		p.start()
		processes.append(p)
	for p in processes:
		p.join()
	