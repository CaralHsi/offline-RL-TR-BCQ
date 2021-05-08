# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""The entry point for running experiments for collecting replay datasets.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

from absl import app
from absl import flags
import argparse

from batch_rl.baselines.agents import dqn_agent
from batch_rl.baselines.agents import quantile_agent
from batch_rl.baselines.agents import random_agent
from batch_rl.baselines.run_experiment import LoggedRunner

from dopamine.discrete_domains import run_experiment
import tensorflow as tf

parser = argparse.ArgumentParser(description='OffPolicyTrain')
parser.add_argument('--AGENT', default='DQN', type=str, help='Name of the agent.')
parser.add_argument('--BUFFER_SIZE', default=1000, type=int, help='replay buffer size')
parser.add_argument('--ATARI_ENV', default='Breakout0', type=str, help='name of the env')
parser.add_argument('--DATA_SET_MODE', default='POOR', type=str,
                    help='the quality of the dataset, you can choose from {ALL,'
                         'POOR, HIGH, MEDIUM}')

parser.add_argument('--checkpointDir', default='./data', type=str, help='Directory from which to load the '
                    'replay data')
parser.add_argument('--init_checkpoint_dir', default=None, help='Directory from which to load '
                    'the initial checkpoint before training starts.')
parser.add_argument('--buckets', default='./off_line_train', type=str, help='bucketDir, store off-line train data')
FLAGS = flags.FLAGS


def create_agent(sess, environment, agent, replay_log_dir, buffer_size, summary_writer=None):
  """Creates a DQN agent.

  Args:
    sess: A `tf.Session`object  for running associated ops.
    environment: An Atari 2600 environment.
    replay_log_dir: Directory to which log the replay buffers periodically.
    summary_writer: A Tensorflow summary writer to pass to the agent
      for in-agent training statistics in Tensorboard.

  Returns:
    A DQN agent with metrics.
  """
  if agent == 'DQN':
    agent = dqn_agent.LoggedDQNAgent
  elif agent == 'Quantile':
    agent = quantile_agent.LoggedQuantileAgent
  elif agent == 'Random':
    agent = random_agent.RandomAgent
  else:
    raise ValueError('{} is not a valid agent name'.format(agent))

  return agent(sess, num_actions=environment.action_space.n,
               replay_log_dir=replay_log_dir, summary_writer=summary_writer, replay_capacity=buffer_size)


def main(args):
  cfg = {
    'AGENT': args.AGENT,
    'BUFFER_SIZE': args.BUFFER_SIZE,
    'ATARI_ENV': args.ATARI_ENV[:-1],
    'NUM_EXPERIMENT': args.ATARI_ENV[-1],
    'DATA_SET_MODE': args.DATA_SET_MODE,

    'replay_dir': args.checkpointDir,
    'init_checkpoint_dir': args.init_checkpoint_dir,
    'base_dir': args.buckets

  }

  tf.logging.set_verbosity(tf.logging.INFO)
  replay_log_dir = os.path.join(cfg['replay_dir'],
                                 cfg['ATARI_ENV'] + str(cfg['NUM_EXPERIMENT']) + '/replay_logs')
  tf.logging.info('Saving replay buffer data to {}'.format(replay_log_dir))
  create_agent_fn = functools.partial(
    create_agent, agent=cfg['AGENT'], replay_log_dir=replay_log_dir,
    buffer_size=cfg['BUFFER_SIZE'])
  runner = LoggedRunner(cfg['replay_dir'], create_agent_fn, cfg=cfg, mode='generator')
  runner.run_experiment()


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
