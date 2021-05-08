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

r"""The entry point for running experiments with fixed replay datasets.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import argparse
import os
import numpy as np


from absl import flags

from batch_rl.fixed_replay import run_experiment, run_bail_experiment
from batch_rl.fixed_replay.agents import dqn_agent
from batch_rl.fixed_replay.agents import multi_head_dqn_agent
from batch_rl.fixed_replay.agents import quantile_agent
from batch_rl.fixed_replay.agents import rainbow_agent
from batch_rl.fixed_replay.agents import bcq_agent
from batch_rl.fixed_replay.agents import rem_bcq_agent
from batch_rl.fixed_replay.agents import bail
from batch_rl.fixed_replay.agents import off_rainbow

import tensorflow as tf


parser = argparse.ArgumentParser(description='OffPolicyTrain')
# parser.add_argument('--AGENT', default='REM_BCQ', type=str, help='Name of the agent.')
parser.add_argument('--BUFFER_SIZE', default=1000, type=int, help='replay buffer size')
parser.add_argument('--ATARI_ENV', default='Breakout.0.BCQ', type=str, help='name of the env')
parser.add_argument('--DATA_SET_MODE', default='POOR_last_0.6', type=str,
                    help='the quality of the dataset, you can choose from {ALL,'
                         'POOR, HIGH, MEDIUM}')
parser.add_argument('--checkpointDir', default='./data', type=str, help='Directory from which to load the '
                    'replay data')
parser.add_argument('--init_checkpoint_dir', default=None, help='Directory from which to load '
                    'the initial checkpoint before training starts.')
parser.add_argument('--buckets', default='./off_line_train', type=str, help='bucketDir, store off-line train data')

# argument for BAIL
parser.add_argument('--select_percentage', default=0.8, type=float, help='select_percentage of BAIL')
parser.add_argument('--only_cal_bc_data', default='True', type=str, help='only calculate the ba data without train BAIL')

FLAGS = flags.FLAGS


def create_agent(sess, environment, replay_data_dir, agent,
                 summary_writer=None, init_checkpoint_dir=None,
                 buffer_size=1000000, data_set_mode='ALL', select_percentage=0.3,
                 border=None):
  """Creates a DQN agent.

  Args:
    sess: A `tf.Session`object  for running associated ops.
    environment: An Atari 2600 environment.
    replay_data_dir: Directory to which log the replay buffers periodically.
    summary_writer: A Tensorflow summary writer to pass to the agent
      for in-agent training statistics in Tensorboard.

  Returns:
    A DQN agent with metrics.
  """
  # arguments for specific agent

  kwargs = {}
  kwargs['tf_device'] = '/gpu:0'

  if agent == 'DQN':
    agent = dqn_agent.FixedReplayDQNAgent
  elif agent == 'C51':
    agent = rainbow_agent.FixedReplayRainbowAgent
  elif agent == 'Quantile':
    kwargs['num_atoms'] = 200
    kwargs['optimizer'] = tf.train.AdamOptimizer(
      learning_rate=0.00005,
      epsilon=0.0003125
    )
    agent = quantile_agent.FixedReplayQuantileAgent
  elif agent == 'MultiHeadDQN':
    kwargs['num_heads'] = 200
    kwargs['transform_strategy'] = 'IDENTITY'
    kwargs['optimizer'] = tf.train.AdamOptimizer(
      learning_rate=0.00005,
      epsilon=0.0003125
    )
    agent = multi_head_dqn_agent.FixedReplayMultiHeadDQNAgent
  elif agent == 'REM':
    kwargs['num_heads'] = 200
    kwargs['transform_strategy'] = 'STOCHASTIC'
    kwargs['optimizer'] = tf.train.AdamOptimizer(
      learning_rate=0.00005,
      epsilon=0.0003125
    )
    agent = multi_head_dqn_agent.FixedReplayMultiHeadDQNAgent
  elif agent in ['BCQ', 'BAIL_BCQ_weighted']:
    kwargs['optimizer'] = tf.train.AdamOptimizer(
      learning_rate=0.00005,
      epsilon=0.0003125
    )
    kwargs['name'] = agent
    if agent == 'BAIL_BCQ_weighted':
      kwargs['border'] = border
    agent = bcq_agent.BCQAgent
  elif agent == 'REM_BCQ2':
    kwargs['num_heads'] = 200
    kwargs['transform_strategy'] = 'STOCHASTIC'
    kwargs['optimizer'] = tf.train.AdamOptimizer(
      learning_rate=0.00005,
      epsilon=0.0003125
    )
    agent = rem_bcq_agent.FixedReplayREMBCQAgent
  elif agent == 'REM_BCQ_average':
    kwargs['num_heads'] = 200
    kwargs['transform_strategy'] = 'IDENTITY'
    kwargs['optimizer'] = tf.train.AdamOptimizer(
      learning_rate=0.00005,
      epsilon=0.0003125
    )
    agent = rem_bcq_agent.FixedReplayREMBCQAgent
  elif agent == 'BAIL':
    kwargs['optimizer'] = tf.train.AdamOptimizer(
      learning_rate=0.00005,
      epsilon=0.0003125
    )
    kwargs['select_percentage'] = select_percentage
    agent = bail.FixedReplayBailAgent
  else:
    raise ValueError('{} is not a valid agent name'.format(FLAGS.agent_name))

  return agent(
    sess=sess,
    num_actions=environment.action_space.n,
    replay_data_dir=replay_data_dir,
    summary_writer=summary_writer,
    init_checkpoint_dir=init_checkpoint_dir,
    replay_capacity=buffer_size,
    data_set_mode=data_set_mode,
    **kwargs
  )


def main(args):
  cfg = {
    'AGENT': args.ATARI_ENV.split('.')[2],
    'BUFFER_SIZE': args.BUFFER_SIZE,
    'ATARI_ENV': args.ATARI_ENV.split('.')[0],
    'NUM_EXPERIMENT': args.ATARI_ENV.split('.')[1],
    'DATA_SET_MODE': args.DATA_SET_MODE,
    # 'checkpointDir_suffix': args.ATARI_ENV.split('.')[3],  # fixme: temporal used

    'select_percentage': args.select_percentage,
    'only_cal_bc_data': args.only_cal_bc_data,

    'replay_dir': args.checkpointDir,
    'init_checkpoint_dir': args.init_checkpoint_dir,
    'base_dir': args.buckets

  }

  np.random.seed(int(cfg['NUM_EXPERIMENT']))

  tf.logging.set_verbosity(tf.logging.INFO)
  replay_data_dir = os.path.join(cfg['replay_dir'],
                                 cfg['ATARI_ENV'] + str(cfg['NUM_EXPERIMENT']) + '/replay_logs')

  if cfg['AGENT'] != 'BAIL_BCQ_weighted':
    create_agent_fn = functools.partial(
        create_agent,
      replay_data_dir=replay_data_dir,
      agent=cfg['AGENT'],
      init_checkpoint_dir=cfg['init_checkpoint_dir'],
      buffer_size=cfg['BUFFER_SIZE'],
      data_set_mode=cfg['DATA_SET_MODE'],
      select_percentage=cfg['select_percentage']
    )
    Runner = getRunner(cfg['AGENT'])
    runner = Runner(cfg['base_dir'], create_agent_fn, cfg=cfg)
    runner.run_experiment()

  else:
    # assertion
    assert cfg['only_cal_bc_data'] == 'True', 'THE BAIL-BCQ is training. We do not need to do behavior cloning in BAIL'

    # # get the border
    # cfg['AGENT'] = 'BAIL'
    # create_agent_fn = functools.partial(
    #   create_agent,
    #   replay_data_dir=replay_data_dir,
    #   agent=cfg['AGENT'],
    #   init_checkpoint_dir=cfg['init_checkpoint_dir'],
    #   buffer_size=cfg['BUFFER_SIZE'],
    #   data_set_mode=cfg['DATA_SET_MODE'],
    #   select_percentage=cfg['select_percentage']
    # )
    # Runner = getRunner(cfg['AGENT'])
    # runner_bail = Runner(cfg['base_dir'], create_agent_fn, cfg=cfg)
    # border = runner_bail.run_experiment(ret_border=True)
    # # clear all tensorflow graph
    # tf.reset_default_graph()

    border = 1.0

    # BCQ training stage
    cfg['AGENT'] = 'BAIL_BCQ_weighted'
    create_agent_fn = functools.partial(
      create_agent,
      replay_data_dir=replay_data_dir,
      agent=cfg['AGENT'],
      init_checkpoint_dir=cfg['init_checkpoint_dir'],
      buffer_size=cfg['BUFFER_SIZE'],
      data_set_mode=cfg['DATA_SET_MODE'],
      select_percentage=cfg['select_percentage'],
      border=float(border),
    )
    Runner = getRunner(cfg['AGENT'])
    runner_bcq = Runner(cfg['base_dir'], create_agent_fn, cfg=cfg)
    runner_bcq.run_experiment()


def getRunner(agent):
  return run_bail_experiment.BailFixedReplayRunner if agent == 'BAIL' else run_experiment.FixedReplayRunner


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
