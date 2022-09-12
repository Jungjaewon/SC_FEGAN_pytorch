import os
import argparse
import yaml
import shutil
import os.path as osp


def make_train_directory(config, path):
    # Create directories if not exist.
    if not os.path.exists(config['TRAINING_CONFIG']['TRAIN_DIR']):
        os.makedirs(config['TRAINING_CONFIG']['TRAIN_DIR'])
    if not os.path.exists(os.path.join(config['TRAINING_CONFIG']['TRAIN_DIR'], config['TRAINING_CONFIG']['LOG_DIR'])):
        os.makedirs(os.path.join(config['TRAINING_CONFIG']['TRAIN_DIR'], config['TRAINING_CONFIG']['LOG_DIR']))
    if not os.path.exists(os.path.join(config['TRAINING_CONFIG']['TRAIN_DIR'], config['TRAINING_CONFIG']['SAMPLE_DIR'])):
        os.makedirs(os.path.join(config['TRAINING_CONFIG']['TRAIN_DIR'], config['TRAINING_CONFIG']['SAMPLE_DIR']))
    if not os.path.exists(os.path.join(config['TRAINING_CONFIG']['TRAIN_DIR'], config['TRAINING_CONFIG']['RESULT_DIR'])):
        os.makedirs(os.path.join(config['TRAINING_CONFIG']['TRAIN_DIR'], config['TRAINING_CONFIG']['RESULT_DIR']))
    if not os.path.exists(os.path.join(config['TRAINING_CONFIG']['TRAIN_DIR'], config['TRAINING_CONFIG']['MODEL_DIR'])):
        os.makedirs(os.path.join(config['TRAINING_CONFIG']['TRAIN_DIR'], config['TRAINING_CONFIG']['MODEL_DIR']))

    shutil.copy(path, osp.join(config['TRAINING_CONFIG']['TRAIN_DIR'], path))


def main(config):

    assert config['TRAINING_CONFIG']['MODE'] in ['train', 'test']

    print('{} is started'.format(config['TRAINING_CONFIG']['MODE']))
    if config['TRAINING_CONFIG']['MODAL'] == 'paper':
        from solver_paper import Solver
        solver = Solver(config)
    elif config['TRAINING_CONFIG']['MODAL'] == 'gan':
        from solver_gan import Solver
        solver = Solver(config)
    else:
        raise NotImplemented

    if config['TRAINING_CONFIG']['MODE'] == 'train':
        solver.train()
    print('{} is finished'.format(config['TRAINING_CONFIG']['MODE']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yml', help='specifies config yaml file')

    params = parser.parse_args()

    assert osp.exists(params.config)
    config = yaml.load(open(params.config, 'r'), Loader=yaml.FullLoader)
    make_train_directory(config, params.config)
    main(config)


