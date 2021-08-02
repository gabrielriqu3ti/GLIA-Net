import argparse
import os

from core import Transfer
from utils.project_utils import load_config, get_logger, get_devices, str2bool

parser = argparse.ArgumentParser(description='AneurysmSeg evaluation')
parser.add_argument('-c', '--config', type=str, required=False, default='default',
                    help='config name. default: \'default\'')
parser.add_argument('-n', '--exp_id', type=int, required=False, default=1,
                    help='to identify different exp ids.')
parser.add_argument('-d', '--device', type=str, required=False, default='0',
                    help='device id for cuda and \'cpu\' for cpu. can be multiple devices split by \',\'.')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='whether to use verbose/debug logging level.')
args = parser.parse_args()


def save_model_adapted_to_input(config, exp_path, devices, logger, verbose):

    transferor = Transfer(config, exp_path, devices, logger)
    if verbose:
        print(transferor.model)
    transferor.adapt_model()
    transferor.save_checkpoint()
    if verbose:
        print(transferor.model)


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    exp_path = os.path.join('exp', args.config.replace('transfer_', ''))
    config = load_config(os.path.join('configs', args.config + '.yaml'))

    exp_path = os.path.join(exp_path, str(args.exp_id))

    logging_folder = os.path.join(exp_path, config.get('logging_folder')) \
        if config.get('logging_folder') is not None else None
    logger = get_logger('Task%sTransfer' % config['task'], logging_folder, args.verbose)
    logger.debug('config loaded:\n%s', config)
    devices = get_devices(args.device, logger)
    logger.info('use device %s' % args.device)

    try:
        save_model_adapted_to_input(config, exp_path, devices, logger, args.verbose)
    except Exception as e:
        logger.exception(e)
