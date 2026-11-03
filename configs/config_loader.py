import configargparse

def config_parser():
    parser = configargparse.ArgumentParser(description='Parathdetect')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='path to latest checkpoint')

    parser.add_argument('--start_epoch', type=int, default=0,
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('--epochs', type=int, default=30,
                        help='total number of training rounds')

    return parser


def get_config():
    parser = config_parser()
    cfg = parser.parse_args()
    return cfg

