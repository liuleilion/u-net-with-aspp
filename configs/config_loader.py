import configargparse


def config_parser():
    parser = configargparse.ArgumentParser(description='Parathdetect')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='path to latest checkpoint')

    parser.add_argument('--start_epoch', type=int, default=0,
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('--epochs', type=int, default=30,
                        help='total number of training rounds')

    parser.add_argument('--data_root', type=str, default='./data',
                        help='dataset root containing train/test with Img and GT')

    parser.add_argument('--img_size', type=int, default=256,
                        help='image and mask resize')

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')

    parser.add_argument('--num_workers', type=int, default=0,
                        help='data loader workers')

    parser.add_argument('--pos_weight', type=float, default=5.0,
                        help='positive weight for BCEWithLogitsLoss')

    return parser


def get_config():
    parser = config_parser()
    cfg = parser.parse_args()
    return cfg
