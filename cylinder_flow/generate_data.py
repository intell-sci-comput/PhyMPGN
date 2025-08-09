from kogger import Logger
import pprint

from src.common.config import Config
from src.datasets.dataset import PDEGraphDataset


def main():
    # load and set config
    args = Config.get_parser().parse_args()
    config = Config(yaml_filename=args.filename)

    logger = Logger('MAIN')
    logger.info('Load config successfully!')
    logger.info(pprint.pformat(config.data))

    logger.info('Train data...')
    tr_dataset = PDEGraphDataset(
        root=config['data_root_dir'],
        raw_files=config['tr_raw_data'],
        processed_file=config['tr_processed_file'],
        dataset_start=config['dataset_start'],
        dataset_used=config['dataset_used'],
        time_start=config['time_start'],
        time_used=config['time_used'],
        window_size=config['window_size'],
        dtype=config['dtype'],
        training=True
    )

    logger.info('Validate data...')
    val_dataset = PDEGraphDataset(
        root=config['data_root_dir'],
        raw_files=config['val_raw_data'],
        processed_file=config['val_processed_file'],
        dataset_start=config['dataset_start'],
        dataset_used=config['dataset_used'],
        time_start=config['time_start'],
        time_used=config['time_used'],
        window_size=config['window_size'],
        dtype=config['dtype']
    )

    logger.info('Test data...')
    te_dataset = PDEGraphDataset(
        root=config['data_root_dir'],
        raw_files=config['te_raw_data'],
        processed_file=config['te_processed_file'],
        dataset_start=config['te_dataset_start'],
        dataset_used=config['te_dataset_used'],
        time_start=config['time_start'],
        time_used=config['time_used'],
        window_size=config['te_window_size'],
        dtype=config['dtype']
    )


if __name__ == '__main__':
    main()