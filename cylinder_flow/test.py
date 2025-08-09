import time
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
import pprint
import numpy as np
import matplotlib.pyplot as plt
from kogger import Logger

from src.common.config import Config
from src.models.model import Model
from src.datasets.dataset import PDEGraphDataset
from src.utils.utils import compute_armse


def test(model, te_loader, te_writer, device, logger):
    model.eval()

    te_num = len(te_loader)
    te_loss_list, te_armse_list = [], []
    mse_list = []
    pred_list = []
    truth_list = []
    inference_time_list = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(te_loader):
            # batch: DataBatch(y=[bxn, t, 2], pos=[bxn, 2], edge_index=[2, bxe], batch=[bxn])
            batch = batch.to(device)
            target = batch.y.transpose(0, 1)  # [t, n, 2]
            batch.y = target[0]  # [n, 2]
            graph = batch
            # target = target[:400]
            t = target.shape[0]
            start_time = time.time()
            U_pred = model(graph, steps=t - 1)  # [t, n, 2]
            inference_time = time.time() - start_time
            inference_time_list.append(inference_time)
            # [t, n', 2]
            target = torch.index_select(target, 1, batch.truth_index)

            # dimensional
            pos = torch.index_select(graph.pos, 0, graph.truth_index)
            U_pred, target, pos = PDEGraphDataset.dimensional(
                U_pred=U_pred,
                U_gt=target,
                pos=pos,
                u_m=graph.u_m,
                D=graph.r * 2
            )

            te_loss = torch.mean(F.mse_loss(U_pred.reshape(t, -1),
                                            target.reshape(t, -1),
                                            reduction='none'), dim=1)  # [t,]
            armse = compute_armse(U_pred, target)
            info_str = '[TEST {:>2}/{}] MSE at {}t: {:.2e}, armse: {:.3f}'\
                .format(batch_idx, te_num, t, te_loss.mean(),
                        armse[-1])
            info_str += ', time: {:.2f}s'.format(inference_time)
            logger.info(info_str)
            te_writer.add_text('Results', info_str)

            te_loss_list.append(te_loss.mean())
            te_armse_list.append(armse[-1])
            mse_list.append(te_loss.detach().cpu().numpy())

            pred_np = U_pred.detach().cpu().numpy()
            truth_np = target.detach().cpu().numpy()
            pred_list.append(pred_np)
            truth_list.append(truth_np)

            plt.close('all')

    mean_loss = sum(te_loss_list) / len(te_loss_list)
    mean_armse = sum(te_armse_list) / len(te_armse_list)

    info_str = '[Test {}] Mean Loss: {:.2e}, Mean Armse: {:.3f}'\
        .format(len(te_loader), mean_loss, mean_armse)
    info_str += ', time: {:.2f}'.format(np.mean(inference_time_list))
    logger.info(info_str)
    te_writer.add_text('Results', info_str)

    return mean_loss, mean_armse


def main():
    # load and set config
    args = Config.get_parser().parse_args()
    config = Config(yaml_filename=args.filename)

    accelerator = Accelerator()
    te_writer = SummaryWriter(log_dir=config['log_dir'] + '_test')
    logger = Logger('PID %d' % accelerator.process_index)
    if accelerator.is_main_process:
        logger.info('Load config successfully!')
        logger.info(pprint.pformat(config.data))

    te_dataset = PDEGraphDataset(
        root=config['data_root_dir'],
        raw_files=config['te_raw_data'],
        processed_file=config['te_processed_file'],
        dataset_start=config['dataset_start'],
        dataset_used=config['dataset_used'],
        time_start=config['time_start'],
        time_used=config['time_used'],
        window_size=config['te_window_size'],
        dtype=config['dtype']
    )
    te_loader = DataLoader(
        dataset=te_dataset
    )

    model = Model(
        encoder_config=config['encoder_config'],
        mpnn_block_config=config['mpnn_block_config'],
        decoder_config=config['decoder_config'],
        laplace_block_config=config['laplace_block_config'],
        dtype=config['dtype'],
        device=accelerator.device,
        integral=config['integral']
    )

    model = accelerator.prepare(model)
    accelerator.load_state(input_dir=config['ckpt_path_val'])

    logger.info('Test...')
    test(
        model=model,
        te_writer=te_writer,
        te_loader=te_loader,
        device=accelerator.device,
        logger=logger
    )

    te_writer.close()
    logger.info('Done!')


if __name__ == '__main__':
    main()
