import argparse
import torch
import networkx
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from data_loader.dataset import QuerySet, MPNNDataset
from data_loader.data_loaders import mpnn_collate_fn
from base.base_data_loader import BaseDataLoader
from data_loader.data_utils import get_query_key


def main(config):
    logger = config.get_logger('test')

    # set up query set and dataloader
    model_type = config['arch']['type']
    data_loader_cfg = config['data_loader']['args']
    query_set = QuerySet(data_loader_cfg['data_dir'], data_loader_cfg['dataset'], data_loader_cfg['query_size'],
                         data_loader_cfg['train_set_max_id'])
    if model_type in ["GCN", "GIN", "GAT", "GraphSAGE"]:
        dataset = MPNNDataset(query_set.test_queries, query_set.data_graph)
        collate_fn = mpnn_collate_fn
    else:
        raise NotImplementedError("SubgraphCountingDataLoader only supports MPNNDataset.")
    data_loader = BaseDataLoader(dataset, batch_size=1, shuffle=False, validation_split=0.0,
                                 num_workers=data_loader_cfg['num_workers'], collate_fn=collate_fn)

    # build model architecture
    input_size = 1 + max(networkx.get_node_attributes(query_set.data_graph, 'label').values())
    model = config.init_obj('arch', module_arch, input_size=input_size)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    n_samples = len(query_set.test_queries)
    predict_log = torch.zeros(n_samples)
    target_log = torch.zeros(n_samples)
    total_metrics = torch.zeros(len(metric_fns))
    total_loss = 0.0

    result = []

    with torch.no_grad():
        for i, (x, edge_index, log_count, query_name) in enumerate(tqdm(data_loader)):
            query_name = query_name[0]
            x, edge_index, log_count = x[0].to(device), edge_index[0].to(device), log_count.to(device)
            output = model(x, edge_index)

            predict_log[i] = output.item()
            target_log[i] = log_count.item()

            # computing loss
            loss = loss_fn(output, log_count)
            total_loss += loss.item()

            # evaluation the prediction
            ground_truth, predict = torch.pow(10, log_count).item(), torch.pow(10, output).item() #ground truth, predict is x+1
            ground_truth, predict = round(ground_truth), round(predict)
            q_error = max(ground_truth, predict) / min(ground_truth, predict)    # q_error is max(c+1, c'+1)/min(c+1, c'+1)
            log_q_error = (output - log_count).item()
            result.append([query_name, ground_truth, predict, q_error, log_q_error])

        # computing metrics on test set
        for i, metric in enumerate(metric_fns):
            total_metrics[i] += metric(predict_log, target_log)

    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item()  for i, met in enumerate(metric_fns)
    })
    logger.info(log)

    # Save the result as csv
    result = sorted(result, key=lambda q: get_query_key(q[0]))
    df = pd.DataFrame(result, columns=['query_name', 'ground_truth', 'predict', 'q_error', 'log_q_error'])
    path = Path(config.resume).parent
    dir_name = []
    save_dir = Path(config['trainer']['save_dir'])
    while path.name != save_dir.name:
        dir_name.append(path.name)
        path = path.parent
    dir_name[-1] = "result"
    for dir_name in reversed(dir_name):
        save_dir = save_dir / dir_name
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=False)
    save_dir = save_dir / 'result.csv'
    df.to_csv(save_dir, index=False)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
