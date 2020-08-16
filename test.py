import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from model.inhibited_softmax import inhibited_softmax as IS
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from train import get_instance
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt


def main(config, resume):
    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
    #data_loader = module_data.FashionMnistDataLoader("data/",
        batch_size=16,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )
    
    cross_test = True

    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    model.summary()

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # load state dict
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    if cross_test == True:
        probs_test = []
        groundtruth_test = []
    
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            if cross_test == True:
                output_inh = IS(output, config)
                probs_test.append(output_inh.data.cpu().numpy())
                groundtruth_test.append(target.cpu().numpy())
            #
            # save sample images, or do something with output here
            #
            
            # computing loss, metrics on test set
            if(config['last_layer_l2_regularizer'] == True):
                weight_reg = torch.norm(model.resnet.fc.weight, 2)
                l2_panelty = config['l2_panelty']
            elif(config['last_layer_l2_regularizer'] == False):
                weight_reg = 0.0
                l2_panelty = 0.0
            loss = loss_fn(output, target, (weight_reg*l2_panelty), config)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size
    
    '''Uncertainity Calculations'''
    
    if cross_test == True:
        probs_out = []
        groundtruth_out = []
        cross_data_loader = getattr(module_data, 'HAMDataLoader')(
            config['data_loader']['args']['data_dir'],
            batch_size=16,
            shuffle=False,
            validation_split=0.0,
            training=False,
            num_workers=2)
        with torch.no_grad():
            for i, (data, target) in enumerate(tqdm(cross_data_loader)):
                data, target = data.to(device), target.to(device)
                output = model(data)
                output_ihs = IS(output, config)

                probs_out.append(output_ihs.data.cpu().numpy())
                groundtruth_out.append(target.cpu().numpy())

                if(config['last_layer_l2_regularizer'] == True):
                    weight_reg = torch.norm(model.fc.weight, 2)
                    l2_panelty = config['l2_panelty']
                elif(config['last_layer_l2_regularizer'] == False):
                    weight_reg = 0.0
                    l2_panelty = 0.0
                loss = loss_fn(output, target, (weight_reg*l2_panelty), config)
                batch_size = data.shape[0]
                total_loss += loss.item() * batch_size
                for i, metric in enumerate(metric_fns):
                    total_metrics[i] += metric(output, target) * batch_size

        groundtruth_test = np.concatenate(groundtruth_test)
        probs_test = np.concatenate(probs_test)
        probs_test_sum = np.sum(probs_test, axis=1)

        groundtruth_out = np.concatenate(groundtruth_out)
        probs_out = np.concatenate(probs_out)
        probs_out_sum = np.sum(probs_out, axis=1)

        target = np.zeros(len(groundtruth_test)+len(groundtruth_out), dtype=np.int)
        target[len(groundtruth_test):] = 1

        all_data = np.concatenate([1-probs_test_sum, 1-probs_out_sum])
        roc = roc_auc_score(target, all_data)
        print("ROC:", roc)
        ap = average_precision_score(target, all_data)
        fpr, tpr, _ = roc_curve(target, all_data)
        pr, re, _ = precision_recall_curve(target, all_data)
        print("AP:", ap)
        #print(pr, re)

        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
    
    
    else:
        n_samples = len(data_loader.sampler)
        log = {'loss': total_loss / n_samples}
        log.update({met.__name__ : total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)})
        print(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LeNet_Inhibited')

    parser.add_argument('-r', '--resume', default='saved/Resnet18/model_best.pth',
                        type=str,help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device

    main(config, args.resume)
