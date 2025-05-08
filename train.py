import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import argparse
import logging
import time
import torch.nn as nn
import torchvision.datasets as datasets
import random
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=True, choices=[True, False])
parser.add_argument('--downsampling-method', type=str, default='res', choices=['conv', 'res'])
parser.add_argument('--nepochs', type=int, default=50)
parser.add_argument('--data_aug', type=eval, default=False, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--test_batch_size', type=int, default=4)

parser.add_argument('--use_anode', type=eval, default=False, choices=[True, False], help='Whether to use Augmented Neural ODE')
parser.add_argument('--anode_dim', type=int, default=0, help='Dimension of augmentation for ANODE (if use_anode is True)')
parser.add_argument('--use_lyapunov_loss', type=eval, default=False, choices=[True, False], help='Whether to use Lyapunov loss')
parser.add_argument('--lyapunov_coeff', type=float, default=0.01, help='Coefficient for Lyapunov loss term')

# New arguments for VLLM fallback
parser.add_argument('--use_vllm_fallback', type=eval, default=False, choices=[True, False], help='Whether to use VLLM classification as fallback.')
parser.add_argument('--vllm_confidence_threshold', type=float, default=0.5, help='Confidence threshold k for main model. Below this, VLLM label is used.')
parser.add_argument('--vllm_label_filename', type=str, default="label_llm.pth", help='Filename of the VLLM predicted labels (expected in each dataset folder, for the test split).')

parser.add_argument('--save', type=str, default='./experiment1')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x
        out = self.relu(self.norm1(x))
        if self.downsample is not None:
            shortcut = self.downsample(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + shortcut

class ConcatConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )
    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)

class ODEfunc(nn.Module):
    def __init__(self, processed_dim):
        super(ODEfunc, self).__init__()
        self.processed_dim = processed_dim
        self.norm1 = norm(self.processed_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(self.processed_dim, self.processed_dim, 3, 1, 1)
        self.norm2 = norm(self.processed_dim)
        self.conv2 = ConcatConv2d(self.processed_dim, self.processed_dim, 3, 1, 1)
        self.norm3 = norm(self.processed_dim)
        self.nfe = 0
        self.use_lyapunov_integration = False

    def forward(self, t, x_potentially_with_lyap_channel):
        self.nfe += 1
        actual_x = x_potentially_with_lyap_channel
        if self.use_lyapunov_integration:
            actual_x = x_potentially_with_lyap_channel[:, :-1, :, :]

        out = self.norm1(actual_x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        dx_dt = self.conv2(t, out)
        dx_dt = self.norm3(dx_dt)

        if self.use_lyapunov_integration:
            lyap_val_dt = torch.sum(dx_dt**2, dim=(1, 2, 3), keepdim=True)
            num_elements = dx_dt.size(1) * dx_dt.size(2) * dx_dt.size(3)
            if num_elements > 0:
                 lyap_val_dt = lyap_val_dt / num_elements
            else:
                 lyap_val_dt = torch.zeros_like(lyap_val_dt)
            dummy_lyap_channel_shape_template = x_potentially_with_lyap_channel[:, -1:, :, :]
            lyap_val_dt_channel = lyap_val_dt.expand_as(dummy_lyap_channel_shape_template)
            return torch.cat((dx_dt, lyap_val_dt_channel), dim=1)
        else:
            return dx_dt

class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()
        self.last_lyapunov_value = torch.tensor(0.0)

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        current_ode_input = x
        if args.use_anode and args.anode_dim > 0:
            aug = torch.zeros(x.size(0), args.anode_dim, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            current_ode_input = torch.cat([x, aug], dim=1)

        ode_solver_input = current_ode_input
        if args.use_lyapunov_loss and args.lyapunov_coeff > 0:
            self.odefunc.use_lyapunov_integration = True
            lyap_init_channel = torch.zeros(current_ode_input.size(0), 1,
                                            current_ode_input.size(2), current_ode_input.size(3),
                                            device=current_ode_input.device, dtype=current_ode_input.dtype)
            ode_solver_input = torch.cat([current_ode_input, lyap_init_channel], dim=1)
        else:
            self.odefunc.use_lyapunov_integration = False

        integrated_output_tuple = odeint(self.odefunc, ode_solver_input, self.integration_time, rtol=args.tol, atol=args.tol)
        output_from_solver = integrated_output_tuple[1]

        features_after_ode = output_from_solver
        self.last_lyapunov_value = torch.tensor(0.0, device=x.device)

        if args.use_lyapunov_loss and args.lyapunov_coeff > 0 and self.odefunc.use_lyapunov_integration:
            features_after_ode = output_from_solver[:, :-1, :, :]
            self.last_lyapunov_value = torch.mean(output_from_solver[:, -1:, :, :])
            self.odefunc.use_lyapunov_integration = False
        
        final_features = features_after_ode
        if args.use_anode and args.anode_dim > 0:
            final_features = features_after_ode[:, :x.size(1), :, :]
            
        return final_features

    @property
    def nfe(self): return self.odefunc.nfe
    @nfe.setter
    def nfe(self, value): self.odefunc.nfe = value

class Flatten(nn.Module):
    def __init__(self): super(Flatten, self).__init__()
    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

class RunningAverageMeter(object):
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()
    def reset(self):
        self.val = None
        self.avg = 0
    def update(self, val):
        if self.val is None: self.avg = val
        else: self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

def inf_generator(iterable):
    iterator = iterable.__iter__()
    while True:
        try: yield iterator.__next__()
        except StopIteration: iterator = iterable.__iter__()

def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = args.lr * batch_size / batch_denom
    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]
    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]
    return learning_rate_fn

def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)

# Modified accuracy function
def accuracy(types, model, dataset_loader, device, config_args): # Pass full args object
    total_correct = 0
    model.eval()
    fallback_count = 0

    # The dataloader now yields (x, y_true, y_vllm_batch)
    # y_vllm_batch will be -1 for samples where it's not available or applicable
    for batch_idx, (x_batch, y_true_batch, y_vllm_batch) in enumerate(dataset_loader):
        x_batch = x_batch.to(device).float()
        y_true_batch_np = y_true_batch.numpy()
        y_vllm_batch_np = y_vllm_batch.numpy() # Contains -1 if no VLLM label for a sample

        with torch.no_grad():
            logits = model(x_batch)
            probabilities = torch.softmax(logits, dim=1)
            confidences, predicted_classes_model = torch.max(probabilities, dim=1)

        predicted_classes_model_np = predicted_classes_model.cpu().numpy()
        confidences_np = confidences.cpu().numpy()
        
        final_predicted_classes = np.copy(predicted_classes_model_np)

        if config_args.use_vllm_fallback:
            for j in range(len(confidences_np)):
                if confidences_np[j] < config_args.vllm_confidence_threshold and y_vllm_batch_np[j] != -1:
                    final_predicted_classes[j] = y_vllm_batch_np[j]
                    fallback_count +=1
        
        total_correct += np.sum(final_predicted_classes == y_true_batch_np)

    model.train()
    if len(dataset_loader.dataset) == 0: return 0, 0
    
    final_accuracy = total_correct / len(dataset_loader.dataset)
    return final_accuracy, fallback_count


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def makedirs(dirname):
    if not os.path.exists(dirname): os.makedirs(dirname)

def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger(filepath)
    logger.handlers.clear() 
    
    if debug: level = logging.DEBUG
    else: level = logging.INFO
    logger.setLevel(level)
    
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    return logger

class myDataset(Dataset):
    def __init__(self, data, label, vllm_label=None): # Added vllm_label
        self.data = data
        self.len = data.shape[0] 
        self.label = label
        self.vllm_label = vllm_label

        if self.vllm_label is not None and len(self.vllm_label) != self.len:
            print(f"Warning: Mismatch in length between main labels ({self.len}) and VLLM labels ({len(self.vllm_label)}). VLLM labels for this dataset will be disabled.")
            self.vllm_label = None

    def __getitem__(self, index):
        tdata = self.data[index]
        tlabel = self.label[index] 
        tvllm_label = self.vllm_label[index] if self.vllm_label is not None else torch.tensor(-1, dtype=torch.long)
        return tdata, tlabel, tvllm_label
    
    def __len__(self):
        return self.len 

def De(label_tensor):
    re = torch.tensor(list(map(lambda x: int(x.item()) - 1, label_tensor)))
    return re

def ReadData(args_config, rootpath_dataset_current):
    train_data_path = os.path.join(rootpath_dataset_current, 'train.pth')
    train_label_path = os.path.join(rootpath_dataset_current, 'train_label.pth')
    test_data_path = os.path.join(rootpath_dataset_current, 'test.pth')
    test_label_path = os.path.join(rootpath_dataset_current, 'test_label.pth')

    if not all(os.path.exists(p) for p in [train_data_path, train_label_path, test_data_path, test_label_path]):
        raise FileNotFoundError(f"One or more data/label .pth files not found in {rootpath_dataset_current}")

    train_data=torch.load(train_data_path)
    train_label=torch.load(train_label_path).long()
    test_data=torch.load(test_data_path)
    test_label=torch.load(test_label_path).long()
    
    channels = int(train_data.shape[1])
    alldata = torch.cat([test_data,train_data],dim=0)
    alllabel = torch.cat([test_label,train_label],dim=0)
    
    if torch.any(alllabel < 0):
        print('ERROR: Negative labels found in data.')
        raise ValueError("Negative labels found")

    unique_labels = torch.unique(alllabel)
    if 0 not in unique_labels:
        print(f'Warning for {os.path.basename(rootpath_dataset_current)}: Class labels appear to be 1-indexed. Remapping to 0-indexed.')
        types = int(torch.max(alllabel)) 
        alllabel = De(alllabel) 
    else:
        types = int(torch.max(alllabel)) + 1
    
    if alldata.shape[0] != alllabel.shape[0]:
        print(f'Warning for {os.path.basename(rootpath_dataset_current)}! Data and label counts mismatch!')
    
    index_list = list(range(alldata.shape[0]))
    random.seed(42)
    random.shuffle(index_list)
    
    num_total = len(index_list)
    num_train = int(num_total * 0.8)
    num_val = int(num_total * 0.1)
    
    train_index = index_list[0:num_train]
    val_index = index_list[num_train : num_train + num_val]
    test_index = index_list[num_train + num_val :]
    
    data_train = alldata[train_index]
    data_val = alldata[val_index]
    data_test = alldata[test_index]
    
    label_train = alllabel[train_index]
    label_val = alllabel[val_index]
    label_test = alllabel[test_index]
    
    vllm_label_test_loaded = None
    if args_config.use_vllm_fallback:
        potential_vllm_test_label_path = os.path.join(rootpath_dataset_current, args_config.vllm_label_filename)
        if os.path.exists(potential_vllm_test_label_path):
            try:
                vllm_labels_for_this_test_split = torch.load(potential_vllm_test_label_path).long()
                if len(vllm_labels_for_this_test_split) == len(data_test):
                    vllm_label_test_loaded = vllm_labels_for_this_test_split
                    print(f"Successfully loaded VLLM labels for test set from {potential_vllm_test_label_path} for dataset {os.path.basename(rootpath_dataset_current)}.")
                else:
                    print(f"Warning for {os.path.basename(rootpath_dataset_current)}: VLLM labels at {potential_vllm_test_label_path} length ({len(vllm_labels_for_this_test_split)}) "
                          f"does not match test set length ({len(data_test)}). VLLM fallback disabled for this test set.")
            except Exception as e:
                print(f"Error loading VLLM labels from {potential_vllm_test_label_path} for {os.path.basename(rootpath_dataset_current)}: {e}. VLLM fallback disabled for this test set.")
        else:
            print(f"Note for {os.path.basename(rootpath_dataset_current)}: VLLM label file {potential_vllm_test_label_path} not found. VLLM fallback will be inactive for this test set.")

    train_dataset = myDataset(data=data_train, label=label_train)
    val_dataset = myDataset(data=data_val, label=label_val)
    test_dataset = myDataset(data=data_test, label=label_test, vllm_label=vllm_label_test_loaded)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args_config.batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=args_config.test_batch_size, shuffle=False, num_workers=0) 
    test_dataloader = DataLoader(test_dataset, batch_size=args_config.test_batch_size, shuffle=False, num_workers=0)
    
    return types, channels, train_dataloader, val_dataloader, test_dataloader

class HydraAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, all_head_size):
        super().__init__()
        self.query = nn.Linear(hidden_size, all_head_size)
        self.key = nn.Linear(hidden_size, all_head_size)
        self.value = nn.Linear(hidden_size, all_head_size)
    def forward(self, x):
        q = self.query(x); k = self.key(x); v = self.value(x)
        q = q / q.norm(dim=-1, keepdim=True); k = k / k.norm(dim=-1, keepdim=True)
        kv = (k * v).sum(dim=-2, keepdim=True); out = q * kv
        return out

if __name__ == '__main__':
    root_data_path = r'' 
    if not os.path.exists(root_data_path):
        print(f"WARNING: Data root path does not exist: {root_data_path}")
        print("Attempting to create and use dummy data for a single run.")
        dummy_data_name = "DummyDataset"
        root_data_path = "./temp_ucr_tensor" 
        dummy_root_path_dataset = os.path.join(root_data_path, dummy_data_name)
        os.makedirs(dummy_root_path_dataset, exist_ok=True)
        
        dummy_n_train, dummy_n_test = 80, 20
        dummy_all_N = dummy_n_train + dummy_n_test
        dummy_train_data = torch.randn(dummy_n_train, 1, 8, 8) 
        dummy_train_label = torch.randint(0, 3, (dummy_n_train,))
        dummy_test_data = torch.randn(dummy_n_test, 1, 8, 8)
        dummy_test_label = torch.randint(0, 3, (dummy_n_test,))

        torch.save(dummy_train_data, os.path.join(dummy_root_path_dataset, 'train.pth'))
        torch.save(dummy_train_label, os.path.join(dummy_root_path_dataset, 'train_label.pth'))
        torch.save(dummy_test_data, os.path.join(dummy_root_path_dataset, 'test.pth'))
        torch.save(dummy_test_label, os.path.join(dummy_root_path_dataset, 'test_label.pth'))
        num_test_dummy_final_split = int(dummy_all_N * 0.1) if int(dummy_all_N * 0.1) > 0 else 1 # ensure at least 1
        if (dummy_all_N - int(dummy_all_N * 0.8) - int(dummy_all_N*0.1)) < num_test_dummy_final_split: # check for off-by-one due to int
             num_test_dummy_final_split = (dummy_all_N - int(dummy_all_N * 0.8) - int(dummy_all_N*0.1))


        dummy_vllm_labels = torch.randint(0, 3, (num_test_dummy_final_split,))
        torch.save(dummy_vllm_labels, os.path.join(dummy_root_path_dataset, args.vllm_label_filename))
        
        print(f"Created dummy data in {dummy_root_path_dataset}, including {args.vllm_label_filename} with {num_test_dummy_final_split} labels.")
        sdata_dirs = [dummy_data_name]
    else:
        sdata_dirs = os.listdir(root_data_path)

    total_acc_summary = {}

    for dataset_dirname in tqdm(sdata_dirs, desc="Processing datasets"):
        current_dataset_name_for_log = dataset_dirname
        print(f"\nProcessing dataset: {current_dataset_name_for_log}")
        
        # logger needs to be dataset-specific for file logging
        # Moved logger setup inside the loop
        current_save_dir = os.path.join(args.save, current_dataset_name_for_log) # Save per-dataset results
        makedirs(current_save_dir)
        log_file_path = os.path.join(current_save_dir, f'log.txt')
        logger = get_logger(logpath=log_file_path, filepath=os.path.abspath(__file__) + "_" + current_dataset_name_for_log) # Unique logger name
        logger.info(f"Args for {current_dataset_name_for_log}: {args}")

        try:
            # rootpath_dataset is the specific dataset's folder
            rootpath_dataset = os.path.join(root_data_path, dataset_dirname)
            types, channels, train_loader, val_loader, test_loader = ReadData(args, rootpath_dataset)
        except FileNotFoundError as e:
            logger.error(f'ERROR: Files not found for dataset {dataset_dirname}: {e}. Skipping.')
            total_acc_summary[current_dataset_name_for_log] = f"Error: {e}"
            continue
        except Exception as e:
            logger.error(f'ERROR loading data for {dataset_dirname}: {e}. Skipping.', exc_info=True)
            total_acc_summary[current_dataset_name_for_log] = f"Error: {e}"
            continue
        
        if len(train_loader.dataset) == 0:
            logger.warning(f"No training data for {dataset_dirname}. Skipping.")
            total_acc_summary[current_dataset_name_for_log] = "No training data"
            continue

        try:
            val_acc_history = []
            device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {device} for {current_dataset_name_for_log}")
        
            is_odenet = args.network == 'odenet'
            feature_dim_before_ode = 64

            if args.downsampling_method == 'conv':
                downsampling_layers = [
                    nn.Conv2d(channels, feature_dim_before_ode, 3, 1), norm(feature_dim_before_ode), nn.ReLU(inplace=True),
                    nn.Conv2d(feature_dim_before_ode, feature_dim_before_ode, 4, 2, 1), norm(feature_dim_before_ode), nn.ReLU(inplace=True),
                    nn.Conv2d(feature_dim_before_ode, feature_dim_before_ode, 4, 2, 1)]
            elif args.downsampling_method == 'res':
                downsampling_layers = [
                    nn.Conv2d(channels, feature_dim_before_ode, 3, 1),
                    ResBlock(feature_dim_before_ode, feature_dim_before_ode, stride=2, downsample=conv1x1(feature_dim_before_ode, feature_dim_before_ode, 2)),
                    ResBlock(feature_dim_before_ode, feature_dim_before_ode, stride=2, downsample=conv1x1(feature_dim_before_ode, feature_dim_before_ode, 2))]
            
            odefunc_processed_dim = feature_dim_before_ode
            if is_odenet and args.use_anode and args.anode_dim > 0:
                odefunc_processed_dim += args.anode_dim
                logger.info(f"Using ANODE with augmentation dim: {args.anode_dim}. ODE func processed dim: {odefunc_processed_dim}")

            if is_odenet:
                odefunc_instance = ODEfunc(processed_dim=odefunc_processed_dim)
                feature_layers = [ODEBlock(odefunc_instance)]
            else:
                feature_layers = [ResBlock(feature_dim_before_ode, feature_dim_before_ode) for _ in range(1)]

            fc_layers = [norm(feature_dim_before_ode), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(feature_dim_before_ode, int(types))]
            model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers).to(device)
        
            logger.info(f"Model for {current_dataset_name_for_log}:\n{model}")
            logger.info(f'Number of parameters for {current_dataset_name_for_log}: {count_parameters(model)}')
        
            criterion = nn.CrossEntropyLoss().to(device)
            data_gen = inf_generator(train_loader)
            batches_per_epoch = len(train_loader)

            if batches_per_epoch == 0:
                logger.warning(f"Train loader for {current_dataset_name_for_log} is empty. Skipping training.")
                total_acc_summary[current_dataset_name_for_log] = "Empty train loader"
                continue

            lr_fn = learning_rate_with_decay(
                args.batch_size, batch_denom=128, batches_per_epoch=batches_per_epoch, boundary_epochs=[60, 100, 140],
                decay_rates=[1, 0.1, 0.01, 0.001])
        
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
            best_val_acc = 0
            best_val_epoch = -1
            batch_time_meter = RunningAverageMeter(); f_nfe_meter = RunningAverageMeter(); b_nfe_meter = RunningAverageMeter()
            end = time.time()
        
            for itr in range(args.nepochs * batches_per_epoch):
                model.train()
                current_epoch = itr // batches_per_epoch
                for param_group in optimizer.param_groups: param_group['lr'] = lr_fn(itr)
            
                optimizer.zero_grad()
                x, y, _ = data_gen.__next__() # Train loader also yields 3 items, ignore vllm label for training
                x = x.to(device).float(); y = y.to(device).long()
                
                logits = model(x)
                main_loss = criterion(logits, y)
                total_loss = main_loss

                nfe_forward_iter = 0
                if is_odenet:
                    nfe_forward_iter = feature_layers[0].nfe; feature_layers[0].nfe = 0
                if is_odenet and args.use_lyapunov_loss and args.lyapunov_coeff > 0:
                    lyapunov_term = feature_layers[0].last_lyapunov_value
                    total_loss = total_loss + args.lyapunov_coeff * lyapunov_term
                    # Log Lyapunov term less frequently to avoid clutter
                    if itr % (batches_per_epoch * 5) == 0 : # e.g. every 5 epochs
                         logger.info(f"Epoch {current_epoch:04d} | Lyapunov Term: {lyapunov_term.item():.4f} | Main Loss: {main_loss.item():.4f}")
            
                total_loss.backward()
                optimizer.step()
            
                nfe_backward_iter = 0
                if is_odenet:
                    nfe_backward_iter = feature_layers[0].nfe; feature_layers[0].nfe = 0
                    f_nfe_meter.update(nfe_forward_iter); b_nfe_meter.update(nfe_backward_iter)

                batch_time_meter.update(time.time() - end); end = time.time()
            
                if itr > 0 and itr % batches_per_epoch == 0:
                    val_acc, _ = accuracy(types, model, val_loader, device, args) 
                    val_acc_history.append(val_acc)

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_val_epoch = current_epoch
                        torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(current_save_dir, f'model_best_val.pth'))
                        logger.info(f"*** Epoch {current_epoch:04d} | Saved new best val_model with Val Acc: {best_val_acc:.4f} ***")
                    
                    train_acc, _ = accuracy(types, model, train_loader, device, args)


                    log_msg_parts = [f"Epoch {current_epoch:04d}", f"Time {batch_time_meter.val:.3f}s ({batch_time_meter.avg:.3f}s)"]
                    if is_odenet: log_msg_parts.extend([f"NFE-F {f_nfe_meter.avg:.1f}", f"NFE-B {b_nfe_meter.avg:.1f}"])
                    log_msg_parts.extend([
                        f"LR {optimizer.param_groups[0]['lr']:.5f}", f"Loss {total_loss.item():.4f}",
                        f"Train Acc {train_acc:.4f}", f"Val Acc {val_acc:.4f} (Best Val @Ep{best_val_epoch}: {best_val_acc:.4f})"])
                    logger.info(" | ".join(log_msg_parts))
            
            logger.info(f"Finished training for {current_dataset_name_for_log}. Best validation accuracy: {best_val_acc:.4f} at epoch {best_val_epoch}")
            logger.info("Loading best validation model for test set evaluation...")
            if os.path.exists(os.path.join(current_save_dir, 'model_best_val.pth')):
                checkpoint = torch.load(os.path.join(current_save_dir, 'model_best_val.pth'), map_location=device)
                model.load_state_dict(checkpoint['state_dict'])
            else:
                logger.warning("No best validation model saved. Evaluating with the last epoch model.")

            test_acc, test_fallback_count = accuracy(types, model, test_loader, device, args)
            logger.info(f"Test Accuracy for {current_dataset_name_for_log}: {test_acc:.4f}")
            if args.use_vllm_fallback:
                logger.info(f"VLLM fallbacks used on test set for {current_dataset_name_for_log}: {test_fallback_count} times.")

            total_acc_summary[current_dataset_name_for_log] = test_acc

            del model, optimizer, train_loader, val_loader, test_loader #, data_gen
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Unhandled exception during training/evaluation for {dataset_dirname}: {e}", exc_info=True)
            total_acc_summary[current_dataset_name_for_log] = f"Error: {e}"
            continue

    print("\n\n--- Overall Test Accuracy Summary ---")
    for dataset_name_summary, best_acc_val_summary in total_acc_summary.items():
        if isinstance(best_acc_val_summary, float):
            print(f"Dataset: {dataset_name_summary}, Final Test Accuracy: {best_acc_val_summary:.4f}")
        else:
            print(f"Dataset: {dataset_name_summary}, Result: {best_acc_val_summary}")