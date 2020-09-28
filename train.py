import argparse
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from model import *
from dataset import *
from utils import *
import torchnet as tnt
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from torch.utils.data.distribute import DistributedSampler

def train(args):
    ## Parameter
    state = {k: v for k, v in args._get_kwargs()}
    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    
    mode = args.mode
    train_continue = args.train_continue
    T = args.T
    n_labeled = args.n_labeled
    result_dir = args.result_dir
    checkpoint = args.checkpoint
    data_dir = args.data_dir
    log_dir = args.log_dir
    lambda_u = args.lambda_u
    alpha = args.alpha
    train_iteration = args.train_iteration
    ema_decay = args.ema_decay
    threshold = args.threshold
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_acc = 0
    local_rank = -1 # if distributed, 0

    np.random.seed(0)

    if not os.path.isdir(result_dir):
        mkdir_p(result_dir)
    if not os.path.isdir(log_dir):
        mkdir_p(log_dir)

    if local_rank == -1:
        device = torch.device('cuda')
        world_size = 1
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        torch.distributed.init_process_group(backend='nccl') # 분산학습초기화, nccl for using multi-gpus
        world_size = torch.distributed.get_world_size()
        n_gpu = 1

    if local_rank in [-1, 0]:
        os.makedirs(result_dir, exist_ok=True)
        writer = SummaryWriter(result_dir)

    if local_rank not in [-1, 0]:
        torch.distributed.barrier()


    train_sampler = RandomSampler if local_rank==-1 else DistributedSampler

    
    transform_train = TransformFix(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
        ])
    # Get dataset for CIFAR10
    train_labeled_set, train_unlabeled_set, val_set = get_cifar10('./data', n_labeled, transform_train=transform_train, transform_val=transform_val, mode="train")
    labeled_trainloader = DataLoader(train_labeled_set, batch_size=batch_size, shuffle=True, num_workers=0,drop_last=True, sampler=train_sampler(train_labeled_set))
    unlabeled_trainloader = DataLoader(train_unlabeled_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True, sampler=train_sampler(train_unlabeled_set))
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, sampler=SequentialSampler(val_set))
    
    model = WideResNet(num_classes=10)
    model = model.to(device)

    ema_model = WideResNet(num_classes=10)
    ema_model = ema_model.to(device)
    for param in ema_model.parameters():
        param.detach_()

    cudnn.benchmark = True  # looking for optimal algorithms for this device
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=nesterov)
    ema_optimizer= WeightEMA(model, ema_model, lr, alpha=ema_decay)
    start_epoch = 0

    if train_continue == "on":
        model, optimizer, ema_model, best_acc, start_epoch = load(checkpoint, model, ema_model, optimizer)
    
    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)


    writer = SummaryWriter(log_dir)
    step = 0
    
    for epoch in range(start_epoch, num_epoch):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, num_epoch, state['lr']))

        ######################################################################## Train

        train_losses = tnt.meter.AverageValueMeter()
        train_losses_x = tnt.meter.AverageValueMeter()
        train_losses_u = tnt.meter.AverageValueMeter()

        bar = Bar('Training', max=train_iteration)
        labeled_train_iter = iter(labeled_trainloader)
        unlabeled_train_iter = iter(unlabeled_trainloader)
        model.train()
        for batch_idx in range(train_iteration):
            try:
                inputs_x, targets_x = labeled_train_iter.next()
            except:
                labeled_train_iter = iter(labeled_trainloader)
                inputs_x, targets_x = labeled_train_iter.next()


            try:
                (inputs_u_weak, inputs_u_strong), _ = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                (inputs_u_weak, inputs_u_strong), _ = unlabeled_train_iter.next()



            batch_size = inputs_x.size(0)
            inputs = torch.cat((inputs_x, inputs_u_weak, inputs_u_strong)).to(device)
            targets_x = targets_x.to(device)
            logits = model(inputs)
            logits_x = logits[:batch_size]
            logits_u_weak, logits_u_strong = logits[batch_size:].chunk(2)
            
            Lx = F.cross_entropy(logits_x, targets_x, reduction="mean")

            pseudo_label = torch.softmax(logits_u_weak.detach_(), dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1) # pseudo-label
            mask = max_probs.ge(threshold).float() # max_probs >= threshold인지 element-wise로 비교하여 BoolTensor를 리턴함.
            
            Lu = (F.cross_entropy(logits_u_strong, targets_u, reduction="none") * mask).mean()

            loss =  Lx + lambda_u * Lu

            train_losses.add(loss.item())
            train_losses_x.add(Lx.item())
            train_losses_u.add(Lu.item())

            model.zero_grad()
            loss.backward()
            optimizer.step()
            ema_optimizer.step()

            bar.suffix = '({batch}/{size}) Loss: {loss:.4f} | Loss_x : {loss_x:.4f} | Loss_u : {loss_u:.4f} | W: {w:.4f}'.format(
                    batch=batch_idx+1,
                    size=train_iteration,
                    loss=train_losses.value()[0],
                    loss_x=train_losses_x.value()[0],
                    loss_u=train_losses_u.value()[0],
                    w=ws.value()[0])
            bar.next()
        bar.finish()

        train_loss, train_loss_x, train_loss_u = train_losses.value()[0], train_losses_x.value()[0], train_losses_u.value()[0]



        ################################################## validate
        T_val_losses = tnt.meter.AverageValueMeter()
        V_val_losses = tnt.meter.AverageValueMeter()
        T_val_acc = tnt.meter.ClassErrorMeter(accuracy=True, topk=[1,5])
        V_val_acc = tnt.meter.ClassErrorMeter(accuracy=True, topk=[1,5])
        bar_T = Bar('Train Stats', max=len(labeled_trainloader))
        bar_V = Bar('Valid Stats', max=len(val_loader))
    
        ema_model.eval()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(labeled_trainloader):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = ema_model(inputs)
                loss = F.cross_entropy(outputs, targets)
            
                T_val_losses.add(loss.item())
                T_val_acc.add(outputs.data.cpu(), targets.cpu().numpy())
            
                bar_T.suffix = '({batch}/{size}) Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx+1,
                size=len(labeled_trainloader),
                loss=T_val_losses.value()[0],
                top1=T_val_acc.value(k=1),
                top5=T_val_acc.value(k=5))
               
                bar_T.next()
            bar_T.finish()

            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = ema_model(inputs)
                loss = F.cross_entropy(outputs, targets)

                V_val_losses.add(loss.item())
                V_val_acc.add(outputs.data.cpu(), targets.cpu().numpy())

                bar_V.suffix = '({batch}/{size}) Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx+1,
                    size=len(val_loader),
                    loss=V_val_losses.value()[0],
                    top1=V_val_acc.value(k=1),
                    top5=V_val_acc.value(k=5))

                bar_V.next()
            bar_V.finish()

        train_acc = T_val_acc.value(k=1)
        val_loss, val_acc = V_val_losses.value()[0], V_val_acc.value(k=1)
       

        step = train_iteration * (epoch + 1)
        writer.add_scalar('losses/train_loss', train_loss, step)
        writer.add_scalar('losses/val_loss', val_loss, step)
        writer.add_scalar('accuracy/train_acc', train_acc, step)
        writer.add_scalar('accuracy/val_acc', val_acc, step)

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save(result_dir, epoch, model, ema_model, val_acc, best_acc, is_best, optimizer)

    writer.close()

    print('Best acc :')
    print(best_acc)


def test(args):
    ## Parameter
    state = {k: v for k, v in args._get_kwargs()}
    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch

    mode = args.mode
    train_continue = args.train_continue
    T = args.T
    n_labeled = args.n_labeled
    result_dir = args.result_dir
    data_dir = args.data_dir
    train_iteration = args.train_iteration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    np.random.seed(0)

    if not os.path.isdir(result_dir):
        mkdir_p(result_dir)

    transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
            ])
    test_set = get_cifar10('./data', n_labeled, transform_val=transform_test, mode="test")
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)


    ema_model = WideResNet(num_classes=10)
    ema_model = ema_model.to(device)
    for param in ema_model.parameters():
                param.detach_()

    cudnn.benchmark = True  # looking for optimal algorithms for this device
    
    step = 0
    test_accs = []

    # compute the median accuracy of the last 20 checkpoints of test accuracy
    for i in range(1, 21):
        checkpoint = os.path.join(result_dir, 'checkpoint'+ str(num_epoch - i) + '.pth.tar')
        _, _, ema_model, _, _ = load(checkpoint, None, ema_model, None)

        test_losses = tnt.meter.AverageValueMeter()
        test_acc = tnt.meter.ClassErrorMeter(accuracy=True, topk=[1,5])
        bar = Bar('Test Stats', max=len(test_loader))

        ema_model.eval()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = ema_model(inputs)
                loss = F.cross_entropy(outputs, targets)

                test_losses.add(loss.item())
                test_acc.add(outputs.data.cpu(), targets.cpu().numpy())

                bar.suffix = '({batch}/{size}) Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx+1,
                    size=len(test_loader),
                    loss=test_losses.value()[0],
                    top1=test_acc.value(k=1),
                    top5=test_acc.value(k=5))

                bar.next()
            bar.finish()


        Test_loss, Test_acc = test_losses.value()[0], test_acc.value(k=1)

        test_accs.append(Test_acc)
    print("Mean acc: ")
    print(np.mean(test_accs))


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_stpes, num_cycles=7./16., last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max91,num_warmup_steps))
        no_progress= float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

        return max(0., math.cos(math.pi * num_cycles * no_progress))
    return LamdbaLR(optimizer, _lr_lambda, last_epoch)


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch): 
    # function for correct batchnorm calculation. Because when labeled dataset is too small, mean and std is biased.
    # This function prevents it.
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]
