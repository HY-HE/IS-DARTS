import os
import time
import random
import argparse
import torch
from copy import deepcopy
from PIL import ImageFile
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

from xautodl.config_utils import load_config, dict2config, configure2str
from xautodl.datasets import get_datasets, get_nas_search_loaders
from xautodl.procedures import (
    prepare_seed,
    prepare_logger,
    save_checkpoint,
    copy_checkpoint,
    get_optim_scheduler,
)
from xautodl.utils import get_model_infos, obtain_accuracy
from xautodl.log_utils import AverageMeter, time_string, convert_secs2time
from xautodl.models import get_cell_based_tiny_net, get_search_spaces
from nas_201_api import NASBench201API as API

from shrink_metric import add_iim_methods, get_iim, get_iim_nasnet, remove_iim_methods


def search_func(
        xloader, network, criterion, scheduler, w_optimizer, epoch_str, print_freq, logger, gradient_clip):
    data_time, batch_time = AverageMeter(), AverageMeter()
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    network.train()
    end = time.time()
    for step, (inputs, targets) in enumerate(xloader):
        scheduler.update(None, 1.0 * step / len(xloader))
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        # measure data loading time
        data_time.update(time.time() - end)

        w_optimizer.zero_grad()
        _, logits = network(inputs)

        loss = criterion(logits, targets)
        loss.backward()
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(network.parameters(), gradient_clip)
        w_optimizer.step()
        # record
        prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % print_freq == 0 or step + 1 == len(xloader):
            Sstr = ("*SEARCH* " + time_string() + " [{:}][{:03d}/{:03d}]".format(epoch_str, step, len(xloader)))
            Tstr = "Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})".format(
                batch_time=batch_time, data_time=data_time)
            Wstr = "Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]".format(
                loss=losses, top1=top1, top5=top5)
            logger.log(Sstr + " " + Tstr + " " + Wstr)
    return losses.avg, top1.avg, top5.avg


def valid_func(xloader, network, criterion):
    data_time, batch_time = AverageMeter(), AverageMeter()
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    network.eval()
    end = time.time()
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(xloader):
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            # measure data loading time
            data_time.update(time.time() - end)
            # prediction
            _, logits = network(inputs)
            loss = criterion(logits, targets)
            # record
            prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    return losses.avg, top1.avg, top5.avg


def prune_edges(xloader, network, search_model, search_space, step_total, step_count):
    network.eval()
    add_iim_methods(network, search_space)

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(xloader):
            inputs = inputs.cuda(non_blocking=True)
            _, _ = network(inputs)
    if search_space == "nas-bench-201":
        edges = get_iim(search_model)
        gene = search_model.update_mask(edges)
    else:
        normal_edges, reduce_edges = get_iim_nasnet(search_model)
        # prune by rate 1/step_total
        # darts search space: select 2 in 2-5(number of input nodes) * 8(size of search space) input candidates
        # NAS-Bench-201: select 1 in 5(size of search space) input candidates
        gene = search_model.update_mask(normal_edges, reduce_edges, step_total, step_count)

    remove_iim_methods(network, search_space)
    return gene


def main(xargs):
    assert torch.cuda.is_available(), "CUDA is not available."
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(xargs.workers)
    prepare_seed(xargs.rand_seed)
    logger = prepare_logger(args)

    train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1)
    config = load_config(xargs.config_path, {"class_num": class_num, "xshape": xshape}, logger)
    shrink_epoch = list(range(  # set aside 1 epoch
        config.epochs - 1, config.epochs - 1 - args.shrink_steps * args.shrink_intervals, -args.shrink_intervals))
    logger.log("{:}".format(shrink_epoch))

    if xargs.dataset.startswith("cifar"):
        sampler = None
    else:
        sample_size = len(train_data) // 10
        sampler = torch.utils.data.sampler.SubsetRandomSampler(
            np.random.choice(range(len(train_data)), sample_size, replace=False))

    search_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size, num_workers=xargs.workers, pin_memory=True, sampler=sampler)
    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=config.batch_size, num_workers=xargs.workers, pin_memory=True)

    logger.log("||||||| {:10s} ||||||| Search-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}".format(
        xargs.dataset, len(search_loader), len(valid_loader), config.batch_size))
    logger.log("||||||| {:10s} ||||||| Config={:}".format(xargs.dataset, config))

    search_space = get_search_spaces("cell", xargs.search_space_name)   # ops str list
    model_name = "ISDARTS" if xargs.dataset.startswith("cifar") else "ISDARTS_IMAGENET"
    model_config = load_config(xargs.model_config, {
        "name": model_name,
        "num_classes": class_num,
        "space": search_space,
        "affine": False,
        "track_running_stats": bool(xargs.track_running_stats),
    }, None)
    search_model = get_cell_based_tiny_net(model_config)
    search_model.cuda()
    logger.log("search-model :\n{:}".format(search_model))

    w_optimizer, w_scheduler, criterion = get_optim_scheduler(search_model.get_weights(), config)
    logger.log("w-optimizer : {:}".format(w_optimizer))
    logger.log("w-scheduler : {:}".format(w_scheduler))
    logger.log("criterion   : {:}".format(criterion))
    flop, param = get_model_infos(search_model, xshape)
    # logger.log('{:}'.format(search_model))
    logger.log("FLOP = {:.2f} M, Params = {:.2f} MB".format(flop, param))
    if xargs.arch_nas_dataset is None:
        api = None
    else:
        api = API(xargs.arch_nas_dataset)
    logger.log("{:} create API = {:} done".format(time_string(), api))

    last_info, model_base_path, model_best_path = (logger.path("info"), logger.path("model"), logger.path("best"))
    network, criterion = torch.nn.DataParallel(search_model).cuda(), criterion.cuda()

    if last_info.exists():  # automatically resume from previous checkpoint
        logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info))
        last_info = torch.load(last_info)
        start_epoch = last_info["epoch"]
        checkpoint = torch.load(last_info["last_checkpoint"])
        genotypes = checkpoint["genotypes"]
        valid_accuracies = checkpoint["valid_accuracies"]
        search_model.load_state_dict(checkpoint["search_model"])
        w_scheduler.load_state_dict(checkpoint["w_scheduler"])
        w_optimizer.load_state_dict(checkpoint["w_optimizer"])
        logger.log("=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(
            last_info, start_epoch))
    else:
        logger.log("=> do not find the last-info file : {:}".format(last_info))
        start_epoch, valid_accuracies, genotypes = (0, {"best": -1}, {-1: None})

    # start training
    start_time, search_time, epoch_time, total_epoch = (
        time.time(), AverageMeter(), AverageMeter(), config.epochs + config.warmup)
    for epoch in range(start_epoch, total_epoch):
        w_scheduler.update(epoch, 0.0)
        need_time = "Time Left: {:}".format(convert_secs2time(epoch_time.val * (total_epoch - epoch), True))
        epoch_str = "{:03d}-{:03d}".format(epoch, total_epoch)
        logger.log("\n[Search the {:}-th epoch] {:}, LR={:}".format(epoch_str, need_time, min(w_scheduler.get_lr())))

        search_loss, search_top1, search_top5 = search_func(
            search_loader, network, criterion, w_scheduler, w_optimizer,
            epoch_str, xargs.print_freq, logger, xargs.gradient_clip)
        search_time.update(time.time() - start_time)
        logger.log("[{:}] searching : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%, time-cost={:.1f} s".format(
            epoch_str, search_loss, search_top1, search_top5, search_time.sum))
        valid_loss, valid_top1, valid_top5 = valid_func(valid_loader, network, criterion)
        logger.log("[{:}] evaluate  : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%".format(
            epoch_str, valid_loss, valid_top1, valid_top5))

        # check the best accuracy
        valid_accuracies[epoch] = valid_top1
        if valid_top1 > valid_accuracies["best"]:
            valid_accuracies["best"] = valid_top1
            find_best = True
        else:
            find_best = False

        if epoch + 1 in shrink_epoch:
            logger.log("Pruned operations: {:}".format(prune_edges(
                search_loader, network, search_model, xargs.search_space_name,
                len(shrink_epoch), shrink_epoch.index(epoch + 1))))

        if epoch + 1 >= shrink_epoch[-1]:
            genotypes[epoch] = search_model.genotype()
            logger.log("<<<--->>> The {:}-th epoch : {:}".format(epoch_str, genotypes[epoch]))

        # save checkpoint
        save_path = save_checkpoint({
            "epoch": epoch + 1,
            "args": deepcopy(xargs),
            "search_model": search_model.state_dict(),
            "w_optimizer": w_optimizer.state_dict(),
            "w_scheduler": w_scheduler.state_dict(),
            "genotypes": genotypes,
            "valid_accuracies": valid_accuracies,
        }, model_base_path, logger)
        last_info = save_checkpoint({
            "epoch": epoch + 1,
            "args": deepcopy(args),
            "last_checkpoint": save_path,
        }, logger.path("info"), logger)
        if find_best:
            logger.log("<<<--->>> The {:}-th epoch : find the highest validation accuracy : {:.2f}%.".format(
                epoch_str, valid_top1))
            copy_checkpoint(model_base_path, model_best_path, logger)

        with torch.no_grad():
            logger.log("{:}".format(search_model.show_alphas()))
        if epoch + 1 >= shrink_epoch[-1] and api is not None:
            logger.log("{:}".format(api.query_by_arch(genotypes[epoch], "200")))
        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

    logger.log("\n" + "-" * 100)
    logger.log("ISDARTS : run {:} epochs, cost {:.1f} s, last-geno is {:}.".format(
        total_epoch, search_time.sum, genotypes[total_epoch - 1]))
    if api is not None:
        logger.log("{:}".format(api.query_by_arch(genotypes[total_epoch - 1], "200")))
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ISDARTS search")
    parser.add_argument("--data_path", type=str, default="/data/cifar", help="Path to dataset")
    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100", "imagenet-1k"],
                        default="cifar10", help="Choose between Cifar10/100 and imagenet-1k.")
    # channels and number-of-cells
    parser.add_argument("--search_space_name", type=str, default="nas-bench-201",
                        # darts  nas-bench-201
                        help="The search space name.")
    # parser.add_argument("--max_nodes", type=int, default=4, help="The maximum number of nodes.")
    # parser.add_argument("--channel", type=int, default=16, help="The number of channels.")
    # parser.add_argument("--num_cells", type=int, default=5, help="The number of cells in one stage.")
    parser.add_argument("--track_running_stats", type=int, choices=[0, 1],
                        default=0, help="Whether use track_running_stats or not in the BN layer.")
    parser.add_argument("--config_path", type=str, default="configs/nas-benchmark/algos/DARTS.config",
                        # configs/nas-benchmark/algos/DARTS.config configs/search-opts/DARTS-NASNet-${base}.config
                        help="The config path.")
    parser.add_argument("--model_config", type=str, default="configs/search-archs/DARTS-NASBENCH.config",
                        # configs/search-archs/DARTS-NASNet.config  DARTS-NASBENCH.config
                        help="The path of the model configuration.")
    parser.add_argument("--gradient_clip", type=float, default=5, help="")
    # architecture leraning rate
    # parser.add_argument("--arch_learning_rate", type=float, default=3e-4, help="learning rate for arch encoding")
    # parser.add_argument("--arch_weight_decay", type=float, default=1e-3, help="weight decay for arch encoding")
    # log
    parser.add_argument("--workers", type=int, default=4, help="number of data loading workers (default: 4)")
    parser.add_argument("--save_dir", type=str, default="./output", help="Folder to save checkpoints and log.")
    parser.add_argument("--number", type=str, default="", help="experiment number")
    parser.add_argument("--print_freq", type=int, default=200, help="print frequency (default: 200)")
    parser.add_argument("--rand_seed", type=int, default=-1, help="manual seed")

    parser.add_argument("--arch_nas_dataset", type=str, default=None,
                        # './NAS-Bench-201-v1_0-e61699.pth'
                        help="The path to load the architecture dataset (nas-benchmark).")

    parser.add_argument("--shrink_steps", type=int, default=4, help="steps to shrink supernet to subnet")
    parser.add_argument("--shrink_intervals", type=int, default=2, help="epochs between shrink steps")

    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)
    args.save_dir = os.path.join(args.save_dir, 'ISDARTS', args.search_space_name, args.number)
    main(args)
