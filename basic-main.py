#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.01 #
#####################################################
import os, time, torch, random, argparse
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy import deepcopy
from pathlib import Path

from xautodl.datasets import get_datasets
from xautodl.config_utils import load_config, obtain_basic_args as obtain_args
from xautodl.procedures import (
    prepare_seed,
    prepare_logger,
    save_checkpoint,
    copy_checkpoint,
)
from xautodl.procedures import get_optim_scheduler, get_procedures
from xautodl.models import obtain_model
from xautodl.nas_infer_model import obtain_nas_infer_model
from xautodl.utils import get_model_infos
from xautodl.log_utils import AverageMeter, time_string, convert_secs2time


def main(args):
    assert torch.cuda.is_available(), "CUDA is not available."
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    # torch.set_num_threads(args.workers)

    prepare_seed(args.rand_seed)
    logger = prepare_logger(args)

    train_data, valid_data, xshape, class_num = get_datasets(
        args.dataset, args.data_path, args.cutout_length
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    # get configures
    model_config = load_config(args.model_config, {"class_num": class_num}, logger)
    optim_config = load_config(args.optim_config, {"class_num": class_num}, logger)

    if args.model_source == "normal":
        base_model = obtain_model(model_config)
    elif args.model_source == "nas":
        base_model = obtain_nas_infer_model(model_config, args.extra_model_path)
    elif args.model_source == "autodl-searched":
        base_model = obtain_model(model_config, args.extra_model_path)
    else:
        raise ValueError("invalid model-source : {:}".format(args.model_source))
    flop, param = get_model_infos(base_model, xshape)
    logger.log("model ====>>>>:\n{:}".format(base_model))
    logger.log("model information : {:}".format(base_model.get_message()))
    logger.log("-" * 50)
    logger.log(
        "Params={:.2f} MB, FLOPs={:.2f} M ... = {:.2f} G".format(
            param, flop, flop / 1e3
        )
    )
    logger.log("-" * 50)
    logger.log("train_data : {:}".format(train_data))
    logger.log("valid_data : {:}".format(valid_data))
    optimizer, scheduler, criterion = get_optim_scheduler(
        base_model.parameters(), optim_config
    )
    logger.log("optimizer  : {:}".format(optimizer))
    logger.log("scheduler  : {:}".format(scheduler))
    logger.log("criterion  : {:}".format(criterion))

    last_info, model_base_path, model_best_path = (
        logger.path("info"),
        logger.path("model"),
        logger.path("best"),
    )
    network, criterion = torch.nn.DataParallel(base_model).cuda(), criterion.cuda()

    if last_info.exists():  # automatically resume from previous checkpoint
        logger.log(
            "=> loading checkpoint of the last-info '{:}' start".format(last_info)
        )
        last_infox = torch.load(last_info)
        start_epoch = last_infox["epoch"] + 1
        last_checkpoint_path = last_infox["last_checkpoint"]
        if not last_checkpoint_path.exists():
            logger.log(
                "Does not find {:}, try another path".format(last_checkpoint_path)
            )
            last_checkpoint_path = (
                last_info.parent
                / last_checkpoint_path.parent.name
                / last_checkpoint_path.name
            )
        checkpoint = torch.load(last_checkpoint_path)
        base_model.load_state_dict(checkpoint["base-model"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        valid_accuracies = checkpoint["valid_accuracies"]
        max_bytes = checkpoint["max_bytes"]
        logger.log(
            "=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(
                last_info, start_epoch
            )
        )
    elif args.resume is not None:
        assert Path(args.resume).exists(), "Can not find the resume file : {:}".format(
            args.resume
        )
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint["epoch"] + 1
        base_model.load_state_dict(checkpoint["base-model"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        valid_accuracies = checkpoint["valid_accuracies"]
        max_bytes = checkpoint["max_bytes"]
        logger.log(
            "=> loading checkpoint from '{:}' start with {:}-th epoch.".format(
                args.resume, start_epoch
            )
        )
    elif args.init_model is not None:
        assert Path(
            args.init_model
        ).exists(), "Can not find the initialization file : {:}".format(args.init_model)
        checkpoint = torch.load(args.init_model)
        base_model.load_state_dict(checkpoint["base-model"])
        start_epoch, valid_accuracies, max_bytes = 0, {"best": -1}, {}
        logger.log("=> initialize the model from {:}".format(args.init_model))
    else:
        logger.log("=> do not find the last-info file : {:}".format(last_info))
        start_epoch, valid_accuracies, max_bytes = 0, {"best": -1}, {}

    train_func, valid_func = get_procedures(args.procedure)

    total_epoch = optim_config.epochs + optim_config.warmup
    # Main Training and Evaluation Loop
    start_time = time.time()
    epoch_time = AverageMeter()
    for epoch in range(start_epoch, total_epoch):
        scheduler.update(epoch, 0.0)
        need_time = "Time Left: {:}".format(
            convert_secs2time(epoch_time.avg * (total_epoch - epoch), True)
        )
        epoch_str = "epoch={:03d}/{:03d}".format(epoch, total_epoch)
        LRs = scheduler.get_lr()
        find_best = False
        # set-up drop-out ratio
        if hasattr(base_model, "update_drop_path"):
            base_model.update_drop_path(
                model_config.drop_path_prob * epoch / total_epoch
            )
        logger.log(
            "\n***{:s}*** start {:s} {:s}, LR=[{:.6f} ~ {:.6f}], scheduler={:}".format(
                time_string(), epoch_str, need_time, min(LRs), max(LRs), scheduler
            )
        )

        # train for one epoch
        train_loss, train_acc1, train_acc5 = train_func(
            train_loader,
            torch.device('cuda'),
            network,
            criterion,
            scheduler,
            optimizer,
            optim_config,
            epoch_str,
            args.print_freq,
            logger,
        )
        # log the results
        logger.log(
            "***{:s}*** TRAIN [{:}] loss = {:.6f}, accuracy-1 = {:.2f}, accuracy-5 = {:.2f}".format(
                time_string(), epoch_str, train_loss, train_acc1, train_acc5
            )
        )

        # evaluate the performance
        if (epoch % args.eval_frequency == 0) or (epoch + 1 == total_epoch):
            logger.log("-" * 150)
            valid_loss, valid_acc1, valid_acc5 = valid_func(
                valid_loader,
                torch.device('cuda'),
                network,
                criterion,
                optim_config,
                epoch_str,
                args.print_freq_eval,
                logger,
            )
            valid_accuracies[epoch] = valid_acc1
            logger.log(
                "***{:s}*** VALID [{:}] loss = {:.6f}, accuracy@1 = {:.2f}, accuracy@5 = {:.2f} | Best-Valid-Acc@1={:.2f}, Error@1={:.2f}".format(
                    time_string(),
                    epoch_str,
                    valid_loss,
                    valid_acc1,
                    valid_acc5,
                    valid_accuracies["best"],
                    100 - valid_accuracies["best"],
                )
            )
            if valid_acc1 > valid_accuracies["best"]:
                valid_accuracies["best"] = valid_acc1
                find_best = True
                logger.log(
                    "Currently, the best validation accuracy found at {:03d}-epoch :: acc@1={:.2f}, acc@5={:.2f}, error@1={:.2f}, error@5={:.2f}, save into {:}.".format(
                        epoch,
                        valid_acc1,
                        valid_acc5,
                        100 - valid_acc1,
                        100 - valid_acc5,
                        model_best_path,
                    )
                )
            num_bytes = (
                torch.cuda.max_memory_cached(next(network.parameters()).device) * 1.0
            )
            logger.log(
                "[GPU-Memory-Usage on {:} is {:} bytes, {:.2f} KB, {:.2f} MB, {:.2f} GB.]".format(
                    next(network.parameters()).device,
                    int(num_bytes),
                    num_bytes / 1e3,
                    num_bytes / 1e6,
                    num_bytes / 1e9,
                )
            )
            max_bytes[epoch] = num_bytes
        if epoch % 10 == 0:
            torch.cuda.empty_cache()

        # save checkpoint
        save_path = save_checkpoint(
            {
                "epoch": epoch,
                "args": deepcopy(args),
                "max_bytes": deepcopy(max_bytes),
                "FLOP": flop,
                "PARAM": param,
                "valid_accuracies": deepcopy(valid_accuracies),
                "model-config": model_config._asdict(),
                "optim-config": optim_config._asdict(),
                "base-model": base_model.state_dict(),
                "scheduler": scheduler.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            model_base_path,
            logger,
        )
        if find_best:
            copy_checkpoint(model_base_path, model_best_path, logger)
        last_info = save_checkpoint(
            {
                "epoch": epoch,
                "args": deepcopy(args),
                "last_checkpoint": save_path,
            },
            logger.path("info"),
            logger,
        )

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

    logger.log("\n" + "-" * 200)
    logger.log(
        "Finish training/validation in {:} with Max-GPU-Memory of {:.2f} MB, and save final checkpoint into {:}".format(
            convert_secs2time(epoch_time.sum, True),
            max(v for k, v in max_bytes.items()) / 1e6,
            logger.path("info"),
        )
    )
    logger.log("-" * 200 + "\n")
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a classification model on typical image classification datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--resume", type=str, default=None, help="Resume path.")
    parser.add_argument("--init_model", type=str, default=None, help="The initialization model path.")
    parser.add_argument("--model_config", type=str, default="./configs/archs/NAS-CIFAR-ISDARTS.config",
                        # ./configs/archs/NAS-IMAGENET-ISDARTS.config
                        help="The path to the model configuration")
    parser.add_argument("--optim_config", type=str, default="./configs/opts/NAS-CIFAR.config",
                        # ./configs/opts/NAS-IMAGENET.config
                        help="The path to the optimizer configuration")
    parser.add_argument("--procedure", type=str, default="basic", help="The procedure basic prefix.")
    parser.add_argument("--model_source", type=str, default="nas", help="The source of model definition.")
    parser.add_argument("--extra_model_path", type=str, default=None,
                        help="The extra model ckp file (help to indicate the searched architecture).")

    parser.add_argument("--dataset", type=str, default="cifar10",
                        # cifar10 cifar100 imagenet-1k
                        help="The dataset name.")
    parser.add_argument("--data_path", type=str, default="/data/cifar", help="The dataset name.")
    parser.add_argument("--cutout_length", type=int, default=16,  # -1
                        help="The cutout length, negative means not use.")
    # Printing
    parser.add_argument("--print_freq", type=int, default=500, help="print frequency (default: 200)")
    parser.add_argument("--print_freq_eval", type=int, default=1000, help="print frequency (default: 200)")
    # Checkpoints
    parser.add_argument("--eval_frequency", type=int, default=1, help="evaluation frequency (default: 200)")
    parser.add_argument("--save_dir", type=str, default="./output", help="Folder to save checkpoints and log.")
    parser.add_argument("--number", type=str, default="", help="experiment number")
    # Acceleration
    parser.add_argument("--workers", type=int, default=4, help="number of data loading workers (default: 8)")
    # Random Seed
    parser.add_argument("--rand_seed", type=int, default=-1, help="manual seed")
    # Optimization options
    parser.add_argument("--batch_size", type=int, default=96,  # 128
                        help="Batch size for training.")
    args = parser.parse_args()

    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)
    assert args.save_dir is not None, "save-path argument can not be None"
    args.save_dir = os.path.join(args.save_dir, "ISDARTS", "train", args.number)
    main(args)