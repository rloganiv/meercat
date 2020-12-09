"""
Entity Linker Training.
"""
import argparse
import logging
import os
import random
import sys
import tempfile

import numpy as np
import torch
import transformers
from tqdm import tqdm

from meercat.models import RelicConfig, RelicModel
import meercat.utils as utils


logger = logging.getLogger(__name__)


def set_random_seeds(random_seed):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def main(args):
    # Handle multi-GPU setup
    world_size = None
    if args.local_rank == -1:
        if args.model_parallel:
            embedding_device = torch.device('cuda:0')
            device = torch.device('cuda:1')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        if args.model_parallel:
            raise RuntimeError('Cannot use model parallel w/ data parallel')
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
        )
        world_size = torch.distributed.get_world_size()
    is_main_process = args.local_rank in [-1, 0]
    logging.basicConfig(level=logging.INFO if is_main_process else logging.WARN)
    logger.warning('Rank: %s - World Size: %s', args.local_rank, world_size)

    # Basic initialization stuff
    set_random_seeds(args.seed)

    if args.entity_vocab is not None:
        entity_vocab = args.entity_vocab
    else:
        entity_vocab = args.model_name
    entity_tokenizer = utils.EntityTokenizer.from_pretrained(entity_vocab)
    counts = torch.tensor(entity_tokenizer.counts, device=device, dtype=torch.float32)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name,
        model_max_length=args.max_length,
        use_fast=True,
    )
    logger.info('Adding separators to tokenizer')
    utils.add_mention_seps(tokenizer)

    logger.info('Loading model')
    config = RelicConfig.from_pretrained(
        args.model_name,
        entity_vocab_size=len(entity_tokenizer),
        entity_embedding_dim=args.entity_embedding_dim,
    )
    model = RelicModel.from_pretrained(
        args.model_name,
        config=config,
    )
    model.resize_token_embeddings(len(tokenizer))  # Opt. resize for entity separators
    model.to(device)
    if args.model_parallel:
        model.entity_embeddings.to(embedding_device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
        )

    logger.info('Loading train')
    # train_data = utils.ELDataset.load(args.train)
    train_data = utils.ELIterableDataset(
        fname=args.train,
        tokenizer=tokenizer,
        entity_tokenizer=entity_tokenizer,
        rank=args.local_rank,
        world_size=world_size,
    )
    logger.info('Loading dev')
    # dev_data = utils.ELDataset.load(args.dev)
    dev_data = utils.ELIterableDataset(
        fname=args.dev,
        tokenizer=tokenizer,
        entity_tokenizer=entity_tokenizer,
        rank=args.local_rank,
        world_size=world_size,
    )
    logger.info('Loading test')
    # test_data = utils.ELDataset.load(args.test)
    test_data = utils.ELIterableDataset(
        fname=args.test,
        tokenizer=tokenizer,
        entity_tokenizer=entity_tokenizer,
        rank=args.local_rank,
        world_size=world_size,
    )

    # if args.local_rank != -1:
        # train_sampler = torch.utils.data.DistributedSampler(train_data)
        # dev_sampler = torch.utils.data.DistributedSampler(dev_data)
        # test_sampler = torch.utils.data.DistributedSampler(test_data)
    # else:
        # train_sampler = torch.utils.data.RandomSampler(train_data)
        # dev_sampler = torch.utils.data.SequentialSampler(dev_data)
        # test_sampler = torch.utils.data.SequentialSampler(test_data)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        # sampler=train_sampler,
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_data,
        batch_size=args.batch_size,
        # sampler=dev_sampler,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        # sampler=test_sampler,
    )
    optimizer = transformers.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    if is_main_process:
        writer = torch.utils.tensorboard.SummaryWriter(
            log_dir=args.output_dir,
        )

    logger.info('Starting training')
    best_dev_accuracy = 0
    n_iter = 0
    for epoch in range(args.epochs):
        logger.info('Epoch: %s', epoch)
        # if args.local_rank != -1:
            # train_sampler.set_epoch(epoch)

        # Train loop
        model.train()
        optimizer.zero_grad()
        correct = torch.tensor(0.0, device=device)
        total = torch.tensor(0.0, device=device)
        total_loss = torch.tensor(0.0, device=device)
        if args.local_rank in [-1, 0]:
            train_loader = tqdm(train_loader, file=sys.stdout)
        for i, model_inputs in enumerate(train_loader):
            n_iter += 1
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            with torch.cuda.amp.autocast(enabled=args.fp16):
                loss, logits, *_ = model(**model_inputs, counts=counts)
                total_loss += (loss * logits.size(0)).detach()
                loss /= args.accumulation_steps  # fp16 adjustment
            total += logits.size(0)
            _, preds = torch.max(logits, dim=-1)
            correct += preds.eq(0).sum()
            if i % args.accumulation_steps == 0:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            if not i % 100 and i != 0:
                # If distrubuted, then need to accumulate results across
                # processes.
                if args.local_rank != -1:
                    torch.distributed.reduce(total_loss, 0)
                    torch.distributed.reduce(correct, 0)
                    torch.distributed.reduce(total, 0)
                if is_main_process:
                    writer.add_scalar('Loss/train', (total_loss / (total + 1e-13)).item(), n_iter)
                    writer.add_scalar('Accuracy/train', (correct / (total + 1e-13)).item(), n_iter)
                # Reset accumulators
                correct = torch.tensor(0.0, device=device)
                total = torch.tensor(0.0, device=device)
                total_loss = torch.tensor(0.0, device=device)

        # Eval loop
        model.eval()
        correct = torch.tensor(0.0, device=device)
        total = torch.tensor(0.0, device=device)
        total_loss = torch.tensor(0.0, device=device)
        if args.local_rank in [-1, 0]:
            dev_loader = tqdm(dev_loader, file=sys.stdout)
        for model_inputs in dev_loader:
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            with torch.no_grad():
                loss, logits, *_ = model(**model_inputs, counts=counts)
                _, preds = torch.max(logits, dim=-1)
                correct += preds.eq(0).sum()
                total += preds.size(0)
                total_loss += loss * preds.size(0)

        # Gather accuracy accross processes
        if args.local_rank != -1:
            torch.distributed.reduce(total_loss, 0)
            torch.distributed.reduce(correct, 0)
            torch.distributed.reduce(total, 0)
        accuracy = (correct / total).item()
        total_loss = (total_loss / total).item()
        if is_main_process:
            writer.add_scalar('Loss/dev', total_loss, epoch)
            writer.add_scalar('Accuracy/dev', accuracy, epoch)

        # Serialize if best
        if accuracy > best_dev_accuracy:
            logger.info('Best dev accuracy so far. Saving.')
            if args.local_rank == -1:
                model.save_pretrained(args.output_dir)
            elif args.local_rank == 0:
                model.module.save_pretrained(args.output_dir)
            if is_main_process:
                config.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                entity_tokenizer.save_pretrained(args.output_dir)
        best_dev_accuracy = max(best_dev_accuracy, accuracy)

    # Final test loop
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training Specific
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--dev', type=str, required=True)
    parser.add_argument('--test', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed',  type=int, default=1234)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=128)

    # Model
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--entity_embedding_dim', type=int, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--entity_vocab', type=str, default=None)

    # Distributed
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--model_parallel', action='store_true')

    args = parser.parse_args()

    main(args)

