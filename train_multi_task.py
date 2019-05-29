import torch
from torch.utils.data import DataLoader
import data
import models
import configargparse
from tensorboardX import SummaryWriter
import os
from tqdm import tqdm, trange

if __name__ == "__main__":
    tasks = [
        "vua", "snli"
    ]

    parser = configargparse.ArgumentParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument(
        "--config", "-C", type=str, is_config_file=True, required=False,
        help="The config specifying arguments to run with."
    )
    parser.add_argument(
        "--output", "-o", type=str, required=False, default=None,
        help="The directory to store all the output."
    )
    parser.add_argument(
        "--learning-rate", "-l", type=float, required=False, default=0.01,
        help="The learning rate to use during training."
    )
    parser.add_argument(
        "--parallel", "-p", action="store_true", required=False, default=False,
        help="Use parallel training if available."
    )
    parser.add_argument(
        "--no-cuda", action="store_true", required=False, default=False,
        help="Disable the use of cuda during training."
    )
    parser.add_argument(
        "--tasks", "-t", type=str, nargs="+", default=tasks, choices=tasks, required=False,
        help="The tasks to perform during training and the order in which they are performed in an epoch."
    )
    parser.add_argument(
        "--random-training", "-r", action="store_true", default=False, required=False,
        help="Perform the tasks in a random order during training."
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=20, required=False,
        help="The amount of epochs to train for."
    )
    parser.add_argument(
        "--num-workers", "-n", type=int, default=0, required=False,
        help="The number of workers to use for data loading."
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    print("Loading word embeddings.")
    word_embedding_module = models.WordEmbeddingModel(device).to(device)

    print("Loading datasets.")
    if "snli" in args.tasks:
        snli_dataset = {
            "train":      data.SnliDataset(os.path.join("data", "snli", "snli_1.0_train.jsonl")),
            "validation": data.SnliDataset(os.path.join("data", "snli", "snli_1.0_dev.jsonl")),
            "test":       data.SnliDataset(os.path.join("data", "snli", "snli_1.0_test.jsonl")),
        }

        snli_loaders = {
            name: DataLoader(
                dataset,
                shuffle=(name == "train"),
                batch_size=64 if name == "train" else 128,
                num_workers=args.num_workers,
                collate_fn=data.snli_collate_fn
            ) for name, dataset in snli_dataset.items()
        }
    
    if "vua" in args.tasks:
        vua_dataset = {
            "train":      data.VuaSequenceDataset(split="train"),
            "validation": data.VuaSequenceDataset(split="validation"),
            "test":       data.VuaSequenceDataset(split="test"),
        }

        vua_loaders = {
            name: DataLoader(
                dataset,
                shuffle=(name == "train"),
                batch_size=64 if name == "train" else 128,
                num_workers=args.num_workers,
                collate_fn=data.vua_sequence_collate_fn
            ) for name, dataset in vua_dataset.items()
        }

    snli_model = models.SnliModel(word_embedding_module).to(device)
    vua_model = models.VuaSequenceModel(snli_model).to(device)

    vua_optimizer =  torch.optim.Adam(vua_model.parameters(),  lr=args.learning_rate, weight_decay=0.01)
    snli_optimizer = torch.optim.Adam(snli_model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    task_objects = {
        "vua":  (vua_model,  vua_optimizer,  vua_loaders,  torch.nn.CrossEntropyLoss()),
        "snli": (snli_model, snli_optimizer, snli_loaders, torch.nn.CrossEntropyLoss())
    }

    writer = SummaryWriter(args.output)

    for epoch in trange(args.epochs, desc="Epoch"):
        for task in tqdm(args.tasks, "Tasks"):
            model, optimizer, loaders, criterion = task_objects[task]

            for i, batch in tqdm(enumerate(loaders["train"])):
                optimizer.zero_grad()
                inputs = tuple(b.to(device) if type(b) != list else b for b in batch[:-1])
                targets = batch[-1].to(device)

                output = model(*inputs)

                if task == "snli":
                    loss = criterion(output, targets)

                    accuracy = torch.sum(torch.argmax(output, dim=1) == targets).item() / targets.size(0)
                else:
                    loss = criterion(output.view(-1, output.size(2)), targets.view(-1))

                    amount = (targets != -100).nonzero().size(0)

                    accuracy = torch.sum((torch.argmax(output, dim=2) == targets) & (targets != -100)).item() / amount

                loss.backward()

                optimizer.step()

                writer.add_scalar(
                    f"{task}/train/loss", loss.item(), global_step=len(loaders["train"]) * epoch + i + 1
                )
                writer.add_scalar(
                    f"{task}/train/accuracy", accuracy, global_step=len(loaders["train"]) * epoch + i + 1
                )
