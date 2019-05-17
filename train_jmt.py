import torch
from torch.utils.data import DataLoader
import data
import models
import configargparse
from tensorboardX import SummaryWriter
import os
from tqdm import tqdm
from output import OutputWriter



if __name__ == "__main__":
    tasks = [
        "pos", "vua", "snli"
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
        "--no-cuda", action="store_true", required=False, default=False,
        help="Disable the use of cuda during training."
    )
    parser.add_argument(
        "--tasks", "-t", type=str, nargs="+", default=tasks, choices=tasks, required=False,
        help="The tasks to perform during training and the order in which they are performed in an epoch."
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=20, required=False,
        help="The amount of epochs to train for."
    )
    parser.add_argument(
        "--num-workers", "-n", type=int, default=0, required=False,
        help="The number of workers to use for data loading."
    )
    parser.add_argument(
        "--epsilon", type=float, default=1.0, required=False,
        help="The base epsilon used for the learning rate schedule."
    )
    parser.add_argument(
        "--rho", type=float, default=0.3, required=False,
        help="The rho parameter used for the learning rate decay."
    )
    parser.add_argument(
        "--delta-classifier", type=float, default=1e-2, required=False,
        help="The delta parameter used for succesive regularization applied on the classifier layers." 
    )
    parser.add_argument(
        "--delta-lstm", type=float, default=1e-3, required=False,
        help="The delta parameter used for succesive regularization applied on the lstm layers."
    )

    args = parser.parse_args()

    arguments = {
        "learning-rate": args.learning_rate,
        "tasks": args.tasks,
        "epochs": args.epochs,
        "num-workers": args.num_workers,
        "epsilon": args.epsilon,
        "rho": args.rho,
        "delta-classifier": args.delta_classifier,
        "delta-lstm": args.delta_lstm
    }

    if args.output:
        arguments["output"] = args.output

    if args.no_cuda:
        arguments["no-cuda"] = True

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    print("Creating model.")
    model = models.JMTModel(device)

    print("Loading datasets.")
    if "pos" in args.tasks:
        pos_dataset = {
            "train": data.PennDataset("train"),
            "validation": data.PennDataset("validation"),
            "test": data.PennDataset("test")
        }
    else:
        pos_loaders = {}

    if "snli" in args.tasks:
        snli_dataset = {
            "train": data.SnliDataset(os.path.join("data", "snli", "snli_1.0_train.jsonl")),
            "validation": data.SnliDataset(os.path.join("data", "snli", "snli_1.0_dev.jsonl")),
            "test": data.SnliDataset(os.path.join("data", "snli", "snli_1.0_test.jsonl"))
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
    else:
        snli_loaders = {}
    
    if "vua" in args.tasks:
        vua_dataset = {
            "train": data.VuaSequenceDataset(split="train"),
            "validation": data.VuaSequenceDataset(split="validation"),
            "test": data.VuaSequenceDataset(split="test")
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
    else:
        vua_loaders = {}

    lr_function = lambda epoch: args.epsilon / (1.0 * args.rho * (epoch))

    pos_optimizer = torch.optim.SGD(
        [
            {"params": model.pos_lstm.parameters(), "weight_decay": 1e-6, "lr": 1.0},
            {"params": model.pos_classifier.parameters(), "weight_decay": 1e-5, "lr": 1.0},
        ]
    )
    pos_lr_schedule = torch.optim.lr_scheduler.LambdaLR(
        pos_optimizer, lr_function
    )

    vua_optimizer = torch.optim.SGD(
        [
            {"params": model.pos_lstm.parameters(), "weight_decay": 1e-6, "lr": (1.0 - 1e-3)},
            {"params": model.pos_classifier.parameters(), "weight_decay": 1e-5, "lr": (1.0 - 1e-2)},
            {"params": model.metaphor_lstm.parameters(), "weight_decay": 1e-6, "lr": 1.0},
            {"params": model.metaphor_classifier.parameters(), "weight_decay": 1e-5, "lr": 1.0}
        ],
        lr=1
    )
    vua_lr_schedule = torch.optim.lr_scheduler.LambdaLR(
        vua_optimizer, lr_function
    )

    snli_optimizer = torch.optim.SGD(
        [
            {"params": model.metaphor_lstm.parameters(), "weight_decay": 1e-6, "lr": (1.0 - 1e-2)},
            {"params": model.metaphor_classifier.parameters(), "weight_decay": 1e-5, "lr": (1.0 - 1e-3)},
            {"params": model.pos_lstm.parameters(), "weight_decay": 1e-6, "lr": (1.0 - 1e-2)},
            {"params": model.pos_classifier.parameters(), "weight_decay": 1e-5, "lr": (1.0 - 1e-2)},
            {"params": model.snli_lstm.parameters(), "weight_decay": 1e-6, "lr": 1.0},
            {"params": model.snli_classifier.parameters(), "weight_decay": 1e-5, "lr": 1.0}
        ],
    )
    snli_lr_schedula = torch.optim.lr_scheduler.LambdaLR(
        snli_optimizer, lr_function
    )

    task_objects = {
        "pos": (model.pos_forward, pos_optimizer, pos_lr_schedule, pos_loaders, torch.nn.CrossEntropyLoss())
        "vua": (model.metaphor_forward, vua_optimizer, vua_lr_schedule, vua_loaders, torch.nn.CrossEntropyLoss()),
        "snli": (model.snli_forward, snli_optimizer, snli_lr_schedula, snli_loaders, torch.nn.CrossEntropyLoss())
    }

    writer = OutputWriter(args.output)

    writer.save_arguments(arguments)

    for epoch in tqdm(range(args.epochs), "Epoch"):
        for task in tqdm(args.tasks, "Tasks"):
            model, optimizer, lr_scheduler, loaders, criterion = task_objects[task]

            lr_scheduler.step()
            model.train()

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

            model.eval()

            if "validation" in loaders:
                with torch.no_grad():
                    accuracies = []
                    losses = []
                    batch_sizes = []

                    for batch in tqdm(loaders["validation"]):
                        inputs = tuple(b.to(device) if type(b) != list else b for b in batch[:-1])
                        targets = batch[-1].to(device)

                        if task == "snli":
                            loss = criterion(output, targets)

                            accuracy = torch.sum(torch.argmax(output, dim=1) == targets).item() / targets.size(0)
                        else:
                            loss = criterion(output.view(-1, output.size(2)), targets.view(-1))

                            amount = (targets != -100).nonzero().size(0)

                            accuracy = torch.sum((torch.argmax(output, dim=2) == targets) & (targets != -100)).item() / amount

                        losses.append(loss.item())
                        accuracies.append(accuracy)
                        batch_sizes.append(targets.size(0))

                    loss = np.average(losses, weights=batch_sizes)
                    accuracy = np.average(accuracies, weights=batch_sizes)
                        
                    writer.add_scalar(
                        f"{task}/validation/loss", loss.item(), global_step=len(loaders["train"]) * (epoch + 1)
                    )
                    writer.add_scalar(
                        f"{task}/validation/accuracy", accuracy, global_step=len(loaders["train"]) * (epoch + 1)
                    )

                    writer.save_model(model, "{}_epoch{:02d}".format(task, epoch+1))
