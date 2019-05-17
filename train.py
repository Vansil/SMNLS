import torch
import argparse
from tensorboardX import SummaryWriter
import data
import os
import models
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from output import OutputWriter

def get_model(model_name):
    if model_name == "base-line":
        return models.BaseModel(300, 512, 3)
    elif model_name == "uni-lstm":
        return models.LstmModel(300, 2048, 512, 3)
    elif model_name == "bi-lstm":
        return models.BiLstmModel(300, 2048, 512, 3)
    else:
        return models.BiLstmMaxModel(300, 2048, 512, 3)

def update_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", "-m", type=str, choices=["base-line", "uni-lstm", "bi-lstm", "bi-lstm-max"],
        default="base-line", required=False,
        help="The type of model to train."
    )
    parser.add_argument(
        "--word-embeddings", "-w", type=str, required=True,
        help="The file containing the word embeddings."
    )
    parser.add_argument(
        "--data-dir", "-d", type=str, required=True,
        help="The directory containing the data to train on."
    )
    parser.add_argument(
        "--parallel", "-p", action="store_true", default=False, required=False,
        help="Turn on parallel training."
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default=None, required=False,
        help="The directory into which tensorboard logs and model checkpoints will be placed."
    )
    parser.add_argument(
        "--embedding-type", "-e", type=str, choices=["ELMo", "GloVe"], default="GloVe",
        help="The type of word embeddings to use."
    )

    args = parser.parse_args()

    print("Loading embeddings.")
    embeddings = data.load_embeddings(args.word_embeddings)

    print("Loading datasets.")
    train_data =      data.SnliDataset_(os.path.join(args.data_dir, "snli_1.0_train.jsonl"), embeddings)
    validation_data = data.SnliDataset_(os.path.join(args.data_dir, "snli_1.0_dev.jsonl"),   embeddings)
    test_data =       data.SnliDataset_(os.path.join(args.data_dir, "snli_1.0_test.jsonl"),  embeddings)

    train_loader = DataLoader(
        train_data, shuffle=True, batch_size=64, num_workers=8,
        collate_fn=data.collate_fn
    )
    validation_loader = DataLoader(
        validation_data, shuffle=False, batch_size=128, num_workers=8,
        collate_fn=data.collate_fn
    )
    test_loader = DataLoader(
        test_data, shuffle=True, batch_size=64, num_workers=8,
        collate_fn=data.collate_fn
    )

    learning_rate = 0.1
    shrink_factor = 0.2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = get_model(args.model).to(device)

    if args.parallel:
        model = torch.nn.DataParallel(base_model).to(device)
    else:
        model = base_model

    writer = SummaryWriter(log_dir=args.output_dir)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.01)

    criterion = torch.nn.CrossEntropyLoss()

    epoch = 0

    step = 1

    validation_history = [0, 0, 0, 0]

    checkpoint_directory = os.path.join(writer.log_dir, args.model)

    os.makedirs(checkpoint_directory, exist_ok=True)

    while learning_rate > 1.0e-5:
        optimizer.zero_grad()

        writer.add_scalar("parameters/learning-rate", learning_rate, step)

        epoch += 1

        for batch in tqdm(train_loader, total=len(train_loader)):
            optimizer.zero_grad()
            input = tuple(d.to(device) for d in batch[1:])
            target = batch[0].to(device)

            output = model(*input)

            loss = criterion(output, target)
            loss.backward()

            optimizer.step()

            writer.add_scalar("train/loss", loss.item(), step)

            accuracy = torch.sum(torch.argmax(output, dim=1) == target).item() / target.size(0)

            writer.add_scalar("train/accuracy", accuracy, step)

            step += 1

        torch.save(base_model.state_dict(), os.path.join(checkpoint_directory, "epoch-{}.pt".format(epoch)))

        validation_accuracies = []
        validation_sizes = []

        with torch.no_grad():
            for batch in tqdm(validation_loader, total=len(validation_loader)):
                input = tuple(d.to(device) for d in batch[1:])
                target = batch[0].to(device)

                output = model(*input)

                accuracy = torch.sum(torch.argmax(output, dim=1) == target).item() / target.size(0)

                validation_accuracies.append(accuracy)
                validation_sizes.append(target.size(0))

        accuracy = np.average(validation_accuracies, weights=validation_sizes)

        writer.add_scalar("validation/accuracy", accuracy, step)

        if accuracy <= validation_history[-1]:
            learning_rate *= shrink_factor
            print("Lowering learnig rate to {}".format(learning_rate))
            update_learning_rate(optimizer, learning_rate)

        validation_history.append(accuracy)
