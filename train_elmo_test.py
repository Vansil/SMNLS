'''

This file is intended to test the ELMo embedding by training a model on it

'''

import torch
import argparse
from tensorboardX import SummaryWriter
import data
import os
import models
# from tqdm import tqdm
import numpy as np

def close_enough(accuracies):
    for a1 in accuracies:
        for a2 in accuracies:
            if not np.isclose(a1, a2, atol=0, rtol=1.0e-2):
                return False

    return True

def update_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

if __name__ == "__main__":
    
    print("Loading datasets.")
    train_data = data.SnliDataset(os.path.join('data', 'snli', "snli_1.0_train.jsonl"))
    validation_data = data.SnliDataset(os.path.join('data', 'snli', "snli_1.0_dev.jsonl"))
    test_data = data.SnliDataset(os.path.join('data', 'snli', "snli_1.0_test.jsonl"))

    train_loader = data.SnliDataLoader(train_data, batch_size=64)
    validation_loader = data.SnliDataLoader(validation_data, batch_size=64)
    test_loader = data.SnliDataLoader(test_data, batch_size=64)

    learning_rate = 0.1
    shrink_factor = 0.2

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device: {}".format(device))

    base_model = models.BaseModelElmo(1024, 512, 3)
    model = base_model
    print("Model:", model)

    log_dir = os.path.join('output','s001_elmotest')
    checkpoint_directory = os.path.join(log_dir,'checkpoints')
    os.makedirs(checkpoint_directory, exist_ok=True)

    writer = SummaryWriter()

    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    criterion = torch.nn.CrossEntropyLoss()

    epoch = 0

    step = 1

    validation_history = [0, 0, 0, 0]

    print("Starting to train")
    while learning_rate > 1.0e-5:
        optimizer.zero_grad()

        epoch += 1

        for batch in train_loader:
            optimizer.zero_grad()
            input = tuple(d for d in batch[1:])
            target = batch[0].to(device)

            output = model(*input)

            loss = criterion(output, target)
            loss.backward()

            optimizer.step()

            writer.add_scalar("train/loss", loss.item(), step)

            step += 1


            norm = 0
            for p in model.parameters():
                norm += p.view(-1).pow(2).sum()

            elmo_params = []
            for p in model.elmo.elmo.scalar_mix_0.parameters():
                elmo_params.append(p.item())

            print("Epoch {:02d}\tStep {:02d}\tLoss: {:06.2f}\tGrad norm: {:06.2f}\tElmo params: {}".format(epoch,step, loss.item(), norm, elmo_params))


        torch.save(base_model.state_dict(), os.path.join(checkpoint_directory, "epoch-{}.pt".format(epoch)))

        validation_accuracies = []
        validation_sizes = []

        with torch.no_grad():
            for batch in validation_loader:
                input = tuple(d.to(device) for d in batch[1:])
                target = batch[0].to(device)

                output = model(*input)

                accuracy = torch.sum(torch.argmax(output, dim=1) == target).item() / target.size(0)

                validation_accuracies.append(accuracy)
                validation_sizes.append(target.size(0))

        accuracy = np.average(validation_accuracies, weights=validation_sizes)

        writer.add_scalar("validation/accuracy", accuracy, step)
        print("Val accuracy: {}".format(accuracy))

        if accuracy <= validation_history[-1] or close_enough(validation_history[-3:] + [accuracy]):
            learning_rate *= shrink_factor
            print("Lowering learnig rate to {}".format(learning_rate))
            update_learning_rate(optimizer, learning_rate)

        validation_history.append(accuracy)
