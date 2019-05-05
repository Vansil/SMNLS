'''

This file is intended to test the ELMo embedding by training a model on it

'''

import torch
import argparse
from tensorboardX import SummaryWriter
import data
import os
import time
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

    batch_size = 1
    train_loader = data.SnliDataLoader(train_data, batch_size=batch_size)
    validation_loader = data.SnliDataLoader(validation_data, batch_size=batch_size)
    test_loader = data.SnliDataLoader(test_data, batch_size=batch_size)

    learning_rate = 0.1
    shrink_factor = 0.2

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device: {}".format(device))

    model = models.BaseModelElmo(1024, 512, 3, device)
    print("Model:", model)
    num_params, num_trainable = models.count_parameters(model)
    print("Number of parameters:\t\t{}\nNumber of trainable parameters:\t{}".format(num_params,num_trainable))

    log_dir = os.path.join('output','s001_elmotest')
    checkpoint_directory = os.path.join(log_dir,'checkpoints')
    os.makedirs(checkpoint_directory, exist_ok=True)

    writer = SummaryWriter('runs/elmo_test_long')
    writer.add_scalar("train/learningrate", learning_rate, 0)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    criterion = torch.nn.CrossEntropyLoss()

    epoch = 0

    step = 0

    validation_history = [0, 0, 0, 0]

    print("Starting to train")
    while learning_rate > 1.0e-5:
        optimizer.zero_grad()

        epoch += 1

        for batch in train_loader:
            t1 = time.time()

            optimizer.zero_grad()
            input = tuple(d for d in batch[1:])
            target = batch[0].to(device)

            output = model(*input)

            loss = criterion(output, target)
            loss.backward()

            optimizer.step()

            t2 = time.time()
            examples_per_second = batch_size/float(t2-t1)

            accuracy = torch.sum(torch.argmax(output, dim=1) == target).item() / target.size(0)            

            step += 1


            norm = 0
            for p in model.parameters():
                norm += p.view(-1).pow(2).sum()

            elmo_params = []
            for p in model.elmo.elmo.scalar_mix_0.parameters():
                elmo_params.append(p.item())

            writer.add_scalar("train/accuracy", accuracy, step)
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/gradnorm", norm, step)
            for i in range(len(elmo_params)):
                writer.add_scalar("train/elmoparam{}".format(i), elmo_params[i], step)

            print("Epoch {:02d}   Step {:02d}   Accuracy: {:.2f}   Loss: {:02.2f}   Grad norm: {:08.2f}   Examples/Sec: {:03.1f}   Elmo params: {}".format(\
                epoch, step, accuracy, loss.item(), norm, examples_per_second, ", ".join(["{:.4f}".format(p) for p in elmo_params])))


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
            writer.add_scalar("train/learningrate", learning_rate, step)
            update_learning_rate(optimizer, learning_rate)

        validation_history.append(accuracy)
