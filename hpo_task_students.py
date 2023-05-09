import numpy as np

import torch
from torch.quasirandom import SobolEngine
import torchvision

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.constraints import Interval

from botorch.models import SingleTaskGP
from botorch.test_functions import Branin
from botorch.utils.transforms import normalize, unnormalize
from botorch.optim import optimize_acqf
from botorch.acquisition import ExpectedImprovement


class CNN(torch.nn.Module):
    def __init__(
            self,
            leaky_relu_slope: float,
            dropout_p: float,
            n_hidden: int
    ):
        super().__init__()
        self.model = torch.nn.Sequential(
            # Input = 3 x 32 x 32, Output = 32 x 32 x 32
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(leaky_relu_slope),
            # Input = 32 x 32 x 32, Output = 32 x 16 x 16
            torch.nn.MaxPool2d(kernel_size=2),

            # Input = 32 x 16 x 16, Output = 64 x 16 x 16
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(leaky_relu_slope),
            # Input = 64 x 16 x 16, Output = 64 x 8 x 8
            torch.nn.MaxPool2d(kernel_size=2),

            # Input = 64 x 8 x 8, Output = 64 x 8 x 8
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(leaky_relu_slope),
            # Input = 64 x 8 x 8, Output = 64 x 4 x 4
            torch.nn.MaxPool2d(kernel_size=2),

            torch.nn.Flatten(),
            torch.nn.Linear(64 * 4 * 4, n_hidden),
            torch.nn.Dropout(dropout_p),
            torch.nn.LeakyReLU(leaky_relu_slope),
            torch.nn.Linear(n_hidden, 10)
        )


    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    normalize_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)
            )]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root="./CIFAR10/train", train=True,
        transform=normalize_transform,
        download=True
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./CIFAR10/test", train=False,
        transform=normalize_transform,
        download=True
    )

    # Generating data loaders from the corresponding datasets
    batch_size = 128
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # Selecting the appropriate training device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device {device}")


    def black_box_function(x: torch.Tensor) -> float:
        x = x.detach().numpy().tolist()
        x[1] = x[1]
        x[2] = 0.0001 * 10 ** (3 * x[2])
        x[3] = 0.001 * 10 ** (3 * x[3])
        x[4] = max(1, int(x[4] * 4096))  # max 2048 hidden units
        x[5] = max(1, int(x[5] * 20))  # max 20 epochs

        leaky_relu_slope = x[0]
        dropout_p = x[1]
        learning_rate = x[2]
        weight_decay = x[3]
        n_hidden = x[4]
        num_epochs = x[5]

        print(f'leaky_relu_slope: {leaky_relu_slope:.3E}', end='\t')
        print(f'dropout_p: {dropout_p:.3E}', end='\t')
        print(f'learning_rate: {learning_rate:.3E}', end='\t')
        print(f'weight_decay: {weight_decay:.3E}')
        print(f'n_hidden: {n_hidden}')

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=8)

        model = CNN(leaky_relu_slope, dropout_p, n_hidden).to(device)

        # Defining the model hyper parameters
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Training process begins
        train_loss_list = []
        for epoch in range(num_epochs):
            train_loss = 0

            # Iterating over the training dataset in batches
            model.train()
            for i, (images, labels) in enumerate(train_loader):
                # Extracting images and target labels for the batch being iterated
                images = images.to(device)
                labels = labels.to(device)

                # Calculating the model output and the cross entropy loss
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Updating weights according to calculated loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Printing loss for each epoch
            train_loss_list.append(train_loss / len(train_loader))
            print(f"Epoch {epoch + 1}/{num_epochs}: Training loss = {train_loss_list[-1]:.3f}", end='\r')

        model.eval()

        with torch.no_grad():
            losses = []
            # Iterating over the training dataset in batches
            for i, (images, labels) in enumerate(test_loader):
                images = images.to(device)
                y_true = labels.to(device)

                # Calculating outputs for the batch being iterated
                outputs = model(images)

                losses.append(criterion(outputs, y_true))

            test_loss = torch.mean(torch.stack(losses)).detach().cpu()
            print(f"\nTest loss = {test_loss:.3f}")

            # We maximize the negative test loss, i.e., minimize the loss
            return -test_loss


    def get_gp(x_train: torch.Tensor, y_train: torch.Tensor) -> SingleTaskGP:
        # ➡️ TODO : Create a new SingleTaskGP and return the model  ⬅️
        # See https://botorch.org/api/_modules/botorch/models/gp_regression.html#SingleTaskGP for hints
        # You may want to constraint the possible noise in the model since our problems are noiseless.
        # Use for instance the following likelihood when defining your SingleTaskGP.
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        model = ...
        return model


    cheap_function = Branin(negate=True)

    # ➡️ TODO : Set this to 'cheap_function' or 'black_box_function' ⬅️
    function_to_optimize = black_box_function
    dim = 2 if isinstance(function_to_optimize, Branin) else 6

    # draw d+1 initial points
    x_init = SobolEngine(dim, scramble=True).draw(dim + 1)
    fx_init = torch.tensor([function_to_optimize(x) for x in x_init]).reshape(-1, 1)

    x = x_init
    fx = fx_init

    # Set to 500 for cheap function and to 24 for the HPO problem.
    N_BO_STEPS = 500 if isinstance(function_to_optimize, Branin) else 24
    # Set to 1 for the HPO problem
    PRINT_EVERY = 50 if isinstance(function_to_optimize, Branin) else 1

    print('*** Starting Bayesian Optimization ***')
    for gp_iter in range(N_BO_STEPS):
        if gp_iter % PRINT_EVERY == 0:
            print(f'** Iteration {gp_iter + 1}/{N_BO_STEPS} **')
        # We define the bounds for the optimization. We assume that all hyperparameters are between 0 and 1.
        bounds = torch.stack([torch.zeros(dim), torch.ones(dim)]) if not isinstance(function_to_optimize,
                                                                                    Branin) else torch.stack([
            torch.tensor(
                [
                    -5,
                    0]),
            torch.tensor(
                [
                    10,
                    15])])

        # ➡️ TODO : Normalize the x values to be between 0 and 1. You may use the botorch transforms  ⬅️
        # https://botorch.org/api/_modules/botorch/utils/transforms.html
        x_std = ...

        # ➡️ TODO : Normalize the y values to have mean zero and standard deviation one.  ⬅️
        fx_std = ...

        # ➡️ TODO : Create a new GP with the normalized x and y values ⬅️
        gp = ...

        # Your GP model has an attribute `likelihood` which you can use to compute the marginal log likelihood.
        # This attribute gives the term p(y|f,X,l) in the marginal log likelihood which in our case
        # is a Gaussian (see https://docs.gpytorch.ai/en/stable/likelihoods.html#gaussianlikelihood )
        # ➡️ TODO : Define the marginal log likelihood of the model (see https://docs.gpytorch.ai/en/stable/marginal_log_likelihoods.html#exactmarginalloglikelihood ) ⬅️
        mll = ...

        # ➡️ TODO : Define an acquisition function for your model. We'll use the Expected Improvement (see https://botorch.org/api/_modules/botorch/acquisition/analytic.html#ExpectedImprovement )  ⬅️
        ei = ...

        # ➡️ TODO : Optimize the acquisition function. You can use the optimize_acqf function (see https://botorch.org/api/optim.html#botorch.optim.optimize.optimize_acqf ) ⬅️
        # Set the number of candidates to 1, num restarts to 10, and raw_samples to 1024 ️
        x_next, acq_value = ...

        # ➡️ TODO : Unnormalize x_next to be in the original bounds ⬅️
        x_unnorm = unnormalize(x_next, bounds)

        # ➡️ TODO : Evaluate your black-box function on the point suggested by the acquisition function ⬅️
        fx_next = ...
        if gp_iter % PRINT_EVERY == 0:
            print(f'New function value: {fx_next.item():.4f}. Current best: {fx.max().item():.4f}')
            print('\n')
        x = torch.cat([x, x_unnorm])
        fx = torch.cat([fx, fx_next.reshape(-1, 1)])

    x_bo = x
    fx_bo = fx

    # save x_bo and fx_bo as csv
    np.savetxt("x_bo.csv", x_bo.detach().cpu().numpy(), delimiter=",")
    np.savetxt("fx_bo.csv", fx_bo.detach().cpu().numpy(), delimiter=",")
