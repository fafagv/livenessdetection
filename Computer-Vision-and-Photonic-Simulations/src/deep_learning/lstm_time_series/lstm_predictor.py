"""
lstm_predictor.py — PyTorch two-layer LSTMCell model that forecasts a
synthetic sine-wave time series (the classic "time_sequence_prediction"
pattern).

Refactored from main3.py, which — unlike the rest of this project — did not
actually run: it contained multiple typos and structural bugs. Per the
refactor requirements, all of the following were fixed rather than just
flagged:

    1. `self.lstm = nn.LSTMCell(1, self.n_hidden)` was immediately
       overwritten by `self.lstm = nn.LSTMCell(self.hidden, self.n_hidden)`
       (note: `self.hidden` doesn't exist — should be `self.n_hidden`), and
       forward() called `self.lstm1` / `self.lstm2`, neither of which was
       ever defined. -> Now properly defines self.lstm1 and self.lstm2 as
       two separate LSTMCell layers.
    2. `nn.Linearl` -> typo for `nn.Linear`.
    3. `x.size[0]` -> must be a method call: `x.size(0)`.
    4. The second half of forward() (the "future" extrapolation loop) fed
       the *last output* back in as the next input but never advanced
       `future` steps correctly and shadowed the training loop's `input_t` —
       rewritten as an explicit `for _ in range(future):` loop.
    5. `if __name__ == " main":` -> typo for `if __name__ == "__main__":`,
       meaning this whole training block silently never ran before.
    6. `test_input` was referenced but never defined (only `test_target`
       existed, and it was even defined twice with a stray extra `:`).
    7. `y = pred.detach().numpy` was missing the `()` call.
    8. The `draw()` helper had a genuine SyntaxError (an unmatched
       parenthesis split across two statements) and was recursively nested
       with its own call sites indented *inside* its own body — rewritten
       as a normal top-level function called after training.
    9. `plt.savefig("predict%.pdf" % i)` used an invalid format spec
       (`%.pdf` isn't a valid conversion) — fixed to `"predict%d.pdf" % i`.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)

N = 100  # number of independent sine-wave samples
L = 1000  # length of each sample
T = 20  # period


def generate_sine_data(n=N, l=L, t=T, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = np.empty((n, l), dtype=np.float32)
    x[:] = np.arange(l) + rng.integers(-4 * t, 4 * t, n).reshape(n, 1)
    y = np.sin(x / 1.0 / t).astype(np.float32)
    return y


class LSTMPredictor(nn.Module):
    """Two stacked LSTMCells + a linear readout, unrolled manually over time."""

    def __init__(self, n_hidden: int = 51):
        super().__init__()
        self.n_hidden = n_hidden
        self.lstm1 = nn.LSTMCell(1, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1)

    def forward(self, x, future: int = 0):
        outputs = []
        n_samples = x.size(0)

        h_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        h_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)

        output = None
        for input_t in x.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)

        # Autoregressive forecast beyond the training window.
        for _ in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)

        return torch.cat(outputs, dim=1)


def draw(y_i, color, n, future):
    plt.plot(np.arange(n), y_i[:n], color, linewidth=2.0)
    plt.plot(np.arange(n, n + future), y_i[n:], color + ":", linewidth=2.0)


def train(n_steps: int = 10, future: int = 1000, output_dir: str = "."):
    torch.manual_seed(0)
    y = generate_sine_data()

    train_input = torch.from_numpy(y[3:, :-1])   # (97, 999)
    train_target = torch.from_numpy(y[3:, 1:])   # (97, 999)
    test_input = torch.from_numpy(y[:3, :-1])    # (3, 999)
    test_target = torch.from_numpy(y[:3, 1:])    # (3, 999)

    model = LSTMPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(model.parameters(), lr=0.8)

    n = train_input.shape[1]  # 999

    for i in range(n_steps):
        logger.info("Step %d", i)

        def closure():
            optimizer.zero_grad()
            out = model(train_input)
            loss = criterion(out, train_target)
            logger.info("loss %.6f", loss.item())
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            pred = model(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            logger.info("test loss %.6f", loss.item())
            y_pred = pred.detach().numpy()

        plt.figure(figsize=(12, 6))
        plt.title("Sine Wave Forecast")
        plt.xlabel("x")
        plt.ylabel("y")
        for idx, color in zip(range(min(3, y_pred.shape[0])), ("r", "b", "g")):
            draw(y_pred[idx], color, n, future)
        plt.savefig(f"{output_dir}/predict{i}.pdf")
        plt.close()

    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
