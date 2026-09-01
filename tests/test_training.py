import pandas as pd
import torch

from training.train_specialist import rank_loss, validate_frame


def test_rank_loss_rewards_correct_order():
    targets = torch.tensor([0.0, 0.5, 1.0])

    correct = rank_loss(torch.tensor([-2.0, 0.0, 2.0]), targets)
    reversed_order = rank_loss(torch.tensor([2.0, 0.0, -2.0]), targets)

    assert correct < reversed_order


def test_validate_frame_accepts_exact_training_contract():
    frame = pd.DataFrame(
        {
            "text1": ["a"],
            "text2": ["b"],
            "target": [0.5],
            "weight": [1.0],
            "category": ["Одежда"],
        }
    )

    validate_frame(frame)
