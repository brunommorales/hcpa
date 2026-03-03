from __future__ import annotations

import shutil
import unittest
from pathlib import Path

import torch

from hcpa_monai import TrainConfig, build_model, train_and_evaluate


class BaselineMonaiSanityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path("/tmp/hcpa_opt_test")
        if self.tmp.exists():
            shutil.rmtree(self.tmp)
        self.tmp.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self.tmp.exists():
            shutil.rmtree(self.tmp)

    def test_forward_shape(self) -> None:
        cfg = TrainConfig(
            results_dir=self.tmp,
            tfrec_dir=Path("/tmp/none"),
            image_size=299,
            num_classes=2,
            model_name="inception_v3",
            pretrained=False,
            batch_size=2,
            eval_batch_size=2,
        )
        model = build_model(cfg)
        x = torch.zeros((2, 3, cfg.image_size, cfg.image_size))
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (2, cfg.num_classes))

    def test_single_epoch_fake_data(self) -> None:
        cfg = TrainConfig(
            results_dir=self.tmp,
            tfrec_dir=Path("/tmp/none"),
            image_size=299,
            num_classes=2,
            model_name="inception_v3",
            pretrained=False,
            batch_size=2,
            eval_batch_size=2,
            epochs=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            use_fake_data=True,
            fake_train_size=8,
            fake_eval_size=4,
            num_workers=0,
            amp=False,
            compile=False,
            channels_last=False,
            mixup_alpha=0.0,
            cutmix_alpha=0.0,
            gradient_accumulation=1,
        )
        metrics = train_and_evaluate(cfg)
        self.assertIn("best_auc", metrics)


if __name__ == "__main__":
    unittest.main()
