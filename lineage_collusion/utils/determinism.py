"""
決定性設定 (Determinism)
emo2 の知見: bf16 の丸め誤差や cuBLAS ワークスペースの差で、確率出力が介入効果と同程度に揺れる。
本番実行の最初（torch の CUDA 初期化前）に呼び出す。
"""

import os
import random
from typing import Optional


def configure_determinism(seed: int = 42, strict: bool = False) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # TF32 は行列積の丸めを変えるため無効化
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # strict=True では非決定的カーネルで例外、既定は警告のみ
    torch.use_deterministic_algorithms(True, warn_only=not strict)
