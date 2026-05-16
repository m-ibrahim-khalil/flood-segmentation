#!/usr/bin/env bash
# Run on AWS Graviton t4g.small after `git clone` and `scp exports/`
set -e
sudo apt-get update && sudo apt-get install -y python3-pip python3-venv
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install onnxruntime numpy
python - <<'PY'
import json, time
import numpy as np
import onnxruntime as ort
results = {}
for name in ("mobilenetv3_small_fp32", "efficientnet_lite0_fp32", "mobilevit_xxs_fp32",
             "mobilenetv3_small_int8", "efficientnet_lite0_int8", "mobilevit_xxs_int8"):
    path = f"exports/{name}.onnx"
    try:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        x = np.random.randn(1, 3, 256, 256).astype(np.float32)
        for _ in range(20): sess.run(None, {"input": x})
        ts = []
        for _ in range(200):
            t0 = time.perf_counter()
            sess.run(None, {"input": x})
            ts.append((time.perf_counter() - t0) * 1000.0)
        ts = np.array(ts)
        results[name] = {"p50_ms": float(np.percentile(ts, 50)),
                         "p95_ms": float(np.percentile(ts, 95)),
                         "fps": float(1000.0/np.percentile(ts, 50))}
        print(name, results[name])
    except Exception as e:
        results[name] = {"error": str(e)}
print(json.dumps(results, indent=2))
PY
