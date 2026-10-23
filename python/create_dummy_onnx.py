"""Export a fixed PyTorch model implementing the existing rho/u dummy formula."""

import argparse
from pathlib import Path

import onnx
import torch
from torch import nn


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False, dtype=torch.float32)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[0.5, 0.25]], dtype=torch.float32))
        self.linear.requires_grad_(False)

    def forward(self, inputs):
        return self.linear(inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    model = DummyModel().eval()
    example = torch.tensor([[1.0, 4.0], [2.0, 8.0]], dtype=torch.float32)
    with torch.no_grad():
        reference = 0.5 * example[:, 0:1] + 0.25 * example[:, 1:2]
        torch.testing.assert_close(model(example), reference, rtol=1e-6, atol=0.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example,
        str(args.output),
        input_names=["physical_inputs"],
        output_names=["cooling_rate"],
        dynamic_axes={"physical_inputs": {0: "N"}, "cooling_rate": {0: "N"}},
        opset_version=18,
        dynamo=False,
    )
    onnx.checker.check_model(str(args.output))
    print(f"PyTorch dummy formula check passed; exported: {args.output}")


if __name__ == "__main__":
    main()
