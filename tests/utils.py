import torch
from dataclasses import dataclass


@dataclass
class NumericalAccuracy:
    rtol: float = 5e-3
    atol: float = 1.5e-4
    ftol: float = 1e-3

    def stats_ok(
        self, actual: torch.Tensor, expected: torch.Tensor, chunk_size: int = 1
    ) -> bool:
        adjusted_rtol = min(0.5, self.rtol * chunk_size)

        actual_fp64 = actual.double()
        expected_fp64 = expected.double()

        diff = (actual_fp64 - expected_fp64).abs()
        frob_rel_error = torch.sqrt(torch.sum(diff**2) / torch.sum(expected_fp64**2))
        rel_err_bound = self.atol + adjusted_rtol * expected_fp64.abs()
        if (diff > rel_err_bound).all():
            print(
                f"ERROR: max relative error larger than bound: {diff.max().item():.6f}. "
                f"ATOL={self.atol} RTOL={adjusted_rtol}"
            )
            return False
        if frob_rel_error > self.ftol:
            print(
                f"ERROR: large Frobenius relative error: {frob_rel_error:.6f}. FTOL={self.ftol}"
            )
            return False
        return True
