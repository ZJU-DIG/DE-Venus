import torch
from scipy.special import ive
import numpy as np

class VMFEstimator:
    """Online estimator for per-class vMF distributions (mean direction + concentration).

    Maintains running mean directions and concentration parameters (kappa) for
    each class, updated incrementally with new feature vectors.

    Args:
        feature_num: Dimensionality of feature vectors.
        class_num: Number of classes (typically 2: incorrect / correct).
    """

    def __init__(self, feature_num: int, class_num: int):
        self.class_num = class_num
        self.feature_num = feature_num
        self._init_params()

    def _init_params(self):
        self.Ave = torch.nn.functional.normalize(
            torch.randn(self.class_num, self.feature_num), dim=1
        ) * 0.9
        self.Amount = torch.zeros(self.class_num)
        self.kappa = torch.ones(self.class_num) * self.feature_num * 90 / 19
        tem = torch.from_numpy(
            ive(self.feature_num / 2 - 1, self.kappa.cpu().numpy().astype(np.float64))
        ).to(self.kappa.device)
        self.logc = (
            torch.log(tem + 1e-300)
            + self.kappa
            - (self.feature_num / 2 - 1) * torch.log(self.kappa + 1e-300)
        )

    def reset(self):
        self._init_params()

    def update_CV(self, features: torch.Tensor, labels: torch.Tensor):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(N, 1, A).expand(N, C, A)
        onehot = torch.zeros(N, C)
        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)
        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1
        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A)
        weight_AV = sum_weight_AV.div(
            sum_weight_AV + self.Amount.view(C, 1).expand(C, A)
        )
        weight_AV[weight_AV != weight_AV] = 0  # NaN guard

        self.Ave = (self.Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()
        self.Amount += onehot.sum(0)

    def update_kappa(self):
        R = torch.linalg.norm(self.Ave, dim=1)
        self.kappa = self.feature_num * R / (1 - R**2)
        self.kappa[self.kappa > 1e5] = 1e5
        self.kappa[self.kappa < 0] = 1e5

        nu, _ = VMFEstimator._miller_recurrence(
            torch.tensor(self.feature_num / 2 - 1).int(), self.kappa.double()
        )
        self.logc = nu + self.kappa - (self.feature_num / 2 - 1) * torch.log(self.kappa + 1e-20)

    @staticmethod
    def compute_log_ratio(k: torch.Tensor, p: int | float, logc: torch.Tensor) -> torch.Tensor:
        """Compute log(C(kappa') / C(kappa)) for vMF normalizing constants."""
        p_tensor = torch.tensor(p, dtype=torch.float32)
        nu, _ = VMFEstimator._miller_recurrence((p_tensor / 2 - 1).int(), k.double())
        return nu + k - (p_tensor / 2 - 1) * torch.log(k + 1e-20) - logc

    @staticmethod
    def _miller_recurrence(order, kappa):
        kappa = kappa.double()
        device = kappa.device
        shape = kappa.shape

        I_n = torch.ones(shape, dtype=torch.float64, device=device)
        I_n1 = torch.zeros(shape, dtype=torch.float64, device=device)

        Estimat_n = [torch.zeros(shape, dtype=torch.float64, device=device),
                     torch.zeros(shape, dtype=torch.float64, device=device)]
        scale0 = torch.zeros(shape, dtype=torch.float64, device=device)
        scale1 = torch.zeros(shape, dtype=torch.float64, device=device)
        scale = torch.zeros(shape, dtype=torch.float64, device=device)

        for i in range(2 * order, 0, -1):
            I_n_tem = 2 * i / kappa * I_n + I_n1
            I_n1_tem = I_n.clone()

            overflow = torch.isinf(I_n_tem)
            if overflow.any():
                log_I_n = torch.log(I_n)
                scale = torch.where(overflow, scale + log_I_n, scale)
                if i >= (order + 1):
                    scale0 = torch.where(overflow, scale0 + log_I_n, scale0)
                    scale1 = torch.where(overflow, scale1 + log_I_n, scale1)
                elif i == order:
                    scale0 = torch.where(overflow, scale0 + log_I_n, scale0)

                I_n1_rescaled = I_n1 / I_n
                I_n_rescaled = torch.ones(shape, dtype=torch.float64, device=device)
                I_n_new_rescaled = 2 * i / kappa * I_n_rescaled + I_n1_rescaled
                I_n1_new_rescaled = I_n_rescaled

                I_n = torch.where(overflow, I_n_new_rescaled, I_n_tem)
                I_n1 = torch.where(overflow, I_n1_new_rescaled, I_n1_tem)
            else:
                I_n, I_n1 = I_n_tem, I_n1_tem

            if i == order:
                Estimat_n[0] = I_n1.clone()
            elif i == (order + 1):
                Estimat_n[1] = I_n1.clone()

        ive0 = torch.special.i0e(kappa)
        Estimat_n[0] = torch.log(ive0) + torch.log(Estimat_n[0]) - torch.log(I_n) + scale0 - scale
        Estimat_n[1] = torch.log(ive0) + torch.log(Estimat_n[1]) - torch.log(I_n) + scale1 - scale

        return Estimat_n[0], Estimat_n[1]
