import torch
import typing


class _SigmaReparam(torch.nn.Module):
  def __init__(
    self,
    weight: torch.Tensor,
    n_power_iterations: int = 1,
    dim: int = 0,
    eps: float = 1e-12,
  ) -> None:
    super().__init__()
    ndim = weight.ndim
    self.dim = dim if dim >= 0 else dim + ndim
    self.eps = eps
    if ndim > 1:
      self.n_power_iterations = n_power_iterations
      weight_mat = self._reshape_weight_to_matrix(weight)

      _, _, vh = torch.linalg.svd(weight_mat.t(), full_matrices=False)
      self.register_buffer('_u', vh[0].detach())
      _, _, vh = torch.linalg.svd(weight_mat, full_matrices=False)
      self.register_buffer('_v', vh[0].detach())
      self._gamma = torch.nn.Parameter(torch.ones(1))

  def _reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
    assert weight.ndim > 1
    if self.dim != 0:
      weight = weight.permute(self.dim, *(d for d in range(weight.dim()) if d != self.dim))
    return weight.flatten(1)

  @torch.autograd.no_grad()
  def _power_method(self, weight_mat: torch.Tensor, n_power_iterations: int) -> None:
    assert weight_mat.ndim > 1
    for _ in range(n_power_iterations):
      self._u = torch.nn.functional.normalize(torch.mv(weight_mat, self._v), dim=0, eps=self.eps, out=self._u)
      self._v = torch.nn.functional.normalize(torch.mv(weight_mat.t(), self._u), dim=0, eps=self.eps, out=self._v)

  def forward(self, weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim == 1:
      return torch.nn.functional.normalize(weight, dim=0, eps=self.eps)
    weight_mat = self._reshape_weight_to_matrix(weight)
    if self.training:
      self._power_method(weight_mat, self.n_power_iterations)
    u = self._u.clone(memory_format=torch.contiguous_format)
    v = self._v.clone(memory_format=torch.contiguous_format)
    sigma = torch.dot(u, torch.mv(weight_mat, v))
    return weight * self._gamma / sigma

  def right_inverse(self, value: torch.Tensor) -> torch.Tensor:
    return value
  

def sigma_reparam(
  module: torch.nn.Module,
  name: str = 'weight',
  n_power_iterations: int = 1,
  eps: float = 1e-12,
  dim: typing.Optional[int] = None
) -> torch.nn.Module:
  weight = getattr(module, name, None)
  if dim is None:
    if isinstance(module, (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)):
      dim = 1
    else:
      dim = 0
  torch.nn.utils.parametrize.register_parametrization(
    module,
    name,
    _SigmaReparam(weight, n_power_iterations, dim, eps),
  )
  return module


def convert_to_sigma_reparam(m: torch.nn.Module) -> None:
  for name, child in m.named_children():
    if isinstance(child, (torch.nn.Linear,
                          torch.nn.Conv1d,
                          torch.nn.Conv2d)):
      setattr(m, name, torch.nn.utils.spectral_norm(child))
    elif isinstance(child, (torch.nn.LayerNorm,
                            torch.nn.GroupNorm,
                            torch.nn.BatchNorm1d,
                            torch.nn.BatchNorm2d,
                            torch.nn.BatchNorm3d,
                            torch.nn.InstanceNorm1d,
                            torch.nn.InstanceNorm2d,
                            torch.nn.InstanceNorm3d)):
      setattr(m, name, torch.nn.Identity())
    else:
      convert_to_sigma_reparam(child)
