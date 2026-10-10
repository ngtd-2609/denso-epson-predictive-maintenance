import torch

from experiment.models import Detector, CVAE


def test_detector_and_generator_support_signed_xyz_and_backpropagation():
    torch.set_num_threads(2)
    x = torch.randn(2, 3, 3000)
    y = torch.tensor([1, 5])
    detector = Detector()
    assert detector(x).shape == (2, 6)
    model = CVAE(16)
    recon, mu, logvar = model(x, y)
    assert recon.shape == x.shape
    loss = (recon-x).square().mean() + 0.001 * (-0.5*(1+logvar-mu.square()-logvar.exp())).mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    generated = model.decode(torch.randn(2, 16), y)
    assert generated.shape == x.shape and torch.isfinite(generated).all()


def test_checkpoint_roundtrip_keeps_predictions(tmp_path):
    torch.manual_seed(3)
    model = Detector().eval()
    x = torch.randn(2, 3, 3000)
    path = tmp_path / 'model.pt'
    torch.save(model.state_dict(), path)
    restored = Detector().eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)
