import numpy as np, torch
from jepa_abs import sim, metrics, model as M


def test_matched_state_equal_anchor_and_ordered_futures():
    b = sim.make_bundle(1, 20)
    ctx, fut = b.clips(8, [1, 8, 16])
    anchor = ctx[:, -1].reshape(20, b.K, 3)
    assert np.abs(anchor - anchor[:, :1]).max() < 1e-9                 # same anchor state under every law
    img = sim.render_ctx(b, ctx)[:, -1].reshape(20, b.K, -1)
    assert np.abs(img - img[:, :1]).max() == 0                        # identical anchor pixels
    f = fut[:, :, 2].reshape(20, b.K, -1)
    assert (f[:, 0] > f[:, 1]).all() and (f[:, 1] > f[:, 2]).all()    # higher g -> lower z


def test_gravity_sign_and_analytic():
    p = sim.positions(np.zeros((1, 3)), np.zeros((1, 3)), [9.0], [1.0])
    assert abs(p[0, 0, 2] + 4.5) < 1e-12


def test_ars_reference_values():
    S, O = np.array([1., 1.]), np.zeros(2)
    assert abs(metrics.evaluate(S, O, O)["ARS"] - 1) < 1e-12            # oracle-level -> 1
    assert abs(metrics.evaluate(S, S, O)["ARS"]) < 1e-12                # state-level -> 0
    assert metrics.evaluate(S, 2 * S, O)["ARS"] < 0                     # worse than state: negative, unclamped
    assert metrics.evaluate(S * 1e-4, S * 1e-4, O)["status"] == "INVALID_GAP"


def test_future_isolation_and_stop_gradient():
    torch.manual_seed(0)
    net = M.JEPA(); net.eval()
    ctx = torch.rand(2, 4, 3, 32, 32)
    z1, _ = net.z(ctx)
    l, _, _ = net.loss(ctx, torch.rand(2, 3, 32, 32), torch.tensor([3, 4])); l.backward()
    assert torch.allclose(z1, net.z(ctx)[0])                           # targets don't alter z
    assert all(p.grad is None for p in net.tgt.parameters())          # no gradient into target encoder


def test_ema_moves_target():
    net = M.JEPA(); before = [p.clone() for p in net.tgt.parameters()]
    with torch.no_grad():
        for p in net.enc.parameters(): p.add_(1.0)
    net.ema(0.9)
    assert all(not torch.equal(a, b) for a, b in zip(before, net.tgt.parameters()))
