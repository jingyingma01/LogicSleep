from copy import deepcopy
import torch.nn as nn
import torch.jit
import utils.util as util

class TTA(nn.Module):
    def __init__(self, model, optimizer, lambda_logic = 0.01, steps = 1):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.lambda_logic = lambda_logic
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def begin_subject(self):
        self.reset()

    def forward(self, x, B, L, center_mask_B, pair_mask_B, overlap, accumulate_only=False):
        for _ in range(self.steps):
            outputs, loss = forward_and_adapt(x, B, L, center_mask_B, pair_mask_B, overlap,
                                                        self.model, self.optimizer,
                                                        self.lambda_crf, self.lambda_logic, accumulate_only)
        return outputs, loss

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.enable_grad()
def forward_and_adapt(x, B, L, center_mask_B, pair_mask_B, overlap,
                      model, optimizer, lambda_crf, lambda_logic, accumulate_only = False):
    [gs, hs] = x
    logits, embedding, salient_spacial, salient_node, ts_trasaction = model(gs, hs)
    S = logits.size(-1)
    logits_seq = logits.view(B, L, S)

    loss_ent = util.entropy_on_centers(logits_seq, center_mask_B)

    loss_dl2 = util.dl2_logic_loss(logits_seq.softmax(-1), pair_mask_B.bool())
    loss = loss_ent + lambda_logic * loss_dl2

    if accumulate_only:
        loss.backward()
        return logits.detach(), loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        logits_post, *_ = model(gs, hs)
    return logits_post, loss

def collect_params(model):
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

def copy_model_and_optimizer(model, optimizer):
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def disable_dropout(module):
    for m in module.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            m.train(False)

def configure_model(model):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model

def check_model(model):
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"