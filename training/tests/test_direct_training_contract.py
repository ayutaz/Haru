import torch

from haru_training.contracts import TrainingConfig, configure_trainable
from haru_training.model import training_losses
from test_training_forward import batch, tiny_model
from test_phase4_haru import tiny_trainer


def test_frozen_lm_stays_in_inference_mode_at_configuration_and_each_fit(tmp_path):
    trainer = tiny_trainer(tmp_path, 'head_output')
    assert trainer.model.head.final.proj.training
    assert not trainer.model.lm.training
    assert not trainer.model.codec.training
    trainer.fit(until_step=1)
    assert trainer.model.head.final.proj.training
    assert not trainer.model.lm.training
    assert not trainer.model.codec.training


def test_causal_training_hidden_matches_incremental_inference_given_the_same_past():
    torch.manual_seed(13)
    model = tiny_model()
    configure_trainable(model, TrainingConfig(mode='head_output'))
    values = batch()
    trained = training_losses(model, values, seed=7)['hidden']
    model.eval()
    cache = model.new_cache()
    parts = []
    for position in range(int(values['attention_mask'][0].sum())):
        spans = [values[k][0:1, position:position+1] for k in
                 ('tokens', 'model_latents', 'user_latents', 'model_mask', 'user_mask')]
        parts.append(model(model.embed(*spans), cache).detach())
    torch.testing.assert_close(torch.cat(parts, 1), trained[0:1, :len(parts)], atol=2e-6, rtol=2e-5)


def test_text_loss_does_not_train_audio_output_but_audio_loss_does():
    model = tiny_model()
    configure_trainable(model, TrainingConfig(mode='head_output'))
    result = training_losses(model, batch(), seed=7)
    assert not result['text_loss'].requires_grad
    result['audio_loss'].backward()
    nonzero = {n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert nonzero == {'head.final.proj.weight'}
