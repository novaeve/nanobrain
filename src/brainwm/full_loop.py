"""Full-corpus iterative fit with explicit model/task adapters.

The batch factory must be deterministic for (epoch, offset) and yield
(batch, number_of_cells). It must apply the offset before stochastic transforms.
Validation returns (primary_metric, rows_scored). No test callback is accepted.
"""
from pathlib import Path
import torch
from .full_training import FullTrainingProgress
from .full_checkpoint import save_full_checkpoint, load_full_checkpoint


def fit_full(model, optimizer, *, train_batches, train_step, validate,
             train_rows, validation_rows, output, data_fingerprint,
             config_fingerprint, minimum_passes=5, patience=5,
             direction='minimize', resume=None, checkpoint_every=1000,
             after_optimizer_step=None):
    if checkpoint_every<=0:raise ValueError('checkpoint_every must be positive')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    progress=FullTrainingProgress(train_rows,minimum_passes,patience,direction)
    steps=0
    if resume:
        progress,steps=load_full_checkpoint(resume,model=model,optimizer=optimizer,
            expected_rows_per_pass=train_rows,data_fingerprint=data_fingerprint,
            config_fingerprint=config_fingerprint)
    def save(name):
        save_full_checkpoint(output/name,model=model,optimizer=optimizer,progress=progress,
            data_fingerprint=data_fingerprint,config_fingerprint=config_fingerprint,
            optimizer_steps=steps)
    while not progress.converged:
        if not progress.pending_validation:
            model.train()
            for batch,rows in train_batches(progress.completed_passes,progress.rows_this_pass):
                # Check coverage before mutating optimizer state.
                if rows<=0 or progress.rows_this_pass+rows>train_rows:
                    raise ValueError('training iterator exceeds declared population')
                optimizer.zero_grad(set_to_none=True)
                loss=train_step(model,batch)
                if loss.ndim!=0 or not torch.isfinite(loss):raise FloatingPointError('invalid training loss')
                loss.backward()
                optimizer.step()
                if after_optimizer_step is not None:after_optimizer_step(model)
                steps+=1;progress.record_batch(rows)
                if steps%checkpoint_every==0:save('last.pt')
            progress.finish_pass()
            save('last.pt')
        model.eval()
        with torch.no_grad():metric,rows=validate(model)
        improved=progress.record_validation(float(metric),rows=rows,expected_rows=validation_rows)
        if improved:save('best.pt')
        save('last.pt')
    return progress
