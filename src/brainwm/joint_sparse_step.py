"""Joint sparse supervised step; development/spatial targets are added by adapters.

This step currently combines measured expression reconstruction and canonical
annotation. It is not the completed multimodal foundation training recipe.
"""
from .sparse_objective import mask_expression,masked_expression_loss
from .annotation_targets import annotation_loss


def joint_sparse_step(model,batch,annotation_targets,mask_probability=.15,
                      annotation_weight=1.,generator=None):
    if batch.cells is None:raise ValueError('joint training requires aligned cell metadata')
    masked=mask_expression(batch,mask_probability,generator)
    output=model.forward_sparse(masked)
    if 'canonical_annotation_logits' not in output:
        raise ValueError('model must include the frozen train annotation head')
    targets=annotation_targets.encode(batch.cells,device=batch.values.device)
    expression=masked_expression_loss(output,masked)
    annotation=annotation_loss(output['canonical_annotation_logits'],targets)
    return dict(total=expression+annotation_weight*annotation,
                expression=expression,annotation=annotation,
                supervised_annotation_cells=int(targets['annotation_supervised'].sum()))
