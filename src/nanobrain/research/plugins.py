"""Install content-checked tensor plugins in an isolated candidate worker only."""
from __future__ import annotations

import functools
import inspect
import torch
import torch.nn.functional as F

from .code_policy import load_tensor_code
from .contracts import digest, parameter_values, validate_bundle

LOADED = {}


def checked_loss(fn, prediction, target):
    loss = fn(prediction, target)
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not bool(torch.isfinite(loss)) or float(loss.detach()) < -1e-8:
        raise ValueError("custom loss must return one finite nonnegative scalar Tensor")
    if prediction.requires_grad and not loss.requires_grad: raise ValueError("custom loss detached its predictions")
    return loss


def install_plugins(bundle):
    validate_bundle(bundle)
    from nanobrain.registry import registry
    from nanobrain import objectives
    from .optimizers import CUSTOM
    for plugin in bundle["plugins"]:
        name, kind = plugin["name"], plugin["kind"]
        code_hash = digest({"plugin":plugin,"source":bundle["files"][plugin["file"]]})
        if name in LOADED:
            if LOADED[name] != code_hash: raise ValueError("a different implementation needs a fresh worker process")
            continue
        entry = load_tensor_code(bundle["files"][plugin["file"]], plugin["entrypoint"], "candidate_"+code_hash[:16])
        settings = parameter_values(plugin["parameters"], plugin["settings"])
        signature=inspect.signature(entry)
        for key,rule in plugin["parameters"].items():
            if key not in signature.parameters or signature.parameters[key].default!=rule['default']:
                raise ValueError('every declared knob must be an explicit factory/loss argument with matching default')
        if kind in {"encoder","attention","operation","head","flow"}:
            required = {"encoder":{"input_dim","hidden_dim"}, "head":{"hidden_dim","output_dim"}}.get(kind,{"hidden_dim"})
            signature=inspect.signature(entry)
            if not required <= set(signature.parameters): raise ValueError(f"{kind} factory is missing inferred dimension arguments")
            if set(plugin["parameters"]) - set(signature.parameters): raise ValueError("declared parameter missing from factory signature")
            for key, rule in plugin["parameters"].items():
                if signature.parameters[key].default != rule["default"]:raise ValueError("manifest default differs from the actual factory default")
            registry(kind).register(name,entry,input_rank=None if kind in {"attention","operation"} else 2,
                                    output_rank=None if kind in {"attention","operation"} else 2,
                                    source="candidate-code:"+code_hash, notes="Restricted tensor-code plugin")
        elif kind == "optimizer":
            required={"parameters","lr","weight_decay"}
            if not required <= set(inspect.signature(entry).parameters): raise ValueError("custom optimizer factory needs parameters, lr, weight_decay")
            def factory(parameters,lr,weight_decay,entry=entry,plugin=plugin,**params):
                params=parameter_values(plugin["parameters"],params)
                return entry(parameters,lr=lr,weight_decay=weight_decay,**params)
            CUSTOM[name]=factory
        else:
            if not {"prediction","target"} <= set(inspect.signature(entry).parameters): raise ValueError("loss signature requires prediction and target")
            criterion = functools.partial(entry,**settings)
            if kind == "regression_loss":
                objectives.REGRESSION_LOSSES[name]=functools.partial(checked_loss,criterion)
            elif kind == "classification_loss":
                def classification(model,batch,embedding,params,task,criterion=criterion):
                    target=batch.labels["annotation"];mask=batch.label_masks["annotation"] & target.ge(0)
                    logits=model.annotation_head(embedding)
                    return (checked_loss(criterion,logits[mask],target[mask].long()) if bool(mask.any()) else logits.sum()*0.),{}
                objectives.register_objective(name,tasks=("annotation",),compute=classification,defaults={})
            elif kind == "niche_loss":
                def niche(model,batch,embedding,params,task,criterion=criterion):
                    prediction,density=model.predict_niche(embedding);target=batch.labels["niche"];mask=batch.label_masks["niche"]
                    terms=[checked_loss(criterion,prediction[mask[:,s],s],target[mask[:,s],s].float())
                           for s in range(target.shape[1]) if bool(mask[:,s].any())]
                    loss=torch.stack(terms).mean() if terms else prediction.sum()*0.
                    valid=batch.label_masks["niche_density"]
                    if bool(valid.any()):loss=loss+params["density_weight"]*F.mse_loss(density[valid].log1p(),batch.labels["niche_density"][valid].float().log1p())
                    return loss,{"custom_composition_loss":loss.detach()}
                objectives.register_objective(name,tasks=("spatial_niche",),compute=niche,defaults={"density_weight":.1})
                objectives.COMPOSITION_OBJECTIVES.add(name)
        LOADED[name]=code_hash
    return LOADED


def check_custom_parameters(bundle):
    declared={p["name"]:p for p in bundle["plugins"]}
    architecture=bundle["genome"]["architecture"]
    nodes=[*architecture["backbone"]["nodes"],*architecture["flow"]["nodes"]]
    for key,head in architecture["heads"].items():
        if key=="completion":continue  # not a trained core task in this campaign
        nodes.append(head)
        if "pre" in head:nodes.extend(head["pre"]["nodes"])
    used=set()
    for node in nodes:
        name=node["name"]
        if name.startswith("custom_"):
            if name not in declared:raise ValueError("undeclared custom component")
            params={k:v for k,v in node.get("params",{}).items() if k not in {"hidden_dim","input_dim","output_dim"}}
            parameter_values(declared[name]["parameters"],params);used.add(name)
    for task,spec in architecture["objectives"].items():
        if task=="spatial_completion":continue
        if spec["name"].startswith("custom_"):used.add(spec["name"])
        for key,value in spec["params"].items():
            if "regression" in key and isinstance(value,str) and value.startswith("custom_"):used.add(value)
    if bundle["optimizer"]["name"].startswith("custom_"):used.add(bundle["optimizer"]["name"])
    if used != set(declared):raise ValueError("all custom plugins must be connected to the model, loss, or optimizer")
