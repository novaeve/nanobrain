"""Export a legacy configuration into editable graphs; never load/mutate weights."""
from .architecture import factory_parameters, canonical_architecture
from .objectives import default_objectives


def legacy_to_graph(genome, input_dim):
    g = genome.to_dict()
    if g.get("architecture") is not None:
        return g["architecture"]
    width = g["hidden_dim"]
    def spec(kind, name, values):
        accepted = factory_parameters(kind, name)
        return {k: v for k, v in values.items() if k in accepted}
    nodes = []; previous = "features"
    for kind in ("encoder", "attention", "spatial", "residual"):
        params = spec(kind, g[kind], dict(hidden_dim=width, depth=g["encoder_depth"],
                                        heads=g["attention_heads"], dropout=g["dropout"]))
        nodes.append(dict(id=kind, kind=kind, name=g[kind], inputs=[previous], params=params)); previous = kind
        if kind == "attention" and g["encoder"] == "panel_patches":
            nodes.append(dict(id="pool", kind="operation", name="mean_pool", inputs=[previous], params={})); previous = "pool"
    if g["flow"] == "none":
        raise ValueError("choose an active development flow before migration")
    flow_params = spec("flow", g["flow"], dict(hidden_dim=width, heads=g["attention_heads"],
                                              depth=g["flow_depth"], dropout=g["dropout"]))
    if g["flow"] == "fmt":
        flow_params["depth"] = max(2, g["flow_depth"] + g["flow_depth"] % 2)
    heads = {}
    for task in ("annotation", "integration", "integration_reconstruction", "niche", "development", "completion"):
        name = g["classification_head"] if task in {"annotation", "integration", "niche"} else g["regression_head"]
        heads[task] = dict(name=name, params=spec("head", name, {"dropout": g["dropout"]}))
    return canonical_architecture(dict(
        version=2, input_dim=input_dim, width=width,
        backbone=dict(nodes=nodes, output=previous),
        flow=dict(nodes=[dict(id="velocity", kind="flow", name=g["flow"],
                             inputs=["state", "context", "time"], params=flow_params)], output="velocity"),
        pairing=dict(name=g["pairing"], params=spec("pairing", g["pairing"],
                                                  dict(epsilon=g["sinkhorn_epsilon"], iterations=g["sinkhorn_iterations"]))),
        heads=heads, objectives=default_objectives(g["flow_endpoint_weight"], g["flow_expression_weight"],
                                                  g["integration_adversarial_strength"]),
        solver=dict(name="heun", steps=g["rollout_steps"])))
