Evolve the existing Cajal v3 shared encoder and its five original joint pretraining objectives plus niche composition.
Start from the actual frozen Cajal v3 architecture/configuration, not its trained weights,
a nanoBrain graph or independent task models. Every new candidate trains from random initialization.
Each candidate has ONE set of weights per seed. You may change allowed Cajal architecture/training
fields and add tensor-only PyTorch token modules, gene/classification losses or an optimizer.
Candidate model/train/data/components values are complete overrides relative to the frozen Cajal v3, not patches
relative to a parent: retain every desired ancestral change explicitly, including plugin sources.
The current Cajal unified_1m data, row identities, gene vocabularies, measurement panels, masks,
label hierarchies, six active objectives, seed 0, 30k LR schedule stopped at 10k and official
evaluation code are immutable. Allowed data fields tune training minibatch/sampling sizes only;
they never select an old 100k subset. The current campaign uses ONE seed, not three. Its scores
are single-seed exploratory evidence; do not invent a standard deviation or significance.
Use the merged component catalogue to combine original nanoBrain parts and newly split Cajal,
scVI and scANVI parts. Respect declared compatible slots and availability; never substitute a
different implementation for an unavailable source. Whole-method baseline scores do not transfer
automatically to a component hybrid. Do not use legacy nanoBrain experiment scores as evidence or
comparison thresholds: those experiments used different data and evaluation protocols. Reuse legacy
component implementations, but use only Cajal-compatible performance results in the LLM context.
Reuse the existing Cajal baseline scores and the saved Cajal v3 10k result for the original six
metrics. The niche comparison uses a fixed composition prior computed from the training
neighbourhoods. Report the result as the prior score. Ancestry transfers architecture, source and
hyperparameters only.
No historical or externally pretrained checkpoint may initialize, supervise or be combined with
a candidate. Historical teacher_gram distillation is unavailable. Only a candidate's own interrupted
training may resume its same-seed checkpoint. The deliverable is a self-contained architecture and
fully specified training recipe reproducible from the declared training data and random seed.
Self-supervised losses and training schedules remain searchable within the fixed training budget;
they must not depend on a separately trained model. Retain desired ancestral changes explicitly.
The complete DAG includes every failed/unselected branch and every parent edge. Read details
with paginated tools. Parent recommendations are a mixture of quality, Pareto and novelty;
any fully evaluated valid ancestor can be a parent. This is not UCT/MCTS. No arbitrary shell,
network or dataset access is provided. Custom modules only receive ONE cell's gene token
states [1,L,D]; they cannot see other cells, target labels, batch IDs or dataset objects.
The seven reported metrics include six from the original five-task evaluation. The seventh is
Niche composition MAE, where smaller is better. The niche head predicts frequencies over the
fine cell-type vocabulary from the centre cell's expression, using 16 same-section neighbours
for supervision. The evaluation reads held-out spatial blocks from the same cohort. Its cells
are disjoint from pretraining. Its sections and donors occur in pretraining. The feedback sets are
reused across research rounds. The sealed V4 test is inaccessible. One shared model must improve the
joint frontier. Never combine different architectures' best task results. Valid non-winning
architectures can remain exploration parents. Submit only through submit_candidate after an
accepted CPU synthetic probe. Prose is not a submission. Never invent measurements or ancestry.
No automatic expansion to new data sizes and no upstream recipe adoption is authorized here.
