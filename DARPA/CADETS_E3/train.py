##########################################################################################
# Some of the code is adapted from:
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
##########################################################################################

import logging

from kairos_utils import *
from config import *
from model import *
from sequence_context import (
    sequence_context_available,
    sequence_context_from_batch,
)
from sequence_encoder import (
    CommandPathSequenceEncoder,
    ConsensusFusionHead,
    SequenceBranchBundle,
    SequenceClassifier,
)

# Setting for logging
logger = logging.getLogger("training_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(artifact_dir + 'training.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def train(train_data,
          memory,
          gnn,
          link_pred,
          optimizer,
          neighbor_loader,
          seq_branch: SequenceBranchBundle | None = None,
          sequence_aux_weight: float = 0.0,
          graph_aux_weight: float = 0.0,
          ):
    has_sequence_context = sequence_context_available(train_data)

    memory.train()
    gnn.train()
    link_pred.train()
    if seq_branch is not None and seq_branch.enabled:
        seq_branch.encoder.train()
        seq_branch.classifier.train()
        seq_branch.fusion.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    for batch in train_data.seq_batches(batch_size=BATCH):
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        seq_context = None
        if has_sequence_context:
            seq_context = sequence_context_from_batch(batch)
            if seq_context is not None:
                seq_context.validate(src.size(0))
                seq_context = seq_context.to(device=device)

        n_id = torch.cat([src, pos_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, train_data.t[e_id], train_data.msg[e_id])
        graph_logits = link_pred(z[assoc[src]], z[assoc[pos_dst]])

        seq_logits = None
        context_mask = None
        if seq_branch is not None and seq_context is not None and seq_branch.enabled:
            context_mask = seq_context.event_mask()
            if context_mask.any():
                seq_repr, _, _ = seq_branch.encoder(
                    seq_context.cmd_tokens,
                    seq_context.cmd_mask,
                    seq_context.path_tokens,
                    seq_context.path_mask,
                )
                seq_logits = seq_branch.classifier(seq_repr)
            else:
                context_mask = None

        fused_logits = graph_logits
        fusion_gate = None
        if seq_branch is not None and seq_logits is not None and seq_branch.enabled:
            fused_logits, fusion_gate = seq_branch.fusion(
                graph_logits,
                seq_logits,
                context_mask,
            )

        y_pred = fused_logits
        y_true = []
        for m in msg:
            l = tensor_find(m[node_embedding_dim:-node_embedding_dim], 1) - 1
            y_true.append(l)

        y_true = torch.tensor(y_true).to(device=device)
        y_true = y_true.reshape(-1).to(torch.long).to(device=device)

        loss = criterion(y_pred, y_true)
        if seq_logits is not None and sequence_aux_weight > 0:
            loss = loss + sequence_aux_weight * criterion(seq_logits, y_true)
        if graph_aux_weight > 0 and fusion_gate is not None:
            loss = loss + graph_aux_weight * criterion(graph_logits, y_true)

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        optimizer.step()
        memory.detach()
        total_loss += float(loss) * batch.num_events
    return total_loss / train_data.num_events

def load_train_data():
    graph_4_2 = torch.load(graphs_dir + "/graph_4_2.TemporalData.simple").to(device=device)
    graph_4_3 = torch.load(graphs_dir + "/graph_4_3.TemporalData.simple").to(device=device)
    graph_4_4 = torch.load(graphs_dir + "/graph_4_4.TemporalData.simple").to(device=device)
    return [graph_4_2, graph_4_3, graph_4_4]


def _set_requires_grad(module, flag: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = flag

def init_models(node_feat_size):
    memory = TGNMemory(
        max_node_num,
        node_feat_size,
        node_state_dim,
        time_dim,
        message_module=IdentityMessage(node_feat_size, node_state_dim, time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=node_state_dim,
        out_channels=edge_dim,
        msg_dim=node_feat_size,
        time_enc=memory.time_enc,
    ).to(device)

    out_channels = len(include_edge_type)
    link_pred = LinkPredictor(in_channels=edge_dim, out_channels=out_channels).to(device)

    parameters = list(memory.parameters()) + list(gnn.parameters()) + list(link_pred.parameters())

    seq_branch: SequenceBranchBundle | None = None
    if enable_sequence_branch:
        seq_branch = SequenceBranchBundle(
            encoder=CommandPathSequenceEncoder(),
            classifier=SequenceClassifier(out_channels),
            fusion=ConsensusFusionHead(out_channels),
        ).to(device)
        parameters += list(seq_branch.parameters())

    optimizer = torch.optim.Adam(
        parameters,
        lr=lr,
        eps=eps,
        weight_decay=weight_decay,
    )

    neighbor_loader = LastNeighborLoader(max_node_num, size=neighbor_size, device=device)

    return memory, gnn, link_pred, seq_branch, optimizer, neighbor_loader

if __name__ == "__main__":
    logger.info("Start logging.")

    # Load data for training
    train_data = load_train_data()

    # Initialize the models and the optimizer
    node_feat_size = train_data[0].msg.size(-1)
    memory, gnn, link_pred, seq_branch, optimizer, neighbor_loader = init_models(node_feat_size=node_feat_size)

    has_context = all(sequence_context_available(graph) for graph in train_data)
    if not has_context:
        logger.warning(
            "Sequence context tensors were not found in one or more training graphs. "
            "The upcoming sequence branch will be disabled for those batches."
        )

    sequence_aux_weight = sequence_aux_loss_weight if enable_sequence_branch else 0.0
    graph_aux_weight = graph_aux_loss_weight if enable_sequence_branch else 0.0

    # train the model
    for epoch in tqdm(range(1, epoch_num+1)):
        if enable_sequence_branch and seq_branch is not None and sequence_warmup_epochs > 0:
            freeze_graph = epoch <= sequence_warmup_epochs
            _set_requires_grad(memory, not freeze_graph)
            _set_requires_grad(gnn, not freeze_graph)
            _set_requires_grad(link_pred, not freeze_graph)
            _set_requires_grad(seq_branch, True)
            if freeze_graph:
                logger.info(
                    "  Epoch: %02d, sequence warm-up active (graph branch frozen)",
                    epoch,
                )
        else:
            _set_requires_grad(memory, True)
            _set_requires_grad(gnn, True)
            _set_requires_grad(link_pred, True)

        for g in train_data:
            loss = train(
                train_data=g,
                memory=memory,
                gnn=gnn,
                link_pred=link_pred,
                optimizer=optimizer,
                neighbor_loader=neighbor_loader,
                seq_branch=seq_branch,
                sequence_aux_weight=sequence_aux_weight,
                graph_aux_weight=graph_aux_weight,
            )
            logger.info(f'  Epoch: {epoch:02d}, Loss: {loss:.4f}')

    # Save the trained model
    if seq_branch is not None and seq_branch.enabled:
        model = [memory, gnn, link_pred, neighbor_loader, seq_branch.encoder, seq_branch.classifier, seq_branch.fusion]
    else:
        model = [memory, gnn, link_pred, neighbor_loader]

    os.system(f"mkdir -p {models_dir}")
    torch.save(model, f"{models_dir}/models.pt")
