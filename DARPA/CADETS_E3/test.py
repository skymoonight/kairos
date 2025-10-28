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
from sequence_encoder import SequenceBranchBundle

# Setting for logging
logger = logging.getLogger("reconstruction_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(artifact_dir + 'reconstruction.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


@torch.no_grad()
def test(inference_data,
          memory,
          gnn,
          link_pred,
          neighbor_loader,
          nodeid2msg,
          path,
          seq_branch: SequenceBranchBundle | None = None,
          ):
    has_sequence_context = sequence_context_available(inference_data)

    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

    memory.eval()
    gnn.eval()
    link_pred.eval()
    if seq_branch is not None and seq_branch.enabled:
        seq_branch.encoder.eval()
        seq_branch.classifier.eval()
        seq_branch.fusion.eval()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    time_with_loss = {}  # key: time，  value： the losses
    total_loss = 0
    edge_list = []

    unique_nodes = torch.tensor([]).to(device=device)
    total_edges = 0


    start_time = inference_data.t[0]
    event_count = 0
    pos_o = []

    # Record the running time to evaluate the performance
    start = time.perf_counter()

    for batch in inference_data.seq_batches(batch_size=BATCH):

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        seq_context = None
        if has_sequence_context:
            seq_context = sequence_context_from_batch(batch)
            if seq_context is not None:
                seq_context.validate(src.size(0))
                seq_context = seq_context.to(device=device)
        unique_nodes = torch.cat([unique_nodes, src, pos_dst]).unique()
        total_edges += BATCH

        n_id = torch.cat([src, pos_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, inference_data.t[e_id], inference_data.msg[e_id])

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

        pos_o.append(fused_logits)
        y_pred = fused_logits
        y_true = []
        for m in msg:
            l = tensor_find(m[node_embedding_dim:-node_embedding_dim], 1) - 1
            y_true.append(l)
        y_true = torch.tensor(y_true).to(device=device)
        y_true = y_true.reshape(-1).to(torch.long).to(device=device)

        loss = criterion(y_pred, y_true)
        total_loss += float(loss) * batch.num_events

        # update the edges in the batch to the memory and neighbor_loader
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        # compute the loss for each edge
        fusion_edge_loss = cal_pos_edges_loss_multiclass(fused_logits, y_true)
        graph_edge_loss = cal_pos_edges_loss_multiclass(graph_logits, y_true)
        seq_edge_loss = None
        if seq_logits is not None:
            seq_edge_loss = cal_pos_edges_loss_multiclass(seq_logits, y_true)

        avg_gate = None
        if fusion_gate is not None:
            avg_gate = fusion_gate.mean(dim=1)

        for i in range(len(fused_logits)):
            srcnode = int(src[i])
            dstnode = int(pos_dst[i])

            srcmsg = str(nodeid2msg[srcnode])
            dstmsg = str(nodeid2msg[dstnode])
            t_var = int(t[i])
            edgeindex = tensor_find(msg[i][node_embedding_dim:-node_embedding_dim], 1)
            edge_type = rel2id[edgeindex]
            edge_loss = fusion_edge_loss[i]

            temp_dic = {}
            temp_dic['loss'] = float(edge_loss)
            temp_dic['graph_loss'] = float(graph_edge_loss[i])
            if seq_edge_loss is not None:
                temp_dic['sequence_loss'] = float(seq_edge_loss[i])
            if avg_gate is not None:
                temp_dic['fusion_gate'] = float(avg_gate[i])
            temp_dic['srcnode'] = srcnode
            temp_dic['dstnode'] = dstnode
            temp_dic['srcmsg'] = srcmsg
            temp_dic['dstmsg'] = dstmsg
            temp_dic['edge_type'] = edge_type
            temp_dic['time'] = t_var

            edge_list.append(temp_dic)

        event_count += len(batch.src)
        if t[-1] > start_time + time_window_size:
            # Here is a checkpoint, which records all edge losses in the current time window
            time_interval = ns_time_to_datetime_US(start_time) + "~" + ns_time_to_datetime_US(t[-1])

            end = time.perf_counter()
            time_with_loss[time_interval] = {'loss': loss,

                                             'nodes_count': len(unique_nodes),
                                             'total_edges': total_edges,
                                             'costed_time': (end - start)}

            log = open(path + "/" + time_interval + ".txt", 'w')

            for e in edge_list:
                loss += e['loss']

            loss = loss / event_count
            logger.info(
                f'Time: {time_interval}, Loss: {loss:.4f}, Nodes_count: {len(unique_nodes)}, Edges_count: {event_count}, Cost Time: {(end - start):.2f}s')
            edge_list = sorted(edge_list, key=lambda x: x['loss'], reverse=True)  # Rank the results based on edge losses
            for e in edge_list:
                log.write(str(e))
                log.write("\n")
            event_count = 0
            total_loss = 0
            start_time = t[-1]
            log.close()
            edge_list.clear()

    return time_with_loss

def load_data():
    # graph_4_3 - graph_4_5 will be used to initialize node IDF scores.
    graph_4_3 = torch.load(graphs_dir + "/graph_4_3.TemporalData.simple").to(device=device)
    graph_4_4 = torch.load(graphs_dir + "/graph_4_4.TemporalData.simple").to(device=device)
    graph_4_5 = torch.load(graphs_dir + "/graph_4_5.TemporalData.simple").to(device=device)

    # Testing set
    graph_4_6 = torch.load(graphs_dir + "/graph_4_6.TemporalData.simple").to(device=device)
    graph_4_7 = torch.load(graphs_dir + "/graph_4_7.TemporalData.simple").to(device=device)

    return [graph_4_3, graph_4_4, graph_4_5, graph_4_6, graph_4_7]


if __name__ == "__main__":
    logger.info("Start logging.")

    # load the map between nodeID and node labels
    cur, _ = init_database_connection()
    nodeid2msg = gen_nodeid2msg(cur=cur)

    # Load data
    graph_4_3, graph_4_4, graph_4_5, graph_4_6, graph_4_7 = load_data()

    has_context = all(sequence_context_available(graph) for graph in (graph_4_3, graph_4_4, graph_4_5, graph_4_6, graph_4_7))
    if not has_context:
        logger.warning(
            "Sequence context tensors were missing from at least one inference graph. "
            "Sequence branch scores will be skipped for those batches."
        )

    # load trained model
    loaded = torch.load(f"{models_dir}/models.pt", map_location=device)
    seq_branch = None
    if isinstance(loaded, (list, tuple)) and len(loaded) >= 7:
        memory, gnn, link_pred, neighbor_loader, seq_encoder, seq_classifier, seq_fusion = loaded
        seq_branch = SequenceBranchBundle(
            encoder=seq_encoder,
            classifier=seq_classifier,
            fusion=seq_fusion,
        ).to(device=device)
    elif isinstance(loaded, (list, tuple)) and len(loaded) == 4:
        memory, gnn, link_pred, neighbor_loader = loaded
    else:
        raise RuntimeError("Unexpected model checkpoint format; cannot restore modules.")

    # Reconstruct the edges in each day
    test(inference_data=graph_4_3,
         memory=memory,
         gnn=gnn,
         link_pred=link_pred,
         neighbor_loader=neighbor_loader,
         nodeid2msg=nodeid2msg,
         path=artifact_dir + "graph_4_3",
         seq_branch=seq_branch)

    test(inference_data=graph_4_4,
         memory=memory,
         gnn=gnn,
         link_pred=link_pred,
         neighbor_loader=neighbor_loader,
         nodeid2msg=nodeid2msg,
         path=artifact_dir + "graph_4_4",
         seq_branch=seq_branch)

    test(inference_data=graph_4_5,
         memory=memory,
         gnn=gnn,
         link_pred=link_pred,
         neighbor_loader=neighbor_loader,
         nodeid2msg=nodeid2msg,
         path=artifact_dir + "graph_4_5",
         seq_branch=seq_branch)

    test(inference_data=graph_4_6,
         memory=memory,
         gnn=gnn,
         link_pred=link_pred,
         neighbor_loader=neighbor_loader,
         nodeid2msg=nodeid2msg,
         path=artifact_dir + "graph_4_6",
         seq_branch=seq_branch)

    test(inference_data=graph_4_7,
         memory=memory,
         gnn=gnn,
         link_pred=link_pred,
         neighbor_loader=neighbor_loader,
         nodeid2msg=nodeid2msg,
         path=artifact_dir + "graph_4_7",
         seq_branch=seq_branch)
