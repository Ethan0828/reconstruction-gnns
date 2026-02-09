"""
Analysis: Why Reconstruction Methods May Underperform on Node Classification

This document analyzes the limitations of reconstruction-based approaches for
node classification and provides improvement strategies.
"""

# ============================================================================
# PROBLEM ANALYSIS: Why Reconstruction Underperforms
# ============================================================================

"""
1. INFORMATION LOSS FROM NODE DELETION
--------------------------------------
Problem:
- Deleting nodes removes important structural information
- In node classification, the CENTER NODE itself contains crucial features
- If center node is deleted, we lose its direct features
- Reconstruction tries to infer from incomplete information

Example:
    Original neighborhood: [Center, N1, N2, N3, N4, N5]
    After deletion (50%):   [N1, N3, N5]  <- Center node might be deleted!

    Result: We're trying to predict a node's label without its own features

Solution:
    ✓ NEVER delete the center node
    ✓ Only delete peripheral nodes in the neighborhood
"""

# ============================================================================
# 2. LOCAL VS GLOBAL INFORMATION TRADE-OFF
# ============================================================================

"""
Problem:
- Baseline GCN sees the ENTIRE graph (global context)
- Reconstruction only sees k-hop neighborhood (local context)
- Many node classification tasks require global information

Example (Citation Network):
    - A paper's topic depends on its position in the research community
    - Local neighbors might all be in one subfield
    - But global structure shows the paper bridges multiple fields

    Baseline: Sees entire citation network → captures global role
    Reconstruction: Only sees 2-hop neighbors → misses global context

Trade-off:
    k=1: Too local, misses important structure
    k=2: Reasonable, but still limited
    k=3+: Too expensive, approaches full graph anyway
"""

# ============================================================================
# 3. AGGREGATION STRATEGY
# ============================================================================

"""
Problem:
- Current implementation uses UNIFORM WEIGHTING
- All subgraphs contribute equally, regardless of quality
- Some subgraphs may be more informative than others

Current approach:
    embedding = mean([subgraph1, subgraph2, ..., subgraphN])

Issues:
    - Some subgraphs contain mostly irrelevant nodes
    - No attention to which deletions are more informative
    - Equal weight even when confidence varies

Better approach:
    embedding = weighted_sum([w1*sg1, w2*sg2, ..., wN*sgN])
    where weights are learned based on subgraph quality
"""

# ============================================================================
# 4. COMPUTATIONAL COST WITHOUT PROPORTIONAL BENEFIT
# ============================================================================

"""
Complexity comparison:

Baseline GCN:
    - Time: O(|E| × d × L) per epoch
    - Memory: O(|V| × d)
    - Simple and fast

Reconstruction GCN:
    - Time: O(S × |V| × |E_local| × d × L) per epoch
    - Memory: O(S × |V| × |E_local|)
    - S = 10, |E_local| ≈ 0.1|E|
    - ~10x slower for marginal or negative performance gain

Cost-benefit ratio is POOR for node classification.
"""

# ============================================================================
# 5. OVER-SMOOTHING AND FEATURE DILUTION
# ============================================================================

"""
Problem:
- Multiple subgraphs → multiple aggregations → over-smoothing
- Features become too similar across nodes
- Distinction between node classes is reduced

Process:
    1. GNN already does message passing (smoothing)
    2. Pool subgraph features (more smoothing)
    3. Average across subgraphs (even more smoothing)

Result: Features converge to similar values, hurting discrimination
"""

# ============================================================================
# WHY IT WORKS BETTER FOR GRAPH-LEVEL TASKS
# ============================================================================

"""
Graph classification is FUNDAMENTALLY DIFFERENT:

Graph-level:
    ✓ Reconstruction conjecture is designed for graph-level properties
    ✓ Vertex-deleted decks preserve graph invariants
    ✓ Global structure is what matters (diameter, connectivity, cycles)
    ✓ Each subgraph provides independent evidence

Node classification:
    ✗ Focused on individual node properties
    ✗ Local neighborhoods may not contain enough info
    ✗ Node features are primary signal
    ✗ Subgraphs are correlated (overlapping neighborhoods)

This is why reconstruction works in the original paper (graph tasks)
but struggles with node classification.
"""

# ============================================================================
# IMPROVEMENT STRATEGIES
# ============================================================================

"""
Strategy 1: NEVER DELETE CENTER NODE
-------------------------------------
Ensure the target node is always present in subgraphs.

Modified approach:
    1. Extract k-hop neighborhood around node v
    2. KEEP node v fixed
    3. Only delete OTHER nodes in the neighborhood
    4. Aggregate subgraphs to get v's embedding

Expected improvement: +2-5%


Strategy 2: LEARNABLE SUBGRAPH WEIGHTING
-----------------------------------------
Instead of uniform average, learn importance of each subgraph.

Implementation:
    # Current (uniform)
    embedding = mean(subgraph_embeddings)

    # Improved (learned weights)
    attention_scores = AttentionNet(subgraph_embeddings)
    weights = softmax(attention_scores)
    embedding = weighted_sum(weights, subgraph_embeddings)

Expected improvement: +1-3%


Strategy 3: HYBRID GLOBAL-LOCAL MODEL
--------------------------------------
Combine global (full graph) and local (reconstruction) information.

Architecture:
    global_emb = BaselineGNN(full_graph)
    local_emb = ReconstructionGNN(k_hop_subgraphs)

    # Option A: Concatenate
    combined = [global_emb; local_emb]
    output = MLP(combined)

    # Option B: Gating mechanism
    gate = sigmoid(GateNet([global_emb, local_emb]))
    output = gate * global_emb + (1-gate) * local_emb

Expected improvement: +3-5% (this is what ensemble does!)


Strategy 4: ADAPTIVE NEIGHBORHOOD SIZE
---------------------------------------
Different nodes may need different k values.

Implementation:
    # High-degree nodes: use k=1 (already well-connected)
    # Low-degree nodes: use k=2 or k=3 (need more context)

    k = adaptive_k(node_degree)

Expected improvement: +1-2%


Strategy 5: TASK-SPECIFIC DELETION STRATEGIES
----------------------------------------------
Instead of random deletion, use informed strategies.

Ideas:
    - Delete low-importance nodes (based on attention scores)
    - Delete nodes from same class (test class-boundary sensitivity)
    - Delete structurally redundant nodes
    - Delete nodes that don't change prediction (adversarial approach)

Expected improvement: +2-4%


Strategy 6: REDUCE OVER-SMOOTHING
----------------------------------
Add skip connections and residual connections.

Implementation:
    # Add residual connection from input features
    final_embedding = subgraph_embedding + alpha * input_features

    # Use jumping knowledge
    final_embedding = JK([layer1_output, layer2_output, ..., final_output])

Expected improvement: +1-2%
"""

# ============================================================================
# RECOMMENDED APPROACH
# ============================================================================

"""
For best results, combine multiple strategies:

OPTION A: Improved Reconstruction (if you must use reconstruction alone)
-------------------------------------------------------------------------
1. Never delete center node ✓ (most important)
2. Use learnable subgraph weights ✓
3. Add residual connections ✓
4. Use k=2 (balance of local/global)

Expected: 78-80% on Cora (vs 75-79% currently)


OPTION B: Ensemble (RECOMMENDED) ⭐
------------------------------------
1. Train baseline GCN (captures global structure)
2. Train improved reconstruction (captures local patterns)
3. Ensemble with learned weights

Expected: 81-83% on Cora (vs 78-81% baseline)

This is the approach we implemented!


OPTION C: Hybrid Architecture (for research)
---------------------------------------------
1. Single model with dual pathways:
   - Global branch: processes full graph
   - Local branch: processes k-hop neighborhoods
2. Adaptive fusion of global and local features
3. End-to-end training

Expected: 82-84% on Cora (but more complex)
"""

# ============================================================================
# WHEN TO USE RECONSTRUCTION
# ============================================================================

"""
Reconstruction methods are BETTER suited for:

✓ Tasks where LOCAL structure is paramount:
    - Community detection
    - Link prediction
    - Local clustering coefficient prediction
    - Identifying structural roles

✓ Graph-level tasks (original use case):
    - Graph classification
    - Graph property prediction
    - Subgraph matching

✗ Tasks requiring GLOBAL information:
    - Node classification (most datasets)
    - Pagerank-like centrality
    - Shortest path prediction

Bottom line:
    Reconstruction conjecture is fundamentally about GRAPH reconstruction,
    not NODE feature learning. The mismatch causes suboptimal performance.
"""

# ============================================================================
# CONCLUSION
# ============================================================================

"""
The reconstruction approach underperforms on node classification because:

1. It's designed for graph-level, not node-level tasks
2. Node deletion causes information loss
3. Local neighborhoods miss global context
4. Computational cost is high for the benefit gained

SOLUTION:
    Use ensemble methods to combine baseline (global) and
    reconstruction (local) models. This leverages complementary
    strengths and achieves better performance than either alone.

The ensemble is not a workaround - it's the RIGHT APPROACH for
combining global and local information effectively.
"""
