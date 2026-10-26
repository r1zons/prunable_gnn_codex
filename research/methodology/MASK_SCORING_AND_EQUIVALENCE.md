# Mask evaluation, prefix sums, and equivalence checks

**Status:** DRAFT technical note; not an authorization to modify algorithms. Read with the canonical `LOCAL_STRUCTURED_V2_SPEC.md`; the earlier `LOCAL_STRUCTURED_V2_DRAFT.md` is a historical ledger only.

## 1. What 'virtual evaluation' means

The research outcome MUST use actual physical channel deletion (rebuilt GCNConv/SAGEConv with reduced parameter tensor shapes), not a dense masked model.

During static candidate selection there are two distinct possible operations:

- **Surrogate score (recommended for main search):** compute an importance score from the dense weights (or once-computed gradients/HVP) and a prospective full mask, **without forward pass** on each candidate. No smaller model has been created yet; this is only score arithmetic.
- **True candidate validation (expensive):** apply correct masks in a full-sized network and perform inference for each candidate. If all dependencies, biases, branches, activation semantics and model.eval() are handled, this can give the same outputs/accuracy as surgery, up to numerical differences. It still performs dense-shaped computations and MUST NOT be called physical compression.

A standard elementwise parameter mask is not automatically sufficient: a removed hidden channel can retain a bias and a nonzero activation, and the downstream weights for that channel must also be zeroed (or the channel activation explicitly gated), including both GraphSAGE branches where relevant. Do not mask graph vertices, input feature dimensions, or output classes. No BatchNorm/residual in currently audited GCN/SAGE architectures, but verify for any future architecture.

Use masked-forward vs local-surgery comparison as a **unit/integration acceptance test** on a few representative masks. Running validation inference for all ~841 width pairs is NOT the cheap static search.

## 2. The full hidden-channel mask

For two hidden layers with dense widths h1 and h2, define retained-index sets K1⊆{0,...,h1-1}, K2⊆{0,...,h2-1}. Full mask M=(K1,K2), widths a=|K1| and b=|K2|. Input features F and classes C remain fixed.

For a 3-conv GCN with weights W0:[h1,F], W1:[h2,h1], W2:[C,h2]:
- Retained W0: rows K1 and all input columns.
- Retained W1: rows K2, columns K1.
- Retained W2: all output rows and columns K2.
- Hidden bias vectors: entries K1, K2; output bias and other untouched params retained.
- Actual parameter compression is P0-P(M), counted separately from importance.

SAGE has distinct neighbor/root projection weights and bias placement; follow actual module parameter layout, not a guessed formula.

A weight between two removed channels disappears once, not twice. For additive parameter scores q_i, unique-removed score is total score minus retained score: S(M)=Σ_all q_i-Σ_retained(M) q_i. Some methods may use a different optimization direction (signed GraSP).

## 3. One-time rankings + prefix sum lookup

For a static method, compute (once, on dense model) a total order of channel identifiers *within each hidden layer*. The order determines the candidate K1(a) and K2(b) for widths a,b. Rankings based on local scores are a heuristic; they need not find the best joint mask.

Let Q0,Q1,Q2 be per-parameter score tensors for GCN, respecting tensor shapes above. Permute rows/columns consistently by channel order. In Q1, rows follow the second-layer rank and columns the first-layer rank.

Build 2D prefix table P1(b,a)=Σ_{u<b,v<a}Q1[rank2[u],rank1[v]]. Build 1D prefix sums P0(a)=Σ_{u<a,f}Q0[rank1[u],f], P2(b)=Σ_{c,v<b}Q2[c,rank2[v]], and matching hidden bias contributions Pb1(a),Pb2(b).

Then total retained score for width pair (a,b):
    Tkeep(a,b) = P0(a) + P1(b,a) + P2(b) + Pb1(a) + Pb2(b) + any fixed (output/other) score.
Unique-removed score:
    S(a,b) = Tall - Tkeep(a,b).

Sanity example: Q1=[[1,2,3],[4,5,6],[7,8,9]] after ranking. Retaining first 2 rows and first 2 columns gives 1+2+4+5=12, obtained directly from prefix P1(2,2).

Preprocessing scans dense score tensors once and builds small prefix tables; after that checking each of (h1-min+1)(h2-min+1)=29²=841 width pairs at H32/min4 is O(1) in number of parameter-score accesses per pair (plus bookkeeping). **Not** 841 neural-network forward passes. General ranking/sorting costs are extra.

Do not claim that prefix sums are universally applicable: they require fixed channel rankings and additive score tensors. Verify signed GraSP score interpretation and orientation separately, and do not use a score oracle to choose Random baselines.

## 4. Method-specific interpretations requiring final scientific review

- **Magnitude:** natural surrogate q_i=|w_i| for prunable tensors. Decide whether to include bias importance in scoring; parameter-count accounting includes removed biases regardless.
- **SNIP-based PT:** once compute q_i=|w_i*g_i| from train loss; use fixed train split, reproducible mode and seed; disclose post-training + structured adaptation rather than original pretraining SNIP.
- **GraSP-based PT (conditional):** compute signed -w_i(Hg)_i; original GraSP optimized preserving gradient flow before training. Need independent confirmation of aggregate over deleted structural parameter groups, rank direction, fixed-mask approximation, sign and dropout/data choices. Existing legacy source does not implement correct H*g and drops sign.
- **Random:** sample candidate mask/widths from a predeclared distribution and log RNG seed; do NOT compute 841 random scores and select the random maximum as an implicit optimizer unless explicitly defined and disclosed.

## 5. Tests

- For several valid mask pairs, do forward with a correct full-size masked reference (or explicit hidden-activation gates) and compare to the physically smaller model at eval with common edges/features: logits allclose; predictions equal; accuracy equal. Check selected channels with nonzero bias and cases where both adjacent channels are removed.
- Compare direct unique-parameter enumeration S(M) against prefix lookup S(a,b) for random/permuted rankings, all width boundary values, both architectures, signed and unsigned toy scores. Test ties, NaN/Inf and bias variants.
- Compare physical param count with analytic formula/checkpoint metadata; do not conflate score parameter set with physical count.
- Evaluate impact of stochastic dropout: run scoring in a predeclared deterministic mode. Validation and test must never influence static mask search.
- Benchmark scoring time vs mask search vs physical surgery vs inference time separately.

## 6. Scientific limitation

The fixed per-layer ranking allows only one candidate mask per width pair. Different channels can interact through shared weights/nonlinear computations. Prefix sums compute the chosen additive surrogate efficiently; they do not prove optimality of mask or predictive correctness for actual accuracy.
