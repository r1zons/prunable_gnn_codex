# New scientific namespace: structural_budget_v5

This is a new harmonized-controller comparison, not a rerun of native V2 or the
threshold Lookahead audit. Every method gets an explicit new implementation ID.
Historical files and evidence are immutable. No previous Q result is a corrected
baseline and no Dense model is retrained.

**Common geometry.** For widths w, count every weight and bias of the physically
reconstructed model. Integer parameter limits use rational target/tolerance,
avoiding floating-point endpoint ambiguity. Enumerate all legal width tuples,
exclude Dense, and retain those within±1pp. If none exist, retain all globally
closest feasible tuples; never widen the band or claim success. Static methods
choose among this goal set. Adaptive actions keep at least one descendant goal.
A cumulative boolean OR lattice answers exact downward-existence queries.

**Finite horizon.** H=sum(initial_width−minimum) counts removable channels.
Each successful action deletes at least one and unit deletion is in the menu.
After t actions the remaining channel count is at most H−t. Therefore every
descendant width goal is reachable within the remaining horizon. A shorter
arbitrary horizon is rejected by the oracle, not approximated. Termination occurs
at first arrival in the goal set; there is no legal off-goal STOP and no cap20.
This prevents identical severe overcompression across nominal targets.

**Policy distinctions.** Static Random keeps uniform eligible widths and random
channels. Static Magnitude/SNIP retain fixed Dense incident saliency rankings and
minimize the union of removed parameter scores. Prefix sums accelerate exhaustive
evaluation but can differ in last-bit floating summation from historical scalar
loops; deterministic width ordering resolves exact ties. Sequential Random samples
reachable actions, selecting channels by current magnitude. Greedy maximizes
immediate reward; Lookahead exhausts depth2, scores r1+.9r2, executes only the first
action and replans. Stable layer/ascending deletion ordering resolves exact ties.
Depth1 matches the new constrained Greedy exactly, not historical threshold Greedy.
Concrete-parent children are reused; no aggregate-state evaluation cache exists.

**Q correction.** The state key includes namespace, target/tolerance, widths,
removed parameter count, remaining horizon, validation accuracy bucket and exact
retained original channel IDs. With a fixed Dense checkpoint these masks determine
the physical tensors; different retained masks no longer alias merely by width.
This larger state space can make coverage even sparser and does not guarantee
useful learning. Epsilon0.2 exploration is uniform over the same reachable legal
actions. Exploitation is deterministic, known Q entries or unseen0, stable first
maximum. Unseen/tied alternatives, full training transitions, Q updates and final
tables are preserved, not interpreted as counterfactual validation scores.
Thirty learning episodes reset Dense each time; deployment resets and uses
epsilon0 without updates. Dense validation can be reused across these identical
resets; every actual candidate forward is counted.

**Reward unchanged.**0.6Δcompression−0.4max(0,A_dense−A_current), plus
−2max(0,|compression−target|−.01) once at actual terminal arrival. It is a repeated
drop penalty, not incremental accuracy loss. Terminal reward is included once in
both Bellman and Lookahead returns; depth cutoffs get zero bootstrap, not a fake
terminal penalty. In-band terminals normally have zero gap penalty; closest
fallbacks can incur one. Depth2 optimizes a truncated return; Q uses discounted
remaining episodes, Greedy immediate reward. These objectives are not identical
despite shared transitions/reward. No performance superiority is predeclared.

**Comparability limits.**±1pp band permits different actual parameter budgets;
report actual compression, widths and retained identities before comparing
quality. Static and adaptive action counts have different meaning. One Dense and
one policy seed per architecture do not establish robustness. Selected Dense
checkpoints retain their original finite500epoch convergence limitations. No
test-label selection, extra seed, fine-tuning or manuscript edit occurs.

**Cost scope.** `search_seconds` begins after frozen input/checkpoint loading and
common width-goal geometry construction; it includes scoring, surgery, candidate
forwards and policy search. Three post-search reporting forwards are separate,
as is SNIP's one gradient scoring forward. Candidate preview counts and total
validation callback forwards are distinct. LSF RUN/CPU/queue histories preserve
end-to-end execution costs; search ratios are not ratios of full job costs.
Technical smoke runs share a process, so RSS is a conservative cumulative
high-water mark and caches can be warm. Research jobs run one condition/process.

**Exact execution reuse (revision2).** The uncached native resource gate failed
Q4/256, so a shared per-run validator now memoizes only identical architecture,
full model tensor SHA256 and frozen graph/split identities. It reuses scalar
validation accuracy, never Q values, rewards or a learned heuristic. All policy
requests still occur; ties, RNG draws, states, actions and updates are unchanged.
Twenty-four independent cached/uncached real-forward controller cases check exact
decision/trace/mask/tensor/Q equality at all depths and both models. Different
tensors at identical widths must miss. Requests, actual callback forwards,
per-decision cache hits and full lookup events are preserved separately. Reporting
and SNIP gradient forwards are outside this scalar validation cache. The revised
technical Q profile has three episodes; research retains all30 episodes. The first
source/preflight evidence remains immutable; this is an execution optimization,
not an accuracy-dependent scientific change or a relabeled historical algorithm.
