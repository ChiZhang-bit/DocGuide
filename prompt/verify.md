
Verify original question Q, complete task DAG H and upstream evidence path Pi in one call.
1. Planning: check Q's requested endpoint, required relations/branches and logical/execution dependencies.
2. Execution: follow Pi from raw inputs and prerequisite results to each aggregation and the
final answer; check each result answers its corresponding task from those inputs.
Graph statuses supported/answered are bookkeeping, not proof. Read the actual fields/text.
Reconstruct Q's interrogative target, explicit alternatives and qualifiers independently
of task wording or the candidate's claimed target. For "first/more: X or Y", check that
the answer compares those choices, not a whole career/history. Returning an intermediate
instead of the asked property/person, or a third item outside those choices, is a concrete
planning/composition defect. Keep correct bindings and repair only the missing/wrong hop.
For "which of X or Y", the selected entity alone is a complete answer; its comparison
values may remain in the evidence, without being repeated in the answer text. Require a
further hop only if Q actually asks an attribute/person beyond that selected entity.

Begin the JSON with a brief focus statement (at most 12 words): what property Q requests
versus what the candidate/evidence supplies. Use Q's interrogative slot, not an entity
in a qualifying premise. Then give status. This is a target comparison,
not a reasoning transcript. Example form: {"focus":"requested property; supplied property","status":"PASS"}.
Use no other fields on PASS. A named intermediate entity does not answer its requested
attribute/count; a cited mention cannot substitute for that missing final hop. Check exact
row/column, highest/lowest direction, entity identity and arithmetic/units.

Accept supported direct composition from raw evidence, contextual coreference, ordinary
word meanings and equivalent answers. Missing explicit task nodes/edges or partial status
alone are harmless when the sources support the final requested property. Do not invent
requirements or demand exhaustive uniqueness for an ordinary lookup. A successful
table_query with order_by sorts matching rows before top_k; relevance top-k alone does not.
Respect the table's perspective and the question's contextual links before challenging a
role/identity. A credits table describes its subject's work; its role/co-performer cells
need not repeat that subject. An opponent is relative to the schedule's team. An empty
filter for the owner in these columns, or omission from a partial cast/list, does not
contradict participation. Do not replace the asked entity/property with one from a qualifier.
For a set, check supported matching members and preserve ties; singular wording may name a set.
Use Q together with actual headers/rows for the question-specific table association; do not
demand the table subject's name in every row. Preserve modifier attachment and apply AND
conditions jointly. A contextual role-to-person link is not contradicted just by a missing
repeated name. Repair the earliest source-established wrong binding and its affected results,
not a valid branch merely lacking a redundant explicit relation.

Do not introduce alternative entities or requirements from background knowledge. Missing
exhaustiveness proof alone is not an error unless a provided source identifies an omission.
Use REPAIR only for a source-established, answer-changing contradiction or a clearly wrong
requested attribute, relation role or unit. An absent formal hop, unexplored alternative or
unproved exhaustiveness is not enough: PASS when raw sources compose the answer, otherwise
UNCERTAIN if support remains unresolved. Preserve clear contextual geographic/entity links.
For an otherwise supported selection, a missing final hop needs repair when Pi establishes
the selected subject but the concrete candidate substitutes it for Q's requested value.
Keep that correct selection and add its missing dependent step even if its value needs retrieval.
UNKNOWN, an unanswered task or missing support alone is not a located error; do not rename
and repeat an unresolved task as repair. Singular wording alone does not invalidate multiple
source-supported answers. Do not add location/time conditions absent from Q. A scope_complete
receipt is not incomplete merely because relevance retrievals elsewhere were top-k.
Before repairing a comparison, focus must state the compared source values and the required
direction. The proposed replacement must actually satisfy that direction within Q's scope;
if the evidence does not establish it is better, keep the candidate rather than substitute it.

For a concrete defect return
{"focus":"requested versus supplied","status":"REPAIR","reason":"specific mismatch",
"invalid_task_ids":[],"invalid_node_ids":[],"invalid_alignment_ids":[],"subquestions":[]}.
Put continuation task objects in subquestions, never an edits field. Omit empty arrays;
name existing IDs and never invalidate raw evidence. Keep valid branches;
code removes affected downstream results. For a final composition error, reason alone
reruns Answer when the required value is already in Pi. A missing requested attribute
requires a continuation with retrieval if Pi lacks its value; reason alone cannot fetch it.
Keep a correct intermediate selection and depend on its surviving task. Supply the minimal
continuation together to avoid another Router:
{"id":"fix1","question":"needed step","depends_on":[],"retrievals":[]}.
replaces_task_ids names revised tasks and invalidates old results. Use fresh local IDs;
depends_on names surviving H task IDs or new local task IDs, never F_ fact/node IDs or
rejected results. Use retrievals=[]
if evidence suffices, else {"kind":"text_retrieve|table_retrieve","query_spec":{"query":"specific search"}}.

If support is unresolved without an identifiable error, return
{"focus":"requested property; unresolved support","status":"UNCERTAIN","reason":"brief unresolved evidence"}.
Always use reason for UNCERTAIN/REPAIR; omit edits. Select the verdict from the evidence and requested target, not from
missing serialization details. Output the short focus first, then the appropriate status.
