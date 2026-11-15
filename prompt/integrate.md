
In ONE response, align needed entities, answer all resolvable subquestions in dependency order,
and grade unused raw evidence. Use only the working view and retrieved records, including
earlier facts and answers produced here. Derive at most one concrete answer/partial fact per
subquestion; unanswered creates no fact. Include needed input IDs and any earlier result that
binds this step's subject. Do not copy unrelated fields or create redundant tasks/facts.
Align only correspondences actually used: located alias/context/mapping evidence, not equal
names/numbers alone. A mention within one record needs a citation, not self-alignment.
Compute counts/comparisons here. Distinguish grouped counts from row counts; exhaustive
claims need scope coverage. Relevance top-k and fallback rows are candidates, not proof of
complete coverage or failed code conditions. Ordinary lookups need no exhaustive uniqueness check.
Preserve every alternative and qualifier in the original question. Compare the actual values
for its named/described candidates only; first among two choices is not first in all history.
For a comparison/set answer, include BOTH/ALL contributing branches in inputs and located
refs, even when the answer names only the winner. Keep tied/multiple matching items together.
Interpret rows from the table subject's perspective; its owner need not appear in role,
co-performer or opponent cells. Do not infer nonparticipation from that absent owner name
or from a nonexhaustive cast/list. A resolved intermediate still needs the final requested hop.
Q and the actual table columns establish the question-specific relation perspective; do not
require a new performer/owner column. Apply AND conditions as an intersection/join, not just
separate lists. Keep dates and other modifiers attached to their original entity/relation.
Set can_answer_original=true ONLY when retained evidence answers the ORIGINAL question,
including all qualifiers, comparisons and requested items. Solving a current subquestion
or identifying the subject alone is not sufficient. Otherwise false, with concrete gaps in
missing for the next round. This readiness decision is not verification.

Return {"alignments":[{"id":"a1","subquestion_id":"current q ID",
"left":{"node_id":"table ID","column":"entity column"},
"right":{"node_id":"text ID","sentence_id":"p1.s0","quote":"exact entity mention"},
"basis":"explicit_alias|contextual|provided_mapping","reason":"brief identity reason",
"evidence_refs":[{"node_id":"text ID","sentence_id":"p1.s0","quote":"exact supporting span"}]}],
"answers":[{"id":"f1","subquestion_id":"current q ID","status":"answered|partial|unanswered",
"value":"answer or null if unanswered","inputs":["existing node IDs or earlier f aliases"],
"evidence_refs":[{"node_id":"table ID","column":"actual column"},
{"node_id":"text ID","sentence_id":"actual sentence ID","quote":"exact supporting span"}],
"used_alignment_ids":["a1 if needed"],"reason":"brief evidence reason","missing":"gap or empty"}],
"grades":{"unprotected raw node ID":1},"original_relevant_ids":["node IDs or f aliases"],
"can_answer_original":false,"candidate_answer":"optional concise original-question answer if now supported"}.
When can_answer_original=true, stop retrieval and let the separate Answer call compose
the original answer from the updated view. Preserve its needed evidence and facts.

Locate every directly used raw input with evidence_refs; derived inputs need only their IDs.
Quote the shortest sufficient exact span, preserving spelling/punctuation; occurrence:0 if
repeated. Keep reasons within about 8 words, missing empty when resolved. Use consistent f/a
aliases, never invent stable IDs. A partial answer needs a concrete value.
If pruning_enabled=false omit grades. Else grade unused raw relevance_candidate_ids only:
1 useful/possibly needed, 0 useless. Answer facts, recursive inputs and used alignments are
protected locally; omit their grades. Select original_relevant_ids for the original question,
including essential context/contradictions; these stay pinned across rounds.
Incomplete but relevant entity/row cues are useful; missing the final attribute is not
itself a reason to grade their source 0. Keep the evidence needed for the next targeted step.
