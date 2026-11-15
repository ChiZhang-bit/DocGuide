
Answer the original question concisely from the working view, including requested qualifiers,
names/dates/units. Prefer raw records over conflicting claims. If incomplete, give the most
plausible candidate with low confidence and inference/model_prior basis; never abstain or invent citations.
First identify the original interrogative slot, then complete its final comparison/count/attribute;
do not copy a premise entity or intermediate result instead of its requested property.
Keep the explicit candidate set and qualifiers of the original question. For "first/more:
X or Y", compare those choices and return the appropriate choice, not a third item from
the whole history. Use the actual two values and direction, not a previous step's conclusion.
For which items, return all supported matching items; preserve the requested full name/date.
Read rows in the question's table context: credits/roles and opponents are relative to the
table's subject, who need not occur in those cells. Contextual links remain valid evidence.
Satisfy the intersection of all original conditions; a list satisfying only one is incomplete.
Distinguish the requested attribute's dates (production, release, premiere, career start).
Return {"requested_target":"original interrogative slot, at most 8 words","answer":"short answer","used_node_ids":["IDs in the working view"],
"evidence_refs":[{"node_id":"raw table ID","column":"column used"},
{"node_id":"raw text ID","sentence_id":"sentence ID","quote":"short exact supporting span"}],
"used_alignment_ids":["alignment IDs actually used, or empty"],
"confidence":"low|medium|high","basis":"retrieved_evidence|inference|model_prior",
"evidence_sufficient":false,"reason":"at most 8 words"}.
Locate directly used raw inputs; derived inputs already carry provenance. evidence_sufficient
is an assessment, not verification. repair_feedback directs corrections, not source evidence.
