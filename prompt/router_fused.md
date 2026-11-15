
Choose answer ONLY if the working evidence answers the ORIGINAL question, including its
qualifiers and final requested property/comparison/count. Identifying a premise entity or
answering one subquestion is not sufficient; plan the remaining dependent steps.
Read the whole original question before choosing steps. Keep its explicit alternatives:
"first/more among X or Y" compares X and Y, not every item in a person's history.
Resolve descriptive alternatives, obtain each candidate's comparison value, then compare;
include a final dependent step for the property/person the question actually asks for.
Treat the provided table as question-specific evidence. Infer its relation perspective from
Q plus the actual headers/rows: a performer's Role cell is a character, not a performer filter.
Q supplies reference context, not unseen factual values. Use table_query first when a literal
year/role/opponent/award condition matches a real column; resolve that row before text-searching
its linked attribute. The table subject need not appear in each row. An empty wrong-column
filter does not refute the relation. Batch table and text accesses when both are needed.
Attach dates/qualifiers to the noun or relation they modify. A reference event's year need
not be the subject's career year. AND/relative clauses require all conditions, not one branch.
An unverified search hypothesis may guide retrieval only; never count it as a bound result
or evidence. Search within the supplied table/text, then confirm from returned records.
Reuse results. Batch needed searches and reasoning; use retrievals:[] when evidence suffices.
Cover the original requested endpoint. Previews guide retrieval but are not answer evidence.
Declare depends_on whenever an earlier answer supplies a subject/value, even a literal name
in a query. Use q1,q2,... for ordered current steps and full rNqM IDs for previous steps.
Prefetch literal searches; defer only searches needing an unknown value. Never query placeholders.
After an unhelpful search, change focus or increase top_k; do not repeat it at the same k.

Return {"action":"retrieve|answer","subquestions":[{"id":"q1","question":"specific missing information",
"retrievals":[{"kind":"table_query|table_retrieve|text_retrieve|text_read","query_spec":{}}],
"depends_on":[]}],
"final_answer":{"requested_target":"original interrogative slot, at most 8 words","answer":"short answer","used_node_ids":[],"evidence_refs":[],
"used_alignment_ids":[],"confidence":"low|medium|high","basis":"retrieved_evidence|inference",
"evidence_sufficient":true,"reason":"at most 8 words"}}.
Include final_answer only for action=answer, supported by the current view, never previews.
Reread the original question: complete its last comparison/count/attribute and full requested
name; do not copy an intermediate result as the final answer.
Locate raw inputs by {node_id,column} or {node_id,sentence_id,quote}; use short exact quotes.
Derived inputs need only IDs. No duplicate candidate_answer is needed.
query_spec for table_retrieve/text_retrieve: {"query":"literal search","top_k":5};
omit top_k for the configured default.
table_query: {"filters":[{"column":"exact header","operator":"eq|ne|in|contains|gt|ge|lt|le",
"datatype":"text|number|date","value":"literal"}],"order_by":[{"column":"exact header",
"datatype":"number","direction":"asc|desc"}],"return_mode":"all_matches|top_k|exact"}.
in uses values:[...]; top_k needs top_k:N; exact needs expected_count:N. Dates require ISO dates.
all_matches covers matching rows; relevance top-k is candidate-only. Empty/error code queries
fall back to hybrid retrieval automatically. text_read: {"document_ids":["p1"]}, only IDs in
expandable_document_ids; otherwise text_retrieve first. Preserve question qualifiers.
