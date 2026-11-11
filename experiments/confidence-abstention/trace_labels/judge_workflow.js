// Judge pipeline for trace labels, run with Claude Code's Workflow tool (it spawns the judge subagents):
//   args = {dir: <abs path to a dir of batch_NNN.jsonl from build_batches.py>, tl: <abs path to trace_labels/>,
//           batches: [["000", [ids...]], ...], note: <optional extra instruction for every judge>}
// Then: python trace_labels/write_labels.py <workflow result json> <dir>  -> labels_{a,b,tiebreak}_000.jsonl
export const meta = {
  name: 'trace-label-judges',
  description: 'Label reasoning traces: two Sonnet judges (rubric_a / rubric_b) per batch, single-trace fallback for dropped ids, per-item tiebreak',
  phases: [{ title: 'Judge' }, { title: 'Fallback' }, { title: 'Tiebreak' }],
}

const TL = args.tl
const file = b => `${args.dir}/batch_${b}.jsonl`
const KINDS = ['RESOLVE', 'JUDGE', 'MIXED', 'OTHER']
const REASONS = ['not_listed', 'question_wrong', 'uncertain_recall', 'tie_between_options', 'instruction_license', 'confident_pick', 'other']
const LABEL = {
  type: 'object',
  properties: {
    id: { type: 'string' }, kind: { type: 'string', enum: KINDS }, candidate: { type: 'string' },
    candidate_in_options: { type: 'integer', minimum: 0, maximum: 4 },
    decision_reason: { type: 'string', enum: REASONS }, rationale: { type: 'string' },
  },
  required: ['id', 'kind', 'candidate', 'candidate_in_options', 'decision_reason', 'rationale'],
}
const LABELS = { type: 'object', properties: { labels: { type: 'array', items: LABEL } }, required: ['labels'] }
const FIELDS = 'id, phase (2 or "verbal"), question, options (1-4), previous_answer (verbal only), final_answer (phase 2 only), reasoning (the trace)'
const showAll = f => `python3 -I -c "import json,sys; r=[json.loads(l) for l in open(sys.argv[1])][int(sys.argv[2])]; print(json.dumps({k:v for k,v in r.items() if k!='reasoning'})); print(r['reasoning'])" "${f}" <i>`
const showOne = (f, id) => `python3 -I -c "import json,sys; r=[x for x in map(json.loads, open(sys.argv[1])) if x['id']==sys.argv[2]][0]; print(json.dumps({k:v for k,v in r.items() if k!='reasoning'})); print(r['reasoning'])" "${f}" ${id}`
const NOTE = args.note ? `\n${args.note}` : ''
const RULES = 'Read the ENTIRE trace yourself and judge it against the rubric. Do NOT classify with keyword scripts, regexes or heuristics. Do not write or modify any files.'

const batchPrompt = (b, rubric, ids) => `You are labelling reasoning traces with a fixed rubric. Read the rubric at ${TL}/${rubric} first.${NOTE}
The batch is ${file(b)} (${ids.length} lines; ids ${ids.join(', ')}); fields: ${FIELDS}. Process items one at a time, printing each with:
  ${showAll(file(b))}
${RULES} Judge each trace on its own. Return exactly one label per id listed above via the structured output.`
const onePrompt = (b, id, rubric) => `You are labelling one reasoning trace with a fixed rubric. Read the rubric at ${TL}/${rubric} first.${NOTE}
Print the item (fields: ${FIELDS}) with:
  ${showOne(file(b), id)}
${RULES} Return your label (id ${id}) via the structured output.`
const tiePrompt = (b, id, a, c) => `Two judges disagreed on the \`kind\` of one reasoning trace. Read the rubric at ${TL}/rubric_a.md (judge_2 used an equivalent rewording, ${TL}/rubric_b.md).${NOTE}
Print the item with:
  ${showOne(file(b), id)}
judge_1: ${JSON.stringify(a)}
judge_2: ${JSON.stringify(c)}
Read the full trace yourself, weigh both rationales against it, and decide; you may agree with either or choose differently. No keyword scripts or heuristics. Do not write any files. Return your label (id ${id}) via the structured output.`

const judgeBatch = async (b, ids, j) => {
  const rubric = j === 'a' ? 'rubric_a.md' : 'rubric_b.md'
  const r = await agent(batchPrompt(b, rubric, ids), { label: `${j} ${b}`, phase: 'Judge', schema: LABELS, model: 'sonnet' })
  const got = {}
  for (const l of (r ? r.labels : [])) if (ids.includes(l.id) && !got[l.id]) got[l.id] = l
  const miss = ids.filter(id => !got[id])
  if (miss.length) log(`${j} ${b}: ${miss.length} ids via single-trace fallback`)
  const fb = await parallel(miss.map(id => () =>
    agent(onePrompt(b, id, rubric), { label: `${j} ${b} ${id}`, phase: 'Fallback', schema: LABEL, model: 'sonnet' })))
  fb.filter(Boolean).forEach(l => { if (ids.includes(l.id)) got[l.id] = l })
  return got
}

const out = await pipeline(
  args.batches,
  ([b, ids]) => parallel(['a', 'b'].map(j => () => judgeBatch(b, ids, j))),
  async ([ga, gb], [b, ids]) => {
    const dis = ids.filter(id => ga[id] && gb[id] && ga[id].kind !== gb[id].kind)
    const tie = (await parallel(dis.map(id => () =>
      agent(tiePrompt(b, id, ga[id], gb[id]), { label: `tie ${b} ${id}`, phase: 'Tiebreak', schema: LABEL, model: 'sonnet' })))).filter(Boolean)
    const missing = ids.filter(id => !ga[id] || !gb[id] || (dis.includes(id) && !tie.find(t => t.id === id)))
    if (missing.length) log(`${b}: still missing ${missing.join(', ')}`)
    return { b, a: Object.values(ga), bb: Object.values(gb), tie, missing }
  },
)
return out