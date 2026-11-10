// Blind reviews, one subagent per prompt file (docs/vetting.md §15). Run with the Workflow tool:
//
//   Workflow({scriptPath: "scripts/stage_a_reviews.js",
//             args: {dir: "<absolute batch dir>", ids: ["<packet id>", ...], model: "<model id>", effort: "<level>"}})
//
// dir holds prompts/<id>.md and an existing, empty replies/ directory. Every reviewer is a general-purpose
// subagent on the named model and reasoning effort, whatever model the launching thread runs on, and is told to
// make one Read of its prompt and one Write of its reply. A workflow runs only a few subagents at a time (two on a
// 4-CPU container), so split the ids over several workflows launched together; `python -m cosmos_search.vetting
// pending` prints the ids that still lack a reply, in chunks. Afterwards scripts/stage_a_audit.py checks every
// transcript: the two tool calls, and that the named model served every turn.
//
// The launch text below is the one every calibration used; changing it, the agent type, the model or the effort
// makes a new reviewer, whose recall and false-shortlist rate have to be measured again (docs/vetting.md §7).
export const meta = {
  name: 'stage-a-reviews',
  description: 'Blind reviews: one general-purpose subagent per prompt file, one Read and one Write each, on a named model',
  phases: [{ title: 'Review' }],
}
const { dir, ids, model, effort } = args
if (!dir || !Array.isArray(ids) || !model || !effort) throw new Error('args needs dir, ids, model and effort')
const launch = id => `You are running one blind vetting task. Make exactly two tool calls and nothing else:
1. Read the file ${dir}/prompts/${id}.md (one Read call, the whole file).
2. Write your complete answer, following the instructions and the reply format given in that file, to ${dir}/replies/${id}.txt (one Write call).
Do not read, list or search any other file, do not run commands, and do not use the web. Base your answer only on the contents of that one file. After writing, reply with the single word: written.`
phase('Review')
const out = await parallel(ids.map(id => () =>
  agent(launch(id), { label: id, phase: 'Review', agentType: 'general-purpose', model, effort })))
const lost = ids.filter((id, i) => !out[i])
if (lost.length) log(`${lost.length} of ${ids.length} reviewers returned nothing: ${lost.join(' ')}`)
return { launched: ids.length, returned: ids.length - lost.length, lost }
