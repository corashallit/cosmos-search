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
//
// Before any review, one small subagent of the same workflow reports what it was handed besides its task
// (docs/vetting.md §15, second amendment). The harness gives every subagent of a workflow the last message the
// person typed in the launching thread, when that thread's context holds one, and a reviewer that has read it is
// not the reviewer the calibration measured: the audit refuses its review. If the check sees such a message the
// script stops with no review launched, so nothing is spent on reviews that cannot count. Launch from a thread
// the person has not typed in. The check is this script's guard, not the record: the audit reads every
// transcript either way. `check: false` in args skips it (for a workflow whose check already passed in this
// turn).
export const meta = {
  name: 'stage-a-reviews',
  description: 'Blind reviews: one general-purpose subagent per prompt file, one Read and one Write each, on a named model',
  phases: [{ title: 'Check' }, { title: 'Review' }],
}
const { dir, ids, model, effort, check } = args
if (!dir || !Array.isArray(ids) || !model || !effort) throw new Error('args needs dir, ids, model and effort')
const launch = id => `You are running one blind vetting task. Make exactly two tool calls and nothing else:
1. Read the file ${dir}/prompts/${id}.md (one Read call, the whole file).
2. Write your complete answer, following the instructions and the reply format given in that file, to ${dir}/replies/${id}.txt (one Write call).
Do not read, list or search any other file, do not run commands, and do not use the web. Base your answer only on the contents of that one file. After writing, reply with the single word: written.`
const CHECK = `This is a check of what a subagent of this workflow is handed. Use no tools other than the one that returns your answer.
Look at the conversation you were given, from its start up to this message. Count the messages in the user's turn that came before this one, leaving out system text and reminders the harness adds by itself. A message that begins "[Workflow harness — user request]" counts: it relays something a person wrote in the thread that started this workflow.
Return: messages_before_task (that count), user_request_before_task (true if any of them relays or contains something a person wrote, false if this task is the first thing you were given), and first_words (the first 60 characters of each such message, joined by " | "; empty if none).`
const CHECK_SCHEMA = {
  type: 'object',
  properties: {
    messages_before_task: { type: 'integer' },
    user_request_before_task: { type: 'boolean' },
    first_words: { type: 'string' },
  },
  required: ['messages_before_task', 'user_request_before_task', 'first_words'],
}
if (check !== false) {
  phase('Check')
  const seen = await agent(CHECK, { label: 'launch check', phase: 'Check', model: 'haiku', effort: 'low', schema: CHECK_SCHEMA })
  if (!seen || seen.user_request_before_task !== false || seen.messages_before_task !== 0) {
    log('Stopped before any review: a subagent of this workflow is handed more than its task (or the check returned nothing).')
    return { launched: 0, returned: 0, lost: [], stopped: 'a message from the launching thread reaches the reviewers', check: seen || null }
  }
}
phase('Review')
const out = await parallel(ids.map(id => () =>
  agent(launch(id), { label: id, phase: 'Review', agentType: 'general-purpose', model, effort })))
const lost = ids.filter((id, i) => !out[i])
if (lost.length) log(`${lost.length} of ${ids.length} reviewers returned nothing: ${lost.join(' ')}`)
return { launched: ids.length, returned: ids.length - lost.length, lost }
