import { formatMs, formatScore } from "@/lib/format";
import { hasTrace, type AgentResponse, type TraceStep } from "@/types";

const ACTION_LABEL: Record<TraceStep["action"], string> = {
  search: "rewrote the query and searched again",
  answer: "answered from the shown extracts",
  clarification: "asked for one clarification",
  refusal: "refused: no extract supported an answer",
};

function TraceStepBlock({ step, index }: { step: TraceStep; index: number }) {
  const shown = step.top.slice(0, 5);

  return (
    <li className="border-t border-rule px-5 py-3 first:border-t-0">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5">
        <p className="text-[13px] text-fg">
          <span className="mr-1.5 font-mono text-[11.5px] text-faint">
            Step {index}
          </span>
          {ACTION_LABEL[step.action]}
        </p>
        <p className="font-mono text-[11px] text-faint tabular">
          {formatMs(step.retrieve_ms + step.decide_ms)}
        </p>
      </div>

      <p className="mt-1 break-words font-mono text-[11.5px] leading-snug text-muted">
        searched “{step.query}”
      </p>
      <p className="mt-0.5 font-mono text-[11px] text-faint tabular">
        {step.results} {step.results === 1 ? "chunk" : "chunks"}, retrieve{" "}
        {formatMs(step.retrieve_ms)}, decide {formatMs(step.decide_ms)}
      </p>
      {step.cited && step.cited.length > 0 && (
        <p className="mt-0.5 font-mono text-[11px] text-faint tabular">
          model cited {step.cited.length === 1 ? "extract" : "extracts"}{" "}
          {step.cited.join(", ")}
        </p>
      )}

      {shown.length > 0 && (
        <ol className="mt-2 space-y-2">
          {shown.map((chunk) => (
            <li key={`${chunk.rank}-${chunk.rel_path}-${chunk.location}`}>
              <div className="flex items-baseline justify-between gap-2">
                <span className="break-words text-[12px] text-fg">
                  {chunk.rel_path}
                </span>
                <span className="font-mono text-[11px] text-fg tabular">
                  {formatScore(chunk.score)}
                </span>
              </div>
              <p className="break-words font-mono text-[10.5px] text-faint">
                {chunk.location}
              </p>
              {chunk.sources.length > 0 && (
                <p className="font-mono text-[10.5px] text-faint">
                  {chunk.sources
                    .map(([source, rank]) => `${source} #${rank}`)
                    .join(", ")}
                </p>
              )}
              {chunk.matched_terms.length > 0 && (
                <p className="break-words font-mono text-[10.5px] text-faint">
                  matched: {chunk.matched_terms.join(", ")}
                </p>
              )}
            </li>
          ))}
        </ol>
      )}
      {step.top.length > shown.length && (
        <p className="mt-1.5 font-mono text-[10.5px] text-faint">
          {step.top.length - shown.length} more candidates not shown
        </p>
      )}
    </li>
  );
}

/**
 * What the agent actually did on one turn: the queries it searched, what each
 * retriever returned, how long each half took, and what the model cited.
 *
 * This panel never reconstructs a trace. A turn whose trace was not recorded
 * says so, because a plausible-looking diagram would be a fabricated one.
 */
export function TracePanel({ response }: { response: AgentResponse }) {
  if (!hasTrace(response)) {
    const explanation =
      response.type === "answer" && response.citations.length > 0
        ? "This answer was loaded from history. The trace of its original run was not stored."
        : "No trace was recorded for this turn. Nothing was retrieved.";

    return (
      <section className="border-t border-rule px-5 py-4">
        <h2 className="font-display text-[17px] text-fg">Run trace</h2>
        <p className="mt-1 text-[12.5px] leading-snug text-muted">
          {explanation}
        </p>
      </section>
    );
  }

  const { trace } = response;

  return (
    <details className="border-t border-rule" open>
      <summary className="cursor-pointer list-none px-5 py-4">
        <span className="flex items-baseline justify-between gap-3">
          <span className="font-display text-[17px] text-fg">Run trace</span>
          <span className="font-mono text-[11px] text-faint tabular">
            {trace.steps.length} {trace.steps.length === 1 ? "step" : "steps"},{" "}
            {formatMs(trace.duration_ms)}
          </span>
        </span>
        <span className="mt-0.5 block font-mono text-[11.5px] text-muted">
          {trace.retrievers.join(" + ") || "no retriever ran"}
        </span>
      </summary>

      {trace.degraded && (
        <p className="mx-5 mb-3 border border-accent-rule bg-accent-wash px-3 py-2 text-[12px] leading-snug text-fg">
          Vector search did not run, so these results and scores are lexical
          only. The answer is still verified against the chunks shown.
        </p>
      )}

      <ol>
        {trace.steps.map((step, index) => (
          <TraceStepBlock
            key={`${step.index}-${index}`}
            step={step}
            index={index + 1}
          />
        ))}
      </ol>
    </details>
  );
}
