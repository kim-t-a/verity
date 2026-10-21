import {
  hasNumeralMapping,
  numeralOf,
  orderedCitations,
} from "@/lib/citations";
import type { AgentResponse, Citation } from "@/types";

function EvidenceList({
  citations,
  selected,
  onSelect,
}: {
  citations: Citation[];
  selected: number | null;
  onSelect: (numeral: number) => void;
}) {
  const numbered = hasNumeralMapping(citations);

  return (
    <ul>
      {orderedCitations(citations).map((citation, index) => {
        const numeral = numeralOf(citation);
        const isSelected = numeral !== null && numeral === selected;
        const label = numeral ?? index + 1;

        return (
          <li key={`${citation.chunk_id}-${citation.location}-${index}`}>
            <button
              type="button"
              className="evidence-row"
              aria-current={isSelected ? "true" : undefined}
              aria-label={`Evidence ${label}: ${citation.rel_path}, ${citation.location}`}
              onClick={() => onSelect(label)}
            >
              <span className="flex gap-2">
                {numbered && numeral !== null ? (
                  <span className="evidence-index" aria-hidden="true">
                    {numeral}
                  </span>
                ) : (
                  <span className="evidence-index" aria-hidden="true">
                    ·
                  </span>
                )}
                <span className="min-w-0 flex-1">
                  <span className="block break-words text-[13.5px] leading-snug text-fg">
                    {citation.rel_path}
                  </span>
                  <span className="mt-0.5 block break-words font-mono text-[11.5px] text-faint">
                    {citation.location}
                  </span>
                  {citation.text && (
                    <span className="evidence-quote block">{citation.text}</span>
                  )}
                </span>
              </span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}

function EmptyEvidence({ response }: { response: AgentResponse }) {
  const message =
    response.type === "refusal"
      ? "Nothing was cited: the corpus did not support an answer."
      : response.type === "clarification"
        ? "Nothing was cited yet. The agent is waiting for one detail before it searches."
        : "This answer cited no evidence.";
  return <p className="px-5 py-4 text-[13px] text-muted">{message}</p>;
}

/**
 * The evidence rail. On desktop it is a standing column beside the reading
 * column; on narrow screens the same list sits under the answer it belongs to,
 * collapsed until the reader asks for it.
 */
export function EvidencePanel({
  response,
  selected,
  onSelect,
  variant = "rail",
  active = false,
}: {
  response: AgentResponse;
  selected: number | null;
  onSelect: (numeral: number) => void;
  variant?: "rail" | "inline";
  active?: boolean;
}) {
  const citations = response.citations;
  const live = hasNumeralMapping(citations);

  const list = citations.length ? (
    <EvidenceList
      citations={citations}
      selected={selected}
      onSelect={onSelect}
    />
  ) : (
    <EmptyEvidence response={response} />
  );

  if (variant === "inline") {
    return (
      <details
        className="mt-3 border-t border-rule pt-2 lg:hidden"
        open={active}
      >
        <summary className="cursor-pointer list-none py-1.5 text-[12.5px] text-muted">
          {citations.length
            ? `Evidence cited (${citations.length})`
            : "Evidence"}
        </summary>
        {!live && citations.length > 0 && (
          <p className="pb-2 text-[12px] text-faint">
            Saved when this conversation was recorded. The original numbering
            was not stored, so markers in the answer are shown as plain text.
          </p>
        )}
        {/* No negative gutter here: the sheet's padding is only 20px, so
            cancelling it plus the row's own -14px would push the numeral
            outside the card's border. */}
        <div>{list}</div>
      </details>
    );
  }

  return (
    <div>
      <div className="border-b border-rule px-5 py-4">
        <h2 className="font-display text-[17px] text-fg">Evidence</h2>
        <p className="mt-0.5 font-mono text-[11.5px] text-faint">
          {citations.length
            ? `${citations.length} cited`
            : "nothing cited"}
        </p>
        {!live && citations.length > 0 && (
          <p className="mt-2 text-[12px] leading-snug text-muted">
            This conversation was reopened. The saved citations are listed, but
            their original numbering was not stored, so markers in the answer
            stay as plain text.
          </p>
        )}
      </div>
      {/* The rows bleed left on hover; this inset keeps their numerals on the
          same line as the header text. */}
      <div className="px-5">{list}</div>
    </div>
  );
}
