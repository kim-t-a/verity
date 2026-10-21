import { useEffect, useRef, useState } from "react";

import { EvidencePanel } from "@/components/conversation/EvidencePanel";
import { citationAt, knownNumerals, paragraphs, parseParagraph } from "@/lib/citations";
import type { AgentResponse, Turn } from "@/types";

function AnswerBody({
  response,
  active,
  selected,
  onSelect,
}: {
  response: AgentResponse;
  active: boolean;
  selected: number | null;
  onSelect: (numeral: number) => void;
}) {
  const known = knownNumerals(response.citations);
  const blocks = paragraphs(response.text);

  if (blocks.length === 0) {
    return <p className="answer-body whitespace-pre-wrap">{response.text}</p>;
  }

  return (
    <div className="answer-body">
      {blocks.map((block, blockIndex) => (
        <p key={blockIndex} className="whitespace-pre-wrap">
          {parseParagraph(block, known).map((segment, index) =>
            segment.kind === "text" ? (
              <span key={index}>{segment.text}</span>
            ) : (
              <button
                key={index}
                type="button"
                className="citation-num"
                aria-pressed={active && selected === segment.numeral}
                aria-label={(() => {
                  const citation = citationAt(
                    response.citations,
                    segment.numeral
                  );
                  return citation
                    ? `Evidence ${segment.numeral}, ${citation.rel_path}, ${citation.location}`
                    : `Evidence ${segment.numeral}`;
                })()}
                onClick={() => onSelect(segment.numeral)}
              >
                {segment.numeral}
              </button>
            )
          )}
        </p>
      ))}
    </div>
  );
}

function SearchedList({ response }: { response: AgentResponse }) {
  if (response.type !== "refusal" || response.searched.length === 0) return null;

  return (
    <div className="mt-4 border-t border-rule pt-3">
      <p className="text-[12.5px] text-faint">
        It searched these passages and found no support:
      </p>
      <ul className="mt-1.5 space-y-1.5">
        {response.searched.map((item, index) => (
          <li key={`${item.rel_path}-${item.location}-${index}`}>
            <span className="block break-words font-mono text-[12px] text-fg">
              {item.rel_path}
            </span>
            <span className="block break-words font-mono text-[11px] text-faint">
              {item.location}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

type CopyState = "idle" | "copied" | "failed";

function AssistantTurn({
  response,
  active,
  selected,
  onSelect,
}: {
  response: AgentResponse;
  active: boolean;
  selected: number | null;
  onSelect: (numeral: number) => void;
}) {
  const [copyState, setCopyState] = useState<CopyState>("idle");
  const timer = useRef<number | null>(null);

  useEffect(
    () => () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
    },
    []
  );

  async function copy() {
    try {
      await navigator.clipboard.writeText(response.text);
      setCopyState("copied");
    } catch {
      setCopyState("failed");
    }
    if (timer.current !== null) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setCopyState("idle"), 2000);
  }

  const steps = response.trace?.steps.length ?? null;
  const sources = response.citations.length;

  return (
    <div className="sheet p-5 sm:p-6">
      {response.type === "clarification" && (
        <p className="mb-2 text-[12.5px] text-muted">
          The agent needs one detail before it can answer.
        </p>
      )}
      {response.type === "refusal" && (
        <p className="mb-2 text-[12.5px] text-muted">
          The agent could not support an answer from the corpus.
        </p>
      )}

      <AnswerBody
        response={response}
        active={active}
        selected={selected}
        onSelect={onSelect}
      />

      <SearchedList response={response} />

      {response.type === "clarification" && (
        <p className="mt-3 text-[12.5px] text-muted">
          Answer in the composer. Your next message is searched fresh, against
          the corpus as it stands.
        </p>
      )}

      <footer className="mt-4 flex flex-wrap items-center justify-between gap-x-4 gap-y-2 border-t border-rule pt-2.5">
        <p className="font-mono text-[11px] text-faint tabular">
          {steps !== null && (
            <span>
              {steps} {steps === 1 ? "pass" : "passes"}
            </span>
          )}
          {steps !== null && sources > 0 && <span>, </span>}
          {sources > 0 && (
            <span>
              {sources} {sources === 1 ? "source" : "sources"}
            </span>
          )}
          {steps === null && sources === 0 && <span>no evidence cited</span>}
        </p>
        <button
          type="button"
          className="text-link"
          onClick={() => void copy()}
        >
          {copyState === "copied"
            ? "Copied"
            : copyState === "failed"
              ? "Copy failed"
              : "Copy answer"}
        </button>
      </footer>

      <EvidencePanel
        variant="inline"
        response={response}
        selected={active ? selected : null}
        onSelect={onSelect}
        active={active}
      />
    </div>
  );
}

/**
 * One exchange. The question is plain ink above the answer; the answer is the
 * only thing on the screen with a border and a shadow, because it is the artifact
 * the reader is here to inspect.
 */
export function TurnView({
  turn,
  active,
  selected,
  onSelect,
}: {
  turn: Turn;
  active: boolean;
  selected: number | null;
  onSelect: (numeral: number) => void;
}) {
  if (turn.role === "user") {
    return (
      <article className="fade-rise">
        <p className="font-display text-[17px] leading-snug text-fg whitespace-pre-wrap">
          {turn.text}
        </p>
        <div className="mt-3 h-px bg-rule" aria-hidden="true" />
      </article>
    );
  }

  return (
    <article className="fade-rise">
      <AssistantTurn
        response={turn.response}
        active={active}
        selected={selected}
        onSelect={onSelect}
      />
    </article>
  );
}
