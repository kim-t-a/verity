import type { DocumentStatus } from "@/types";

const STATUS: Record<
  DocumentStatus,
  { label: string; color: string; text: string }
> = {
  ready: { label: "Ready", color: "var(--ok)", text: "var(--fg)" },
  processing: { label: "Processing", color: "var(--accent)", text: "var(--fg)" },
  pending: { label: "Queued", color: "var(--fg-faint)", text: "var(--fg-muted)" },
  failed: { label: "Failed", color: "var(--stop)", text: "var(--stop)" },
};

/**
 * A word plus a small filled square. Not a pill: a status is a fact about the
 * row, not a badge competing with the filename for attention.
 */
export function StatusMark({ status }: { status: DocumentStatus }) {
  const state = STATUS[status];
  return (
    <span
      className="inline-flex items-center gap-1.5 text-[13px] whitespace-nowrap"
      style={{ color: state.text }}
    >
      <span
        className="status-square"
        style={{ background: state.color }}
        aria-hidden="true"
      />
      {state.label}
    </span>
  );
}
