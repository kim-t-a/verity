import { useRef, useState } from "react";

import { Button } from "@/components/ui/Button";

const MAX_HEIGHT = 160;

/**
 * The composer is deliberately plain: a question, a send. The agent's work
 * happens on the server and is shown in the trace, not simulated here.
 */
export function Composer({
  onAsk,
  asking,
}: {
  onAsk: (question: string) => void;
  asking: boolean;
}) {
  const [value, setValue] = useState("");
  const input = useRef<HTMLTextAreaElement>(null);
  const canSend = value.trim().length > 0 && !asking;

  function resize(element: HTMLTextAreaElement) {
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, MAX_HEIGHT)}px`;
  }

  function submit() {
    const question = value.trim();
    if (!question || asking) return;
    setValue("");
    if (input.current) {
      input.current.style.height = "auto";
      input.current.focus();
    }
    onAsk(question);
  }

  return (
    <div>
      <label htmlFor="question" className="sr-only">
        Ask a question about the documents
      </label>
      <div className="flex items-end gap-2">
        <textarea
          id="question"
          ref={input}
          className="field min-h-[46px] max-h-40 overflow-y-auto"
          rows={1}
          value={value}
          placeholder="Ask about the documents…"
          onChange={(event) => {
            setValue(event.target.value);
            resize(event.currentTarget);
          }}
          onKeyDown={(event) => {
            if (
              event.key === "Enter" &&
              !event.shiftKey &&
              !event.nativeEvent.isComposing
            ) {
              event.preventDefault();
              submit();
            }
          }}
          disabled={asking}
        />
        <Button onClick={submit} disabled={!canSend} className="h-[46px] flex-none">
          {asking ? "Asking…" : "Ask"}
        </Button>
      </div>

      <p className="mt-2 text-[11.5px] text-faint" aria-live="polite">
        {asking
          ? "Searching the corpus and checking citations…"
          : "Enter asks. Shift+Enter adds a line."}
      </p>
    </div>
  );
}
