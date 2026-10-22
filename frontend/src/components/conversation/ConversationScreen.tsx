import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { History } from "lucide-react";

import { Composer } from "@/components/conversation/Composer";
import { ConversationList } from "@/components/conversation/ConversationList";
import { EvidencePanel } from "@/components/conversation/EvidencePanel";
import { TracePanel } from "@/components/conversation/TracePanel";
import { TurnView } from "@/components/conversation/TurnView";
import { sessionsApi } from "@/api/sessions";
import { useConversation } from "@/hooks/useConversation";
import { useSessions } from "@/hooks/useSessions";
import type { Turn } from "@/types";

type AssistantTurn = Extract<Turn, { role: "assistant" }>;

function latestAssistant(turns: Turn[]): AssistantTurn | null {
  for (let index = turns.length - 1; index >= 0; index -= 1) {
    const turn = turns[index];
    if (turn.role === "assistant") return turn;
  }
  return null;
}

function assistantByKey(turns: Turn[], key: string | null): AssistantTurn | null {
  if (key === null) return null;
  const turn = turns.find((candidate) => candidate.key === key);
  if (!turn || turn.role !== "assistant") return null;
  return turn;
}

export function ConversationScreen({ workspaceId = 1 }: { workspaceId?: number }) {
  const {
    sessionId,
    turns,
    ask,
    asking,
    requestError,
    historyState,
    historyError,
    openSession,
    reset,
  } = useConversation(workspaceId);
  const chats = useSessions(workspaceId);
  const [chatsOpen, setChatsOpen] = useState(false);

  const [pinnedKey, setPinnedKey] = useState<string | null>(null);
  const [selection, setSelection] = useState<{
    key: string;
    numeral: number;
  } | null>(null);

  const latest = useMemo(() => latestAssistant(turns), [turns]);
  const activeKey = pinnedKey ?? latest?.key ?? null;
  const activeTurn = assistantByKey(turns, activeKey);
  const selected =
    selection !== null && selection.key === activeKey ? selection.numeral : null;

  const bottom = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const element = bottom.current;
    if (!element) return;
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    element.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "end" });
  }, [turns.length, asking]);

  function handleAsk(question: string) {
    setPinnedKey(null);
    setSelection(null);
    void ask(question);
  }

  function handleReset() {
    setPinnedKey(null);
    setSelection(null);
    setChatsOpen(false);
    reset();
  }

  function handleSelectSession(id: number) {
    setPinnedKey(null);
    setSelection(null);
    setChatsOpen(false);
    void openSession(id);
  }

  async function handleDeleteSession(id: number) {
    await sessionsApi.remove(id);
    await chats.refetch();
    if (id === sessionId) {
      // The chat on screen was the one deleted. Clear it rather than leave a
      // transcript pointing at a session that no longer exists.
      handleReset();
    }
  }

  function select(turnKey: string, numeral: number) {
    setPinnedKey(turnKey);
    setSelection((previous) =>
      previous && previous.key === turnKey && previous.numeral === numeral
        ? null
        : { key: turnKey, numeral }
    );
  }

  const questions = turns.filter((turn) => turn.role === "user").length;
  const openingHistory = historyState === "loading" && turns.length === 0;

  return (
    <div className="flex h-full min-h-0">
      <aside
        className="hidden w-[248px] flex-none border-r border-rule bg-surface xl:block"
        aria-label="Saved chats"
      >
        <ConversationList
          sessions={chats.data}
          loading={chats.isLoading}
          failed={chats.isError}
          activeSessionId={sessionId}
          onSelect={handleSelectSession}
          onNew={handleReset}
          onRetry={() => void chats.refetch()}
          onDelete={handleDeleteSession}
        />
      </aside>

      {chatsOpen && (
        <div
          className="fixed inset-0 z-40 xl:hidden"
          role="dialog"
          aria-modal="true"
          aria-label="Saved chats"
        >
          <button
            type="button"
            className="absolute inset-0 bg-[var(--overlay)]"
            aria-label="Close chats"
            onClick={() => setChatsOpen(false)}
          />
          <div className="absolute inset-y-0 left-0 w-[280px] border-r border-rule bg-surface shadow-xl">
            <ConversationList
              sessions={chats.data}
              loading={chats.isLoading}
              failed={chats.isError}
              activeSessionId={sessionId}
              onSelect={handleSelectSession}
              onNew={handleReset}
              onRetry={() => void chats.refetch()}
              onDelete={handleDeleteSession}
              onClose={() => setChatsOpen(false)}
            />
          </div>
        </div>
      )}

      <section className="flex min-h-0 flex-1 flex-col">
        <header className="flex flex-none items-start justify-between gap-4 border-b border-rule px-5 py-3 sm:px-8">
          <div className="mx-auto flex w-full max-w-col items-start justify-between gap-4">
            <div className="flex min-w-0 items-start gap-2">
              <button
                type="button"
                onClick={() => setChatsOpen(true)}
                aria-label="Open saved chats"
                title="Saved chats"
                className="-ml-1 mt-0.5 flex h-8 w-8 flex-none items-center justify-center rounded-[4px] text-muted transition-colors duration-quick hover:bg-surface-2 hover:text-fg xl:hidden"
              >
                <History className="h-4 w-4" aria-hidden="true" />
              </button>
              <div className="min-w-0">
                <h1 className="font-display text-[19px] leading-tight text-fg">
                  Conversation
                </h1>
                <p className="mt-0.5 text-[12px] text-muted">
                  {questions > 0
                    ? `${questions} ${questions === 1 ? "question" : "questions"} in this session`
                    : "Answers are grounded in the corpus, with the evidence shown."}
                </p>
              </div>
            </div>
            {turns.length > 0 && (
              <button
                type="button"
                className="text-link whitespace-nowrap pt-1"
                onClick={handleReset}
                disabled={asking}
              >
                New conversation
              </button>
            )}
          </div>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto">
          <div className="mx-auto w-full max-w-col px-5 py-6 sm:px-8">
            {historyError && (
              <p className="mb-5 text-[13px] text-stop" role="alert">
                Could not reopen the saved conversation. {historyError}
              </p>
            )}

            {openingHistory ? (
              <p className="font-mono text-[12.5px] text-faint">
                Opening the conversation…
              </p>
            ) : turns.length === 0 && !asking ? (
              <div className="py-10">
                <h2 className="font-display text-[26px] leading-tight text-fg">
                  Ask the corpus.
                </h2>
                <p className="mt-2 max-w-[58ch] text-[14px] leading-relaxed text-muted">
                  Verity answers only from the documents you uploaded. It
                  retrieves candidate chunks, shows them to the model, and
                  accepts a citation only when it names a chunk that was
                  actually shown. If the answer is not in the corpus, it says
                  so instead of guessing.
                </p>
                <p className="mt-4 text-[13px] text-muted">
                  No documents yet? Add them on the{" "}
                  <Link to="/documents" className="text-link">
                    Documents page
                  </Link>
                  .
                </p>
              </div>
            ) : (
              <ol className="space-y-9">
                {turns.map((turn) => (
                  <li key={turn.key}>
                    <TurnView
                      turn={turn}
                      active={turn.key === activeKey}
                      selected={turn.key === activeKey ? selected : null}
                      onSelect={(numeral) => select(turn.key, numeral)}
                    />
                  </li>
                ))}
              </ol>
            )}

            {asking && (
              <div className="fade-rise mt-9" aria-live="polite">
                <p className="flex items-baseline justify-between gap-3 text-[13px] text-muted">
                  <span>Searching the corpus, then checking what can be cited.</span>
                </p>
                <div
                  className="indeterminate mt-2"
                  role="progressbar"
                  aria-label="Waiting for the agent"
                />
              </div>
            )}

            <div ref={bottom} />
          </div>
        </div>

        <div className="flex-none border-t border-rule bg-surface">
          <div className="mx-auto w-full max-w-col px-5 py-4 sm:px-8">
            {requestError && (
              <p className="mb-3 text-[13px] text-stop" role="alert">
                {requestError}
              </p>
            )}
            <Composer onAsk={handleAsk} asking={asking} />
          </div>
        </div>
      </section>

      <aside
        className="hidden w-[360px] flex-none overflow-y-auto border-l border-rule bg-surface lg:block"
        aria-label="Evidence and run trace"
      >
        {activeTurn ? (
          <>
            <EvidencePanel
              response={activeTurn.response}
              selected={selected}
              onSelect={(numeral) => select(activeTurn.key, numeral)}
            />
            <TracePanel response={activeTurn.response} />
          </>
        ) : (
          <div className="px-5 py-6">
            <h2 className="font-display text-[17px] text-fg">Evidence</h2>
            <p className="mt-1 text-[13px] leading-snug text-muted">
              Ask a question. The chunks the agent actually used appear here,
              numbered to match the answer, with the search that found them
              underneath.
            </p>
          </div>
        )}
      </aside>
    </div>
  );
}
