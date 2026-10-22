import { Plus, X } from "lucide-react";

import { formatTimestamp } from "@/lib/format";
import { cn } from "@/lib/cn";
import type { SessionSummary } from "@/types";

interface ConversationListProps {
  sessions: SessionSummary[] | undefined;
  loading: boolean;
  failed: boolean;
  activeSessionId: number | null;
  onSelect: (sessionId: number) => void;
  onNew: () => void;
  onRetry: () => void;
  /** Present only in the narrow-screen drawer, which needs a close control. */
  onClose?: () => void;
}

/**
 * The saved conversations, newest activity first.
 *
 * Titles are the first question the person asked. Nothing here is generated:
 * a chat list that renamed a conversation would be the product's own summary
 * of it, and this product does not summarise without being asked.
 */
export function ConversationList({
  sessions,
  loading,
  failed,
  activeSessionId,
  onSelect,
  onNew,
  onRetry,
  onClose,
}: ConversationListProps) {
  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex flex-none items-center justify-between gap-2 px-3 py-3">
        <h2 className="px-1 font-display text-[15px] text-fg">Chats</h2>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={onNew}
            aria-label="New chat"
            title="New chat"
            className="flex h-7 w-7 items-center justify-center rounded-[4px] text-muted transition-colors duration-quick hover:bg-surface-2 hover:text-fg"
          >
            <Plus className="h-4 w-4" aria-hidden="true" />
          </button>
          {onClose && (
            <button
              type="button"
              onClick={onClose}
              aria-label="Close chats"
              className="flex h-7 w-7 items-center justify-center rounded-[4px] text-muted transition-colors duration-quick hover:bg-surface-2 hover:text-fg"
            >
              <X className="h-4 w-4" aria-hidden="true" />
            </button>
          )}
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
        {failed ? (
          <div className="px-1 py-2">
            <p className="text-[12.5px] text-muted">
              Could not load saved chats.
            </p>
            <button
              type="button"
              onClick={onRetry}
              className="mt-1 text-[12.5px] text-link"
            >
              Try again
            </button>
          </div>
        ) : loading ? (
          <p className="px-1 py-2 font-mono text-[11.5px] text-faint">
            Loading chats…
          </p>
        ) : !sessions || sessions.length === 0 ? (
          <p className="px-1 py-2 text-[12.5px] leading-snug text-faint">
            No saved chats yet. Your first question starts one.
          </p>
        ) : (
          <ul className="space-y-0.5">
            {sessions.map((session) => {
              const active = session.session_id === activeSessionId;
              return (
                <li key={session.session_id}>
                  <button
                    type="button"
                    onClick={() => onSelect(session.session_id)}
                    aria-current={active ? "true" : undefined}
                    className={cn(
                      "block w-full rounded-[4px] px-2.5 py-2 text-left transition-colors duration-quick",
                      active
                        ? "bg-accent-wash"
                        : "hover:bg-surface-2"
                    )}
                  >
                    <span
                      className={cn(
                        "line-clamp-2 block text-[13px] leading-snug",
                        active ? "text-fg" : "text-muted"
                      )}
                    >
                      {session.title}
                    </span>
                    <span className="mt-0.5 block font-mono text-[10.5px] text-faint tabular">
                      {formatTimestamp(session.last_message_at)}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}
