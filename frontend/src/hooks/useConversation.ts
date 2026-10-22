import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError } from "@/api/client";
import { sessionsApi } from "@/api/sessions";
import { SESSIONS_KEY } from "@/hooks/useSessions";
import type { AgentResponse, HistoryMessage, Turn } from "@/types";

const storageKey = (workspaceId: number) => `verity.session.${workspaceId}`;

function readStoredSession(workspaceId: number): number | null {
  try {
    const raw = window.localStorage.getItem(storageKey(workspaceId));
    if (!raw) return null;
    const parsed = Number.parseInt(raw, 10);
    return Number.isFinite(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function storeSession(workspaceId: number, sessionId: number): void {
  try {
    window.localStorage.setItem(storageKey(workspaceId), String(sessionId));
  } catch {
    // A private window may refuse storage. The conversation still works for
    // this page load; only reopening it later is lost.
  }
}

function forgetSession(workspaceId: number): void {
  try {
    window.localStorage.removeItem(storageKey(workspaceId));
  } catch {
    // As above.
  }
}

function newKey(prefix: string): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `${prefix}-${crypto.randomUUID()}`;
  }
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function describeError(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof TypeError) {
    return "Cannot reach the API. Is the backend running?";
  }
  return error instanceof Error ? error.message : "The request failed.";
}

function historyToTurn(message: HistoryMessage): Turn {
  if (message.role === "user") {
    return {
      key: `history-${message.id}`,
      role: "user",
      text: message.content,
      at: message.created_at || null,
    };
  }

  const response: AgentResponse = {
    type: message.kind,
    text: message.content,
    citations: message.citations ?? [],
    searched: [],
    iterations: 0,
  };
  return {
    key: `history-${message.id}`,
    role: "assistant",
    response,
    at: message.created_at || null,
  };
}

export type HistoryState = "loading" | "ready" | "error";

/**
 * One conversation, in the browser's own state.
 *
 * Server state belongs to React Query (documents); this is client state with a
 * side effect — a session id that has to survive a refresh. The session is
 * created lazily on the first question, so opening the screen does not leave
 * empty sessions in the database.
 */
export function useConversation(workspaceId = 1) {
  const queryClient = useQueryClient();
  const [sessionId, setSessionId] = useState<number | null>(() =>
    readStoredSession(workspaceId)
  );
  const [turns, setTurns] = useState<Turn[]>([]);
  const [historyState, setHistoryState] = useState<HistoryState>("loading");
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [requestError, setRequestError] = useState<string | null>(null);
  const [asking, setAsking] = useState(false);
  //: Which open is the current one. Clicking two chats quickly must end on
  //: the second, not on whichever history response happened to arrive last.
  const opening = useRef(0);

  // Hydrate once, from the id this tab remembered. Creating a session during
  // `ask` must not re-run this: it would replace the turn already on screen.
  useEffect(() => {
    const stored = readStoredSession(workspaceId);
    if (stored === null) {
      setHistoryState("ready");
      return;
    }

    let cancelled = false;
    const request = ++opening.current;
    setHistoryState("loading");
    sessionsApi
      .history(stored)
      .then((data) => {
        if (cancelled || opening.current !== request) return;
        setSessionId(stored);
        // Do not overwrite a turn the person has already sent while history
        // was in flight.
        setTurns((previous) =>
          previous.length ? previous : data.messages.map(historyToTurn)
        );
        setHistoryState("ready");
      })
      .catch((error: unknown) => {
        if (cancelled || opening.current !== request) return;
        if (error instanceof ApiError && error.status === 404) {
          // The database was reset or the session deleted. Start clean rather
          // than leaving the screen pointed at a session that does not exist.
          forgetSession(workspaceId);
          setSessionId(null);
          setTurns([]);
          setHistoryState("ready");
          return;
        }
        setHistoryError(describeError(error));
        setHistoryState("error");
      });

    return () => {
      cancelled = true;
    };
  }, [workspaceId]);

  const ask = useCallback(
    async (question: string) => {
      const text = question.trim();
      if (!text || asking) return;

      setRequestError(null);
      setAsking(true);
      setTurns((previous) => [
        ...previous,
        { key: newKey("user"), role: "user", text, at: new Date().toISOString() },
      ]);

      try {
        let activeSession = sessionId;
        if (activeSession === null) {
          const created = await sessionsApi.create(workspaceId);
          activeSession = created.session_id;
          setSessionId(activeSession);
          storeSession(workspaceId, activeSession);
        }

        const response = await sessionsApi.ask(activeSession, text);
        setTurns((previous) => [
          ...previous,
          {
            key: newKey("assistant"),
            role: "assistant",
            response,
            at: new Date().toISOString(),
          },
        ]);
        // The first question turns an empty session into a chat with a title,
        // and every question moves it to the top of the list.
        void queryClient.invalidateQueries({
          queryKey: SESSIONS_KEY(workspaceId),
        });
      } catch (error) {
        // The question stays on screen: it was asked, and hiding it would
        // misrepresent what the corpus has seen.
        setRequestError(describeError(error));
      } finally {
        setAsking(false);
      }
    },
    [asking, queryClient, sessionId, workspaceId]
  );

  const openSession = useCallback(
    async (id: number) => {
      if (id === sessionId) return;

      const request = ++opening.current;
      setRequestError(null);
      setHistoryError(null);
      setHistoryState("loading");
      setTurns([]);

      try {
        const data = await sessionsApi.history(id);
        if (opening.current !== request) return;
        storeSession(workspaceId, id);
        setSessionId(id);
        setTurns(data.messages.map(historyToTurn));
        setHistoryState("ready");
      } catch (error) {
        if (opening.current !== request) return;
        if (error instanceof ApiError && error.status === 404) {
          // Deleted or reset since the list was drawn. Drop the row's
          // selection rather than pointing the screen at nothing.
          forgetSession(workspaceId);
          setSessionId(null);
          setTurns([]);
          setHistoryState("ready");
          void queryClient.invalidateQueries({
            queryKey: SESSIONS_KEY(workspaceId),
          });
          return;
        }
        setHistoryError(describeError(error));
        setHistoryState("error");
      }
    },
    [queryClient, sessionId, workspaceId]
  );

  const reset = useCallback(() => {
    // An open still in flight belongs to the chat being left behind.
    opening.current += 1;
    forgetSession(workspaceId);
    setSessionId(null);
    setTurns([]);
    setRequestError(null);
    setHistoryError(null);
    setHistoryState("ready");
  }, [workspaceId]);

  return {
    sessionId,
    turns,
    ask,
    asking,
    requestError,
    historyState,
    historyError,
    openSession,
    reset,
  };
}
