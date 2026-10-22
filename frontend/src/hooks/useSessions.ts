import { useQuery } from "@tanstack/react-query";

import { sessionsApi } from "@/api/sessions";
import type { SessionSummary } from "@/types";

export const SESSIONS_KEY = (workspaceId: number) =>
  ["sessions", workspaceId] as const;

/**
 * The saved conversations, most recently used first.
 *
 * Kept in React Query rather than the conversation hook because it is server
 * state that several pieces of the screen read: the list itself, and a
 * refresh after the first question of a new chat.
 */
export function useSessions(workspaceId = 1) {
  return useQuery({
    queryKey: SESSIONS_KEY(workspaceId),
    queryFn: () =>
      sessionsApi.list(workspaceId).then((result) => result.sessions),
    // A chat cannot appear without a question, and a question is asked from
    // this screen; refetching on focus keeps a second tab's list honest
    // without polling.
    staleTime: 5_000,
  });
}

export type { SessionSummary };
