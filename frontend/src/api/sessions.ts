import { api, jsonBody } from "@/api/client";
import type { AgentResponse, HistoryResponse, SessionListResponse } from "@/types";

export const sessionsApi = {
  create(workspaceId = 1): Promise<{ session_id: number }> {
    return api(
      "/api/sessions",
      jsonBody("POST", { workspace_id: workspaceId })
    );
  },

  ask(sessionId: number, question: string): Promise<AgentResponse> {
    return api(
      `/api/sessions/${sessionId}/messages`,
      jsonBody("POST", { question })
    );
  },

  history(sessionId: number): Promise<HistoryResponse> {
    return api(`/api/sessions/${sessionId}/messages`);
  },

  list(workspaceId = 1, limit = 50): Promise<SessionListResponse> {
    return api(`/api/sessions?workspace_id=${workspaceId}&limit=${limit}`);
  },
};
