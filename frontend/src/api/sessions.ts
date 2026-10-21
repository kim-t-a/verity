import { api, jsonBody } from "@/api/client";
import type { AgentResponse, HistoryResponse } from "@/types";

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
};
