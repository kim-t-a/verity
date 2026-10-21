import { api } from "@/api/client";
import type { DocumentRecord, UploadResult } from "@/types";

export const documentsApi = {
  list(workspaceId = 1): Promise<{ documents: DocumentRecord[] }> {
    return api(`/api/workspaces/${workspaceId}/documents`);
  },

  upload(file: File, workspaceId = 1): Promise<UploadResult> {
    const form = new FormData();
    form.append("file", file);
    // No Content-Type: the browser adds multipart/form-data with its boundary.
    return api(`/api/workspaces/${workspaceId}/documents`, {
      method: "POST",
      body: form,
    });
  },

  remove(documentId: number, workspaceId = 1): Promise<void> {
    return api(`/api/workspaces/${workspaceId}/documents/${documentId}`, {
      method: "DELETE",
    });
  },
};
