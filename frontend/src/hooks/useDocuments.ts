import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { documentsApi } from "@/api/documents";
import type { DocumentRecord } from "@/types";

const DOCUMENTS_KEY = (workspaceId: number) => ["documents", workspaceId] as const;

function isWorking(documents: DocumentRecord[] | undefined): boolean {
  return Boolean(
    documents?.some(
      (document) =>
        document.status === "pending" || document.status === "processing"
    )
  );
}

/**
 * The corpus, with polling that lives exactly as long as it is needed.
 *
 * `pending` counts: a freshly uploaded row is written as pending and only
 * becomes processing once the background task starts. Polling only on
 * "processing" is why an upload used to sit at "Pending" forever.
 */
export function useDocuments(workspaceId = 1) {
  return useQuery({
    queryKey: DOCUMENTS_KEY(workspaceId),
    queryFn: () => documentsApi.list(workspaceId).then((result) => result.documents),
    refetchInterval: (query) => (isWorking(query.state.data) ? 1500 : false),
  });
}

export function useUploadDocument(workspaceId = 1) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (file: File) => documentsApi.upload(file, workspaceId),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: DOCUMENTS_KEY(workspaceId) });
    },
  });
}

export function useDeleteDocument(workspaceId = 1) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (documentId: number) => documentsApi.remove(documentId, workspaceId),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: DOCUMENTS_KEY(workspaceId) });
    },
  });
}
