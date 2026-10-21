import { useState } from "react";

import { StatusMark } from "@/components/documents/StatusMark";
import { useDeleteDocument } from "@/hooks/useDocuments";
import { formatCount, formatFileSize, formatTimestamp } from "@/lib/format";
import type { DocumentRecord } from "@/types";

const COLUMNS =
  "sm:grid-cols-[minmax(0,1fr)_110px_92px_124px_72px] sm:gap-4";

function chunkLabel(document: DocumentRecord): string {
  if (document.status === "ready") {
    return formatCount(document.chunk_count, "chunk");
  }
  return "—";
}

export function DocumentTable({
  documents,
  workspaceId = 1,
}: {
  documents: DocumentRecord[];
  workspaceId?: number;
}) {
  const remove = useDeleteDocument(workspaceId);
  const [deletingId, setDeletingId] = useState<number | null>(null);

  async function handleDelete(document: DocumentRecord) {
    const confirmed = window.confirm(
      `Delete "${document.filename}"?\n\n` +
        "Its chunks leave the index immediately, so no future answer can cite it."
    );
    if (!confirmed) return;

    setDeletingId(document.id);
    try {
      await remove.mutateAsync(document.id);
    } catch {
      // The mutation's error is rendered above the list. Keeping the row is
      // the correct outcome: the delete did not happen.
    } finally {
      setDeletingId(null);
    }
  }

  const deleteError =
    remove.error instanceof Error ? remove.error.message : null;

  return (
    <div>
      {deleteError && (
        <p className="mb-3 text-[13px] text-stop" role="alert">
          Could not delete that document. {deleteError}
        </p>
      )}

      <div
        className={`hidden border-b border-rule px-1 pb-2 text-[12px] text-faint sm:grid ${COLUMNS}`}
        aria-hidden="true"
      >
        <span>File</span>
        <span>Status</span>
        <span>Chunks</span>
        <span>Added</span>
        <span />
      </div>

      <ul>
        {documents.map((document) => {
          const working =
            document.status === "pending" || document.status === "processing";

          return (
            <li
              key={document.id}
              className="border-b border-rule last:border-b-0"
            >
              <div
                className={`grid grid-cols-1 gap-2 px-1 py-3.5 sm:items-center ${COLUMNS}`}
              >
                <div className="min-w-0">
                  <p
                    className="truncate text-[14.5px] text-fg"
                    title={document.filename}
                  >
                    {document.filename}
                  </p>
                  <p className="mt-0.5 font-mono text-[11.5px] text-faint tabular sm:hidden">
                    {chunkLabel(document)}, {formatFileSize(document.size_bytes)},{" "}
                    {formatTimestamp(document.created_at)}
                  </p>
                  {document.error && (
                    <p className="mt-1 font-mono text-[12px] text-stop">
                      {document.error}
                    </p>
                  )}
                </div>

                <StatusMark status={document.status} />

                <span className="hidden font-mono text-[12.5px] text-muted tabular sm:block">
                  {chunkLabel(document)}
                </span>
                <span className="hidden font-mono text-[12.5px] text-muted tabular sm:block">
                  {formatTimestamp(document.created_at)}
                </span>

                <div className="sm:text-right">
                  <button
                    type="button"
                    className="text-link"
                    onClick={() => void handleDelete(document)}
                    disabled={deletingId === document.id}
                  >
                    {deletingId === document.id ? "Deleting…" : "Delete"}
                  </button>
                </div>
              </div>

              {working && (
                <div
                  className="indeterminate"
                  role="progressbar"
                  aria-label={`Indexing ${document.filename}`}
                />
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
