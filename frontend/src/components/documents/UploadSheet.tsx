import { FileText, Upload } from "lucide-react";
import { useRef, useState } from "react";

import { Button } from "@/components/ui/Button";
import { useUploadDocument } from "@/hooks/useDocuments";
import { formatFileSize } from "@/lib/format";
import type { UploadResult } from "@/types";

const ACCEPT =
  ".pdf,.docx,.xlsx,.csv,.txt,.log,.md,.py,.js,.ts,.json,.png,.jpg,.jpeg,.zip";

export function UploadSheet({ workspaceId = 1 }: { workspaceId?: number }) {
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [accepted, setAccepted] = useState<UploadResult | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const upload = useUploadDocument(workspaceId);

  function choose(next: File | null) {
    setFile(next);
    setAccepted(null);
    upload.reset();
  }

  function pick(next: File | undefined) {
    if (next) choose(next);
  }

  function handleUpload() {
    if (!file) return;
    upload.mutate(file, {
      onSuccess: (result) => {
        setAccepted(result);
        setFile(null);
        if (inputRef.current) inputRef.current.value = "";
      },
    });
  }

  const error = upload.error instanceof Error ? upload.error.message : null;

  return (
    <section className="sheet p-5" aria-labelledby="upload-heading">
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
        <h2 id="upload-heading" className="font-display text-[19px] text-fg">
          Add documents
        </h2>
        <span className="font-mono text-[11.5px] text-faint">
          PDF, DOCX, XLSX, CSV, TXT, PNG, JPG
        </span>
      </div>

      <div
        className="drop-surface mt-4 flex flex-col gap-4 p-5 sm:flex-row sm:items-center sm:justify-between"
        data-active={dragging}
        onDragEnter={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          pick(event.dataTransfer.files[0]);
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          className="sr-only"
          onChange={(event) => pick(event.target.files?.[0])}
        />

        {file ? (
          <div className="flex min-w-0 items-center gap-3">
            <FileText className="h-4 w-4 flex-none text-muted" aria-hidden="true" />
            <div className="min-w-0">
              <p className="truncate text-[14px] text-fg">{file.name}</p>
              <p className="font-mono text-[12px] text-muted tabular">
                {formatFileSize(file.size)}
              </p>
            </div>
          </div>
        ) : (
          <div className="flex items-center gap-3">
            <Upload className="h-4 w-4 flex-none text-muted" aria-hidden="true" />
            <p className="text-[14px] text-fg">
              Drop a document here, or choose a file.
            </p>
          </div>
        )}

        <div className="flex items-center gap-3">
          {file ? (
            <>
              <button
                type="button"
                className="text-link"
                onClick={() => choose(null)}
                disabled={upload.isPending}
              >
                Remove
              </button>
              <Button onClick={handleUpload} disabled={upload.isPending}>
                {upload.isPending ? "Uploading…" : "Upload document"}
              </Button>
            </>
          ) : (
            <Button variant="quiet" onClick={() => inputRef.current?.click()}>
              Choose a file
            </Button>
          )}
        </div>
      </div>

      <p className="mt-3 text-[13px] text-muted">
        Ingestion runs in the background. This list updates as each file
        finishes.
      </p>

      {error && (
        <p className="mt-2 text-[13px] text-stop" role="alert">
          {error}
        </p>
      )}

      {accepted && (
        <p className="mt-2 text-[13px] text-muted" role="status">
          {accepted.duplicate
            ? accepted.message
            : "Accepted. Indexing now — the row below will change to Ready when every chunk is embedded."}
        </p>
      )}
    </section>
  );
}
