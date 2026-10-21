import { DocumentTable } from "@/components/documents/DocumentTable";
import { UploadSheet } from "@/components/documents/UploadSheet";
import { Button } from "@/components/ui/Button";
import { useDocuments } from "@/hooks/useDocuments";
import { formatCount, formatFileSize } from "@/lib/format";

export function DocumentsScreen({ workspaceId = 1 }: { workspaceId?: number }) {
  const query = useDocuments(workspaceId);
  const documents = query.data ?? [];

  const ready = documents.filter((document) => document.status === "ready").length;
  const failed = documents.filter((document) => document.status === "failed").length;
  const chunks = documents.reduce((total, document) => total + document.chunk_count, 0);
  const bytes = documents.reduce((total, document) => total + document.size_bytes, 0);
  const working = documents.some(
    (document) => document.status === "pending" || document.status === "processing"
  );

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto w-full max-w-wide px-5 py-8 sm:px-8">
        <header>
          <h1 className="font-display text-[30px] leading-tight text-fg sm:text-[32px]">
            Documents
          </h1>
          <p className="mt-1 max-w-[60ch] text-[14px] text-muted">
            One shared corpus. Upload files here, then ask questions about
            them — every answer names the file and location it came from.
          </p>
        </header>

        <div className="mt-6">
          <UploadSheet workspaceId={workspaceId} />
        </div>

        <section className="mt-8" aria-labelledby="corpus-heading">
          <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
            <h2 id="corpus-heading" className="font-display text-[20px] text-fg">
              Corpus
            </h2>
            {working && (
              <span className="inline-flex items-center gap-1.5 text-[13px] text-muted">
                <span
                  className="status-square"
                  style={{ background: "var(--accent)" }}
                  aria-hidden="true"
                />
                Indexing — this list updates automatically
              </span>
            )}
          </div>

          {query.isError ? (
            <div className="sheet mt-3 p-6">
              <p className="text-[14px] text-fg">Could not load the corpus.</p>
              <p className="mt-1 text-[13px] text-muted" role="alert">
                {query.error instanceof Error
                  ? query.error.message
                  : "The request failed."}
              </p>
              <Button
                variant="quiet"
                className="mt-4"
                onClick={() => void query.refetch()}
              >
                Try again
              </Button>
            </div>
          ) : query.isLoading ? (
            <p className="mt-4 font-mono text-[12.5px] text-faint">
              Loading documents…
            </p>
          ) : documents.length === 0 ? (
            <div className="sheet mt-3 p-6">
              <p className="font-display text-[17px] text-fg">
                The corpus is empty.
              </p>
              <p className="mt-1 max-w-[58ch] text-[13.5px] text-muted">
                Use Add documents above to upload a PDF, Word file, spreadsheet,
                text file, or image. Verity reads it, splits it into chunks, and
                embeds every chunk before the document is marked Ready.
              </p>
            </div>
          ) : (
            <>
              <p className="mt-3 font-mono text-[12.5px] text-muted tabular">
                {formatCount(documents.length, "document")}, {ready} ready
                {failed > 0 ? `, ${failed} failed` : ""},{" "}
                {formatCount(chunks, "chunk")}, {formatFileSize(bytes)}
              </p>
              <div className="mt-2">
                <DocumentTable
                  documents={documents}
                  workspaceId={workspaceId}
                />
              </div>
            </>
          )}
        </section>
      </div>
    </div>
  );
}
