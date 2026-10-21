/**
 * One fetch wrapper for the whole app.
 *
 * Two details matter and both were previously wrong here:
 *  - a 204 response has no body, so calling res.json() throws on every
 *    successful delete;
 *  - a FormData body must not carry a Content-Type header, because the
 *    browser has to add the multipart boundary itself.
 */

const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/+$/, "");

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function readDetail(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    if (body && typeof body === "object" && "detail" in body) {
      const detail = (body as { detail: unknown }).detail;
      if (typeof detail === "string") return detail;
      if (Array.isArray(detail)) {
        const messages = detail
          .map((entry) =>
            entry && typeof entry === "object" && "msg" in entry
              ? String((entry as { msg: unknown }).msg)
              : null
          )
          .filter((message): message is string => message !== null);
        if (messages.length) return messages.join("; ");
      }
    }
  } catch {
    // Not JSON. Fall through to the status line.
  }
  return `${response.status} ${response.statusText}`;
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);

  if (!response.ok) {
    throw new ApiError(await readDetail(response), response.status);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  const text = await response.text();
  if (!text) return undefined as T;
  return JSON.parse(text) as T;
}

export function jsonBody(method: string, body: unknown): RequestInit {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}
