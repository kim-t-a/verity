import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";

import { AppShell } from "@/components/AppShell";
import { ConversationScreen } from "@/components/conversation/ConversationScreen";
import { DocumentsScreen } from "@/components/documents/DocumentsScreen";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<Navigate to="/documents" replace />} />
          <Route path="documents" element={<DocumentsScreen />} />
          <Route path="conversation" element={<ConversationScreen />} />
          {/* The old route name is kept as a redirect so a bookmark survives. */}
          <Route path="chat" element={<Navigate to="/conversation" replace />} />
          <Route path="*" element={<Navigate to="/documents" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
