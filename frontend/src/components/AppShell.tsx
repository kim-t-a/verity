import { FileText, MessageSquareText } from "lucide-react";
import { NavLink, Outlet } from "react-router-dom";

import { cn } from "@/lib/cn";

const NAV = [
  { to: "/documents", label: "Documents", icon: FileText },
  { to: "/conversation", label: "Conversation", icon: MessageSquareText },
];

export function AppShell() {
  return (
    <div className="flex h-dvh flex-col bg-bg lg:flex-row">
      <aside
        className="flex items-center justify-between border-b border-rule bg-surface px-4 py-3 lg:h-full lg:w-16 lg:flex-none lg:flex-col lg:justify-start lg:border-b-0 lg:border-r lg:px-0 lg:py-5"
        aria-label="Primary"
      >
        <NavLink
          to="/documents"
          aria-label="Verity"
          className="flex items-baseline gap-2 rounded-[4px] lg:mb-8 lg:justify-center"
        >
          <span className="font-display text-[22px] leading-none text-accent lg:text-[24px]">
            V
          </span>
          <span className="font-display text-[20px] leading-none text-fg lg:sr-only">
            Verity
          </span>
        </NavLink>

        <nav className="flex items-center gap-1 lg:flex-col lg:gap-2">
          {NAV.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                title={item.label}
                className={({ isActive }) =>
                  cn(
                    "flex items-center gap-2 rounded-[4px] px-3 py-2 text-[13px] transition-colors duration-quick lg:h-9 lg:w-9 lg:justify-center lg:px-0",
                    isActive
                      ? "bg-accent-wash text-accent"
                      : "text-muted hover:bg-surface-2 hover:text-fg"
                  )
                }
              >
                <Icon className="h-4 w-4" aria-hidden="true" />
                <span className="lg:sr-only">{item.label}</span>
              </NavLink>
            );
          })}
        </nav>
      </aside>

      <main className="flex min-h-0 flex-1 flex-col">
        <Outlet />
      </main>
    </div>
  );
}
