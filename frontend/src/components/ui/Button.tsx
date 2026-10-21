import { forwardRef, type ButtonHTMLAttributes } from "react";

import { cn } from "@/lib/cn";

type Variant = "primary" | "quiet";
type Size = "sm" | "md";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
}

/**
 * One primary action per screen; everything else is a hairline outline.
 * There is no icon-only variant here — icon buttons pass `aria-label`.
 */
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "primary", size = "md", type = "button", ...props }, ref) => {
    const variants: Record<Variant, string> = {
      primary:
        "bg-accent text-[var(--on-accent)] font-medium hover:brightness-95 active:brightness-90",
      quiet:
        "bg-transparent text-muted border border-rule hover:border-accent hover:text-accent",
    };
    const sizes: Record<Size, string> = {
      sm: "px-3 py-1.5 text-[13px]",
      md: "px-4 py-2.5 text-sm",
    };

    return (
      <button
        ref={ref}
        type={type}
        className={cn(
          "inline-flex items-center justify-center gap-2 rounded-[4px] transition-[filter,color,border-color] duration-quick disabled:cursor-not-allowed disabled:opacity-50",
          variants[variant],
          sizes[size],
          className
        )}
        {...props}
      />
    );
  }
);

Button.displayName = "Button";
