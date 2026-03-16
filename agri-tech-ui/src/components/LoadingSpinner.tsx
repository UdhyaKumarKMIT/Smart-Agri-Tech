import { Loader2 } from "lucide-react";

const LoadingSpinner = ({ text }: { text?: string }) => (
  <div className="flex flex-col items-center gap-3 py-8">
    <Loader2 className="h-8 w-8 animate-spin text-primary" />
    {text && <p className="text-sm text-muted-foreground">{text}</p>}
  </div>
);

export default LoadingSpinner;
