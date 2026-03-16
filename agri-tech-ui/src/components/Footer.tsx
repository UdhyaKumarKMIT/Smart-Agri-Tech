import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { Sprout } from "lucide-react";

const Footer = () => {
  const { t } = useTranslation();

  const navLinks = [
    { to: "/", label: t("nav.home") },
    { to: "/crop-recommendation", label: t("nav.cropRecommendation") },
    { to: "/fertilizer-recommendation", label: t("nav.fertilizerRecommendation") },
    { to: "/yield-prediction", label: t("nav.yieldPrediction") },
    { to: "/contact", label: t("nav.contact") },
  ];

  return (
    <footer className="border-t border-border bg-card">
      <div className="container mx-auto px-4 py-10">
        {/* Top row: Logo + Nav */}
        <div className="flex flex-col items-center gap-6 sm:flex-row sm:justify-between">
          <div className="flex items-center gap-2 font-display text-lg font-bold text-primary">
            <Sprout className="h-6 w-6" />
            Smart Agriculture Website for Farmers
          </div>
          <nav className="flex flex-wrap justify-center gap-4">
            {navLinks.map((link) => (
              <Link
                key={link.to}
                to={link.to}
                className="text-sm text-muted-foreground transition-colors hover:text-primary"
              >
                {link.label}
              </Link>
            ))}
          </nav>
        </div>

        {/* Divider */}
        <div className="my-6 border-t border-border" />

        {/* Bottom */}
        <div className="text-center text-sm text-muted-foreground">
          <p>{t("footer.developed")} UdhyaKumarKMIT &middot; Anna University MIT Campus – 2026</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
