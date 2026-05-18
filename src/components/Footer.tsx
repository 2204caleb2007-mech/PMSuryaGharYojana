import { Link } from "react-router-dom";
import { Satellite } from "lucide-react";

const Footer = () => {
  return (
    <footer className="relative border-t border-border bg-background/80">
      <div className="container mx-auto px-4 md:px-6 py-12">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-8 mb-12">
          {/* Brand */}
          <div className="max-w-md">
            <div className="flex items-center gap-2 mb-2">
              <Satellite className="w-5 h-5 text-primary" />
              <span className="font-display font-bold text-xl text-foreground">Surya Sys.</span>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              Advanced solar analytics platform leveraging satellite imagery and AI to accelerate India's solar revolution.
            </p>
          </div>

          {/* Quick Links */}
          <ul className="flex flex-wrap items-center gap-6 md:gap-8">
            {[
              { label: "Home", path: "/" },
              { label: "Solar Analysis", path: "/analysis" },
              { label: "Subsidy Calculator", path: "/calculator" },
              { label: "How It Works", path: "/process" },
              { label: "Contact", path: "/contact" },
            ].map((link) => (
              <li key={link.path}>
                <Link
                  to={link.path}
                  className="text-sm font-medium text-foreground hover:text-primary transition-colors"
                >
                  {link.label}
                </Link>
              </li>
            ))}
          </ul>
        </div>
        <p className="text-sm text-muted-foreground text-center">
          © 2026 Surya Systems. All rights reserved.
        </p>
      </div>
    </footer>
  );
};

export default Footer;
