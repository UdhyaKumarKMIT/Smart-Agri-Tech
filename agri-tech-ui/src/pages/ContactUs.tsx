import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { motion } from "framer-motion";
import { ArrowLeft, MapPin, Mail, Phone, Send } from "lucide-react";

const ContactUs = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [form, setForm] = useState({ name: "", email: "", message: "" });
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitted(true);
    setForm({ name: "", email: "", message: "" });
    setTimeout(() => setSubmitted(false), 3000);
  };

  return (
    <div className="container mx-auto px-4 py-10">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
        <button
          onClick={() => navigate(-1)}
          className="mb-4 flex items-center gap-1 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground"
        >
          <ArrowLeft className="h-4 w-4" />
          {t("common.back")}
        </button>

        <h1 className="mb-2 flex items-center gap-2 font-display text-2xl font-bold text-foreground sm:text-3xl">
          <Mail className="h-7 w-7 text-primary" />
          {t("contact.title")}
        </h1>
        <p className="mb-8 text-muted-foreground">{t("contact.subtitle")}</p>

        <div className="grid gap-8 lg:grid-cols-2">
          {/* Contact Form */}
          <div className="card-elevated rounded-xl border border-border bg-card p-6">
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="flex flex-col gap-1.5">
                <label className="text-sm font-medium text-foreground">{t("contact.name")}</label>
                <input
                  type="text"
                  value={form.name}
                  onChange={(e) => setForm((p) => ({ ...p, name: e.target.value }))}
                  className="rounded-lg border border-input bg-background px-3 py-2.5 text-sm text-foreground outline-none ring-ring transition-shadow focus:ring-2"
                  required
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <label className="text-sm font-medium text-foreground">{t("contact.email")}</label>
                <input
                  type="email"
                  value={form.email}
                  onChange={(e) => setForm((p) => ({ ...p, email: e.target.value }))}
                  className="rounded-lg border border-input bg-background px-3 py-2.5 text-sm text-foreground outline-none ring-ring transition-shadow focus:ring-2"
                  required
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <label className="text-sm font-medium text-foreground">{t("contact.message")}</label>
                <textarea
                  rows={5}
                  value={form.message}
                  onChange={(e) => setForm((p) => ({ ...p, message: e.target.value }))}
                  className="resize-none rounded-lg border border-input bg-background px-3 py-2.5 text-sm text-foreground outline-none ring-ring transition-shadow focus:ring-2"
                  required
                />
              </div>
              <button
                type="submit"
                className="flex items-center gap-2 rounded-lg bg-primary px-6 py-2.5 text-sm font-semibold text-primary-foreground transition-colors hover:bg-primary/90"
              >
                <Send className="h-4 w-4" />
                {t("contact.send")}
              </button>
              {submitted && (
                <p className="text-sm font-medium text-primary">{t("contact.success")}</p>
              )}
            </form>
          </div>

          {/* Info & Map */}
          <div className="space-y-6">
            <div className="card-elevated rounded-xl border border-border bg-card p-6 space-y-4">
              <div className="flex items-start gap-3">
                <MapPin className="mt-0.5 h-5 w-5 text-primary" />
                <div>
                  <h3 className="font-semibold text-foreground">{t("contact.location")}</h3>
                  <p className="text-sm text-muted-foreground">MIT Campus, Anna University, Chromepet, Chennai – 600044</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <Mail className="mt-0.5 h-5 w-5 text-primary" />
                <div>
                  <h3 className="font-semibold text-foreground">{t("contact.emailLabel")}</h3>
                  <p className="text-sm text-muted-foreground">contact@smartagri.edu</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <Phone className="mt-0.5 h-5 w-5 text-primary" />
                <div>
                  <h3 className="font-semibold text-foreground">{t("contact.phone")}</h3>
                  <p className="text-sm text-muted-foreground">+91 44 2251 6000</p>
                </div>
              </div>
            </div>

            <div className="overflow-hidden rounded-xl border border-border">
              <iframe
                src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d680!2d80.136063!3d12.9463776!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x3a525fac0ac17717%3A0xf64005cb4dc6d844!2sMIT%20Main%20Gate!5e0!3m2!1sen!2sin!4v1700000000000!5m2!1sen!2sin"
                width="100%"
                height="300"
                style={{ border: 0 }}
                allowFullScreen
                loading="lazy"
                referrerPolicy="no-referrer-when-downgrade"
                title="MIT Campus Location"
              />
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default ContactUs;
