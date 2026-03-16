import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { motion } from "framer-motion";
import { TrendingUp, ArrowLeft } from "lucide-react";
import { predictYield, type YieldInput, type YieldResult } from "@/services/api";
import FeatureBarChart from "@/components/FeatureBarChart";
import LoadingSpinner from "@/components/LoadingSpinner";

const cropTypes = ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Barley", "Soybean", "Millet"];

const YieldPrediction = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [form, setForm] = useState<YieldInput>({
    cropType: "",
    area: 5,
    rainfall: 200,
    temperature: 28,
    fertilizer: 100,
  });
  const [result, setResult] = useState<YieldResult | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    try {
      const data = await predictYield(form);
      setResult(data);
    } finally {
      setLoading(false);
    }
  };

  const numFields: { key: keyof YieldInput; label: string; min: number; max: number; step: number }[] = [
    { key: "area", label: t("yield.area"), min: 0.1, max: 1000, step: 0.1 },
    { key: "rainfall", label: t("yield.rainfall"), min: 0, max: 500, step: 1 },
    { key: "temperature", label: t("yield.temperature"), min: 0, max: 50, step: 0.1 },
    { key: "fertilizer", label: t("yield.fertilizer"), min: 0, max: 500, step: 1 },
  ];

  return (
    <div className="container mx-auto px-4 py-10">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <button
          onClick={() => navigate(-1)}
          className="mb-4 flex items-center gap-1 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground"
        >
          <ArrowLeft className="h-4 w-4" />
          {t("common.back")}
        </button>
        <h1 className="mb-2 flex items-center gap-2 font-display text-2xl font-bold text-foreground sm:text-3xl">
          <TrendingUp className="h-7 w-7 text-primary" />
          {t("yield.title")}
        </h1>
        <p className="mb-8 text-muted-foreground">{t("yield.subtitle")}</p>

        <form onSubmit={handleSubmit} className="mb-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {/* Crop Type */}
          <div className="flex flex-col gap-1.5">
            <label className="text-sm font-medium text-foreground">{t("yield.cropType")}</label>
            <select
              value={form.cropType}
              onChange={(e) => setForm((p) => ({ ...p, cropType: e.target.value }))}
              className="rounded-lg border border-input bg-background px-3 py-2.5 text-sm text-foreground outline-none ring-ring transition-shadow focus:ring-2"
              required
            >
              <option value="">{t("common.selectOption")}</option>
              {cropTypes.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>

          {numFields.map(({ key, label, min, max, step }) => (
            <div key={key} className="flex flex-col gap-1.5">
              <label className="text-sm font-medium text-foreground">{label}</label>
              <input
                type="number"
                min={min}
                max={max}
                step={step}
                value={form[key]}
                onChange={(e) => setForm((p) => ({ ...p, [key]: parseFloat(e.target.value) || 0 }))}
                className="rounded-lg border border-input bg-background px-3 py-2.5 text-sm text-foreground outline-none ring-ring transition-shadow focus:ring-2"
                required
              />
            </div>
          ))}

          <div className="flex items-end sm:col-span-2 lg:col-span-3">
            <button
              type="submit"
              disabled={loading}
              className="rounded-lg bg-primary px-6 py-2.5 text-sm font-semibold text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-50"
            >
              {t("yield.submit")}
            </button>
          </div>
        </form>

        {loading && <LoadingSpinner text={t("yield.loading")} />}

        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            <div className="card-elevated rounded-xl border border-border bg-card p-6">
              <h3 className="mb-1 text-sm font-medium text-muted-foreground">{t("yield.result")}</h3>
              <p className="text-3xl font-bold text-primary">
                {result.predictedYield} <span className="text-lg font-normal text-muted-foreground">{t("common.tonnes")}</span>
              </p>
            </div>
            <FeatureBarChart data={result.factors} title={t("yield.factorChart")} />
          </motion.div>
        )}
      </motion.div>
    </div>
  );
};

export default YieldPrediction;
