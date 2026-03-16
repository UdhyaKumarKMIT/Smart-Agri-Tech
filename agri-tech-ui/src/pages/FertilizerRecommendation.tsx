import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { motion } from "framer-motion";
import { FlaskConical, ArrowLeft, FileText, ExternalLink } from "lucide-react";

// Crop types from your dataset
const crops = [
  "Arhar/Tur", "Bajra", "Barley", "Coriander", "Cotton (Lint)", "Cowpea (Lobia)",
  "Dry Chillies", "Garlic", "Ginger", "Gram (Chickpea)", "Groundnut", "Jowar",
  "Linseed (Flax)", "Maize (Grain)", "Maize (Fodder)", "Masoor (Red Lentil)",
  "Moong (Green Gram)", "Onion", "Peas & Beans (Pulses)", "Potato",
  "Ragi (Finger Millet)", "Rapeseed & Mustard", "Rice", "Safflower",
  "Sugarcane", "Sunflower", "Turmeric", "Urad (Black Gram)", "Urad Bean", "Wheat"
];

// Soil types from your dataset
const soilTypes = [
  "Sandy Loam", "Loamy", "Sand", "Clay Loam", "Clay", "Sandy", "Loamy Sand",
  "Loam", "Red Clay Loam", "Red Loam", "Silty Loam", "Alluvial", "Black Soil"
];

// API Base URL - change this to your backend URL
const API_BASE_URL = "http://localhost:5000";

interface FertilizerInput {
  temperature: string;
  humidity: string;
  moisture: string;
  soilType: string;
  crop: string;
  nitrogen: string;
  potassium: string;
  phosphorus: string;
}

interface FertilizerResult {
  fertilizer: string;
  input_summary: {
    nitrogen: number;
    phosphorus: number;
    potassium: number;
    temperature: number;
    humidity: number;
    soil_moisture: number;
  };
}

const FertilizerRecommendation = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [form, setForm] = useState<FertilizerInput>({
    temperature: "",
    humidity: "",
    moisture: "",
    soilType: "",
    crop: "",
    nitrogen: "",
    potassium: "",
    phosphorus: "",
  });
  const [result, setResult] = useState<FertilizerResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [reportLoading, setReportLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const predictFertilizer = async (data: FertilizerInput) => {
    const response = await fetch(`${API_BASE_URL}/api/fertilizer/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        soil_type: data.soilType,
        crop_type: data.crop,
        nitrogen: parseFloat(data.nitrogen) || 0,
        phosphorus: parseFloat(data.phosphorus) || 0,
        potassium: parseFloat(data.potassium) || 0,
        temperature: parseFloat(data.temperature) || 0,
        humidity: parseFloat(data.humidity) || 0,
        soil_moisture: parseFloat(data.moisture) || 0,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to predict fertilizer');
    }

    return await response.json();
  };

  const openLatestReport = async () => {
    setReportLoading(true);
    try {
      // First check if report exists by calling the endpoint
      const response = await fetch(`${API_BASE_URL}/fertilizer-report/latest`, {
        method: 'HEAD',
      });
      
      if (response.ok) {
        // Open in new tab
        window.open(`${API_BASE_URL}/fertilizer-report/latest`, '_blank');
      } else {
        // If report doesn't exist, generate one first
        const generateResponse = await fetch(`${API_BASE_URL}/api/fertilizer/generate-report`, {
          method: 'POST',
        });
        
        if (generateResponse.ok) {
          window.open(`${API_BASE_URL}/fertilizer-report/latest`, '_blank');
        } else {
          alert('Failed to generate report. Please try again.');
        }
      }
    } catch (error) {
      console.error('Error opening report:', error);
      alert('Failed to open report. Please check if the server is running.');
    } finally {
      setReportLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    setError(null);
    
    try {
      const data = await predictFertilizer(form);
      setResult({
        fertilizer: data.recommendation,
        input_summary: data.input_summary
      });
    } catch (err: any) {
      setError(err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (field: keyof FertilizerInput, value: string) => {
    setForm(prev => ({ ...prev, [field]: value }));
  };

  const numericFields: { key: keyof FertilizerInput; label: string; placeholder: string }[] = [
    { key: "temperature", label: t("fertilizer.temperature"), placeholder: "e.g., 29.89" },
    { key: "humidity", label: t("fertilizer.humidity"), placeholder: "e.g., 69.58" },
    { key: "moisture", label: t("fertilizer.moisture"), placeholder: "e.g., 27.33" },
    { key: "nitrogen", label: t("fertilizer.nitrogen"), placeholder: "e.g., 17.47" },
    { key: "potassium", label: t("fertilizer.potassium"), placeholder: "e.g., 24.59" },
    { key: "phosphorus", label: t("fertilizer.phosphorus"), placeholder: "e.g., 32.87" },
  ];

  return (
    <div className="container mx-auto px-4 py-10">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 20 }}
        transition={{ duration: 0.5 }}
      >
        <div className="flex items-center justify-between mb-4">
          <button
            onClick={() => navigate(-1)}
            className="flex items-center gap-1 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground"
          >
            <ArrowLeft className="h-4 w-4" />
            {t("common.back")}
          </button>
          
          <button
            onClick={openLatestReport}
            disabled={reportLoading}
            className="flex items-center gap-2 rounded-lg bg-secondary px-4 py-2 text-sm font-medium text-secondary-foreground transition-colors hover:bg-secondary/80 disabled:opacity-50"
          >
            <FileText className="h-4 w-4" />
            {reportLoading ? "Loading..." : "View Latest Report"}
            <ExternalLink className="h-3 w-3" />
          </button>
        </div>

        <h1 className="mb-2 flex items-center gap-2 font-display text-2xl font-bold text-foreground sm:text-3xl">
          <FlaskConical className="h-7 w-7 text-primary" />
          {t("fertilizer.title")}
        </h1>
        <p className="mb-8 text-muted-foreground">{t("fertilizer.subtitle")}</p>

        <form onSubmit={handleSubmit} className="mb-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {/* Soil Type Dropdown */}
          <div className="flex flex-col gap-1.5">
            <label className="text-sm font-medium text-foreground">
              {t("fertilizer.soilType")} <span className="text-red-500">*</span>
            </label>
            <select
              value={form.soilType}
              onChange={(e) => handleInputChange("soilType", e.target.value)}
              className="rounded-lg border border-input bg-background px-3 py-2.5 text-sm text-foreground outline-none ring-ring transition-shadow focus:ring-2"
              required
            >
              <option value="">{t("common.selectOption")}</option>
              {soilTypes.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>

          {/* Crop Dropdown */}
          <div className="flex flex-col gap-1.5">
            <label className="text-sm font-medium text-foreground">
              {t("fertilizer.cropType")} <span className="text-red-500">*</span>
            </label>
            <select
              value={form.crop}
              onChange={(e) => handleInputChange("crop", e.target.value)}
              className="rounded-lg border border-input bg-background px-3 py-2.5 text-sm text-foreground outline-none ring-ring transition-shadow focus:ring-2"
              required
            >
              <option value="">{t("common.selectOption")}</option>
              {crops.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>

          {/* Numeric Inputs (as strings) */}
          {numericFields.map(({ key, label, placeholder }) => (
            <div key={key} className="flex flex-col gap-1.5">
              <label className="text-sm font-medium text-foreground">
                {label} <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                value={form[key]}
                onChange={(e) => handleInputChange(key, e.target.value)}
                placeholder={placeholder}
                className="rounded-lg border border-input bg-background px-3 py-2.5 text-sm text-foreground outline-none ring-ring transition-shadow focus:ring-2"
                required
                pattern="^\d*\.?\d*$"
                title="Please enter a valid number"
              />
            </div>
          ))}

          <div className="flex items-end sm:col-span-2 lg:col-span-3 xl:col-span-4">
            <button
              type="submit"
              disabled={loading}
              className="rounded-lg bg-primary px-6 py-2.5 text-sm font-semibold text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-50"
            >
              {loading ? "Predicting..." : t("fertilizer.submit")}
            </button>
          </div>
        </form>

        {loading && (
          <div className="flex justify-center py-8">
            <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
          </div>
        )}

        {error && (
          <div className="rounded-lg bg-red-50 p-4 text-red-800 dark:bg-red-900/20 dark:text-red-200">
            {error}
          </div>
        )}

        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            <div className="card-elevated rounded-xl border border-border bg-card p-6">
              <h3 className="mb-1 text-sm font-medium text-muted-foreground">✅ Recommendation</h3>
              <p className="text-3xl font-bold text-primary">{result.fertilizer}</p>
              
              <div className="mt-6 border-t pt-6">
                <h4 className="mb-3 text-sm font-medium text-muted-foreground">📝 Input Summary</h4>
                <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
                  <div>
                    <p className="text-xs text-muted-foreground">NPK</p>
                    <p className="font-medium">
                      {result.input_summary.nitrogen}-{result.input_summary.phosphorus}-{result.input_summary.potassium}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Temperature</p>
                    <p className="font-medium">{result.input_summary.temperature}°C</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Humidity</p>
                    <p className="font-medium">{result.input_summary.humidity}%</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Soil Moisture</p>
                    <p className="font-medium">{result.input_summary.soil_moisture}%</p>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </motion.div>
    </div>
  );
};

export default FertilizerRecommendation;