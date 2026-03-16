import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { motion } from "framer-motion";
import { Sprout, ArrowLeft, FileText, ExternalLink } from "lucide-react";

// API Base URL - change this to your backend URL
const API_BASE_URL = "http://localhost:5000";

interface CropInput {
  nitrogen: string;
  phosphorus: string;
  potassium: string;
  temperature: string;
  humidity: string;
  ph: string;
  rainfall: string;
}

interface CropResult {
  crop: string;
  confidence: number;
  input_summary: {
    nitrogen: number;
    phosphorus: number;
    potassium: number;
    temperature: number;
    humidity: number;
    ph: number;
    rainfall: number;
  };
}

const defaultValues: CropInput = {
  nitrogen: "90",
  phosphorus: "42",
  potassium: "43",
  temperature: "20.88",
  humidity: "82.00",
  ph: "6.50",
  rainfall: "202.94",
};

const CropRecommendation = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [form, setForm] = useState<CropInput>(defaultValues);
  const [result, setResult] = useState<CropResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // API function for crop prediction
  const predictCrop = async (data: CropInput) => {
    const response = await fetch(`${API_BASE_URL}/predict-crop`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        nitrogen: parseFloat(data.nitrogen) || 0,
        phosphorus: parseFloat(data.phosphorus) || 0,
        potassium: parseFloat(data.potassium) || 0,
        temperature: parseFloat(data.temperature) || 0,
        humidity: parseFloat(data.humidity) || 0,
        ph: parseFloat(data.ph) || 0,
        rainfall: parseFloat(data.rainfall) || 0,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to predict crop');
    }

    return await response.json();
  };

  // Function to open latest report
  const openLatestReport = async () => {
    try {
      window.open(`${API_BASE_URL}/report/latest`, '_blank');
    } catch (error) {
      console.error('Error opening report:', error);
      alert('Failed to open report. Please check if the server is running.');
    }
  };

  const handleChange = (field: keyof CropInput, value: string) => {
    // Allow only numbers and decimal point
    if (value === '' || /^\d*\.?\d*$/.test(value)) {
      setForm((prev) => ({ ...prev, [field]: value }));
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    setError(null);
    
    try {
      const data = await predictCrop(form);
      setResult({
        crop: data.crop,
        confidence: data.confidence || 0.95,
        input_summary: {
          nitrogen: parseFloat(form.nitrogen) || 0,
          phosphorus: parseFloat(form.phosphorus) || 0,
          potassium: parseFloat(form.potassium) || 0,
          temperature: parseFloat(form.temperature) || 0,
          humidity: parseFloat(form.humidity) || 0,
          ph: parseFloat(form.ph) || 0,
          rainfall: parseFloat(form.rainfall) || 0,
        }
      });
    } catch (err: any) {
      setError(err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const fields: { key: keyof CropInput; label: string; placeholder: string }[] = [
    { key: "nitrogen", label: t("crop.nitrogen"), placeholder: "e.g., 90" },
    { key: "phosphorus", label: t("crop.phosphorus"), placeholder: "e.g., 42" },
    { key: "potassium", label: t("crop.potassium"), placeholder: "e.g., 43" },
    { key: "temperature", label: t("crop.temperature"), placeholder: "e.g., 20.88" },
    { key: "humidity", label: t("crop.humidity"), placeholder: "e.g., 82.00" },
    { key: "ph", label: t("crop.ph"), placeholder: "e.g., 6.50" },
    { key: "rainfall", label: t("crop.rainfall"), placeholder: "e.g., 202.94" },
  ];

  return (
    <div className="container mx-auto px-4 py-10">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
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
            className="flex items-center gap-2 rounded-lg bg-secondary px-4 py-2 text-sm font-medium text-secondary-foreground transition-colors hover:bg-secondary/80"
          >
            <FileText className="h-4 w-4" />
            View Latest Report
            <ExternalLink className="h-3 w-3" />
          </button>
        </div>
        
        <h1 className="mb-2 flex items-center gap-2 font-display text-2xl font-bold text-foreground sm:text-3xl">
          <Sprout className="h-7 w-7 text-primary" />
          {t("crop.title")}
        </h1>
        <p className="mb-8 text-muted-foreground">{t("crop.subtitle")}</p>

        {/* Input Form */}
        <form onSubmit={handleSubmit} className="mb-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {fields.map(({ key, label, placeholder }) => (
            <div key={key} className="flex flex-col gap-1.5">
              <label className="text-sm font-medium text-foreground">
                {label} <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                value={form[key]}
                onChange={(e) => handleChange(key, e.target.value)}
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
              className="rounded-lg bg-primary px-8 py-3 text-sm font-semibold text-primary-foreground transition-all hover:bg-primary/90 hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? "Predicting..." : t("crop.predict")}
            </button>
          </div>
        </form>

        {/* Error Message */}
        {error && (
          <div className="mb-8 rounded-lg bg-red-50 p-4 text-red-800 dark:bg-red-900/20 dark:text-red-200">
            {error}
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mb-8 flex justify-center"
          >
            <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
          </motion.div>
        )}

        {/* Results Section */}
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            {/* Result Card */}
            <div className="rounded-xl border border-border bg-gradient-to-br from-card to-card/50 p-6 shadow-sm">
              <div className="flex flex-col items-start justify-between gap-4">
                <div className="w-full">
                  <h3 className="mb-2 text-sm font-medium text-muted-foreground">
                    ✅ Recommended Crop
                  </h3>
                  <p className="text-4xl font-bold text-primary">{result.crop}</p>
                  
                  {/* Input Summary */}
                  <div className="mt-6 border-t pt-6">
                    <h4 className="mb-3 text-sm font-medium text-muted-foreground">📝 Input Summary</h4>
                    <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                      <div>
                        <p className="text-xs text-muted-foreground">NPK</p>
                        <p className="font-medium">
                          {result.input_summary.nitrogen}-{result.input_summary.phosphorus}-{result.input_summary.potassium}
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Temperature</p>
                        <p className="font-medium">{result.input_summary.temperature.toFixed(2)}°C</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Humidity</p>
                        <p className="font-medium">{result.input_summary.humidity.toFixed(2)}%</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">pH</p>
                        <p className="font-medium">{result.input_summary.ph.toFixed(2)}</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Rainfall</p>
                        <p className="font-medium">{result.input_summary.rainfall.toFixed(2)} mm</p>
                      </div>
                    </div>
                  </div>

                  {/* Confidence */}
                  <div className="mt-4 flex items-center gap-2">
                    <div className="h-2 w-32 rounded-full bg-muted">
                      <div 
                        className="h-2 rounded-full bg-primary" 
                        style={{ width: `${Math.round(result.confidence * 100)}%` }}
                      />
                    </div>
                    <span className="text-sm font-medium text-muted-foreground">
                      {Math.round(result.confidence * 100)}% Confidence
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Report Link */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
              className="rounded-lg border border-border bg-muted/30 p-4"
            >
              <p className="flex items-center gap-2 text-sm text-muted-foreground">
                <FileText className="h-4 w-4" />
                Report generated successfully
                <button
                  onClick={openLatestReport}
                  className="ml-auto flex items-center gap-1 text-primary hover:underline"
                >
                  View Full Report
                  <ExternalLink className="h-3 w-3" />
                </button>
              </p>
            </motion.div>
          </motion.div>
        )}
      </motion.div>
    </div>
  );
};

export default CropRecommendation;