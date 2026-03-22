import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowLeft, Leaf, AlertCircle, BarChart3, TrendingUp, TrendingDown, Info } from "lucide-react";

interface YieldInput {
  district: string;
  crop: string;
}

interface YieldResult {
  success: boolean;
  district: string;
  crop: string;
  area: number;
  predicted_yield: number;
  predicted_production: number;
  model_used: string;
  features: {
    NDVI: number;
    NDWI: number;
    EVI: number;
    SAVI: number;
    SMAI: number;
    Precipitation: number;
    CQI: number;
  };
}

interface DistrictInsights {
  district: string;
  total_area: number;
  total_production: number;
  avg_yield: number;
  top_crop: {
    name: string;
    production: number;
  };
  bottom_crop: {
    name: string;
    production: number;
  };
  best_yield_crop: {
    name: string;
    yield: number;
  };
  worst_yield_crop: {
    name: string;
    yield: number;
  };
  crop_stats: Array<{
    crop: string;
    area: number;
    production: number;
    yield: number;
  }>;
  total_crops: number;
}

const YieldPrediction = () => {
  const navigate = useNavigate();
  const API_BASE_URL = 'http://localhost:5000';
  
  const [districts, setDistricts] = useState<string[]>([]);
  const [crops, setCrops] = useState<string[]>([]);
  const [insights, setInsights] = useState<DistrictInsights | null>(null);
  const [loadingDistricts, setLoadingDistricts] = useState(true);
  const [loadingCrops, setLoadingCrops] = useState(false);
  const [loadingInsights, setLoadingInsights] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  
  const [form, setForm] = useState<YieldInput>({
    district: "",
    crop: "",
  });
  
  const [result, setResult] = useState<YieldResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch districts on component mount
  useEffect(() => {
    fetchDistricts();
  }, []);

  // Fetch crops when district changes
  useEffect(() => {
    if (form.district) {
      fetchCrops(form.district);
    } else {
      setCrops([]);
    }
  }, [form.district]);

  const fetchDistricts = async () => {
    setLoadingDistricts(true);
    setConnectionError(null);
    
    try {
      console.log('Fetching districts from:', `${API_BASE_URL}/api/yield/districts`);
      const response = await fetch(`${API_BASE_URL}/api/yield/districts`);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('Districts received:', data);
      
      if (Array.isArray(data)) {
        setDistricts(data);
      } else {
        console.error('Unexpected data format:', data);
        setConnectionError('Server returned unexpected data format');
      }
      
    } catch (error) {
      console.error('Error fetching districts:', error);
      setConnectionError(error instanceof Error ? error.message : 'Could not connect to backend server');
    } finally {
      setLoadingDistricts(false);
    }
  };

  const fetchCrops = async (district: string) => {
    setLoadingCrops(true);
    setError(null);
    
    try {
      console.log('Fetching crops for district:', district);
      const response = await fetch(`${API_BASE_URL}/api/yield/crops/${encodeURIComponent(district)}`);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('Crops received:', data);
      
      if (Array.isArray(data)) {
        setCrops(data);
      } else {
        console.error('Unexpected crops data format:', data);
        setError('Server returned unexpected crops data format');
      }
    } catch (error) {
      console.error('Error fetching crops:', error);
      setError(error instanceof Error ? error.message : `Failed to load crops for ${district}`);
    } finally {
      setLoadingCrops(false);
    }
  };

  const fetchDistrictInsights = async (district: string) => {
    setLoadingInsights(true);
    
    try {
      console.log('Fetching insights for district:', district);
      // FIXED: Changed from 'district-insights' to 'insights' to match Flask endpoint
      const response = await fetch(`${API_BASE_URL}/api/yield/insights/${encodeURIComponent(district)}`);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('Insights received:', data);
      setInsights(data);
    } catch (error) {
      console.error('Error fetching insights:', error);
    } finally {
      setLoadingInsights(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      console.log('Submitting prediction for:', form);
      const response = await fetch(`${API_BASE_URL}/predict-yield`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(form),
      });

      const data = await response.json();
      console.log('Prediction response:', data);

      if (response.ok && data.success) {
        setResult(data);
        // Fetch insights after successful prediction
        fetchDistrictInsights(form.district);
      } else {
        setError(data.error || 'Prediction failed');
      }
    } catch (error) {
      console.error('Prediction error:', error);
      setError('Failed to connect to server for prediction');
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setForm({ district: "", crop: "" });
    setResult(null);
    setError(null);
    setCrops([]);
    setInsights(null);
  };

  const formatNumber = (num: number | undefined) => {
    if (num === undefined || num === null) return '0';
    return new Intl.NumberFormat('en-IN', { maximumFractionDigits: 2 }).format(num);
  };

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
          Back
        </button>
        
        <div className="mb-6 flex items-center gap-3">
          <div className="rounded-lg bg-primary/10 p-2">
            <Leaf className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h1 className="font-display text-2xl font-bold text-foreground sm:text-3xl">
              Crop Yield Prediction
            </h1>
            <p className="text-muted-foreground">
              Select district and crop to predict yield
            </p>
          </div>
        </div>

        {/* Connection Error */}
        {connectionError && (
          <div className="mb-6 rounded-lg border border-red-200 bg-red-50 p-4 dark:border-red-800 dark:bg-red-950/30">
            <p className="flex items-center gap-2 text-red-700 dark:text-red-400">
              <AlertCircle className="h-4 w-4" />
              {connectionError}
            </p>
          </div>
        )}

        {/* Main Form */}
        {!connectionError && (
          <form onSubmit={handleSubmit} className="mb-8 space-y-6">
            <div className="grid gap-4 sm:grid-cols-2">
              {/* District Selection */}
              <div className="flex flex-col gap-1.5">
                <label className="text-sm font-medium text-foreground">
                  Select District
                </label>
                <select
                  value={form.district}
                  onChange={(e) => {
                    setForm({ district: e.target.value, crop: "" });
                    setResult(null);
                    setError(null);
                    setInsights(null);
                  }}
                  className="rounded-lg border border-input bg-background px-3 py-2.5 text-sm focus:ring-2 focus:ring-primary disabled:opacity-50"
                  required
                  disabled={loadingDistricts}
                >
                  <option value="">Choose a district</option>
                  {districts.map((district) => (
                    <option key={district} value={district}>
                      {district}
                    </option>
                  ))}
                </select>
                {loadingDistricts && (
                  <p className="text-xs text-muted-foreground">Loading districts...</p>
                )}
              </div>

              {/* Crop Selection */}
              <div className="flex flex-col gap-1.5">
                <label className="text-sm font-medium text-foreground">
                  Select Crop
                </label>
                <select
                  value={form.crop}
                  onChange={(e) => {
                    setForm((p) => ({ ...p, crop: e.target.value }));
                    setResult(null);
                    setError(null);
                    setInsights(null);
                  }}
                  className="rounded-lg border border-input bg-background px-3 py-2.5 text-sm focus:ring-2 focus:ring-primary disabled:opacity-50"
                  required
                  disabled={!form.district || loadingCrops}
                >
                  <option value="">Choose a crop</option>
                  {crops.map((crop) => (
                    <option key={crop} value={crop}>
                      {crop}
                    </option>
                  ))}
                </select>
                {loadingCrops && (
                  <p className="text-xs text-muted-foreground">Loading crops...</p>
                )}
              </div>
            </div>

            {/* Error Display */}
            {error && (
              <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700 dark:border-red-800 dark:bg-red-950/30 dark:text-red-400">
                <p className="flex items-center gap-2">
                  <AlertCircle className="h-4 w-4" />
                  {error}
                </p>
              </div>
            )}

            <div className="flex gap-3">
              <button
                type="submit"
                disabled={loading || !form.district || !form.crop}
                className="rounded-lg bg-primary px-6 py-2.5 text-sm font-semibold text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
              >
                {loading ? 'Predicting...' : 'Predict Yield'}
              </button>
              
              {(form.district || form.crop || result) && (
                <button
                  type="button"
                  onClick={resetForm}
                  className="rounded-lg border border-input bg-background px-6 py-2.5 text-sm font-semibold hover:bg-accent"
                >
                  Reset
                </button>
              )}
            </div>
          </form>
        )}

        {/* Prediction Result - Show first */}
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-8 rounded-lg border bg-gradient-to-r from-primary/10 to-primary/5 p-6"
          >
            <h3 className="mb-4 text-lg font-semibold">Prediction Result</h3>
            <div className="grid gap-6 sm:grid-cols-2">
              <div>
                <p className="text-sm text-muted-foreground">Predicted Yield</p>
                <p className="text-4xl font-bold text-primary">
                  {formatNumber(result.predicted_yield)} 
                  <span className="ml-2 text-lg font-normal text-muted-foreground">kg/ha</span>
                </p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Predicted Production</p>
                <p className="text-3xl font-semibold">
                  {formatNumber(result.predicted_production)} 
                  <span className="ml-2 text-base font-normal text-muted-foreground">kg</span>
                </p>
              </div>
            </div>
            <p className="mt-2 text-xs text-muted-foreground">
              Model used: {result.model_used?.replace('_', ' ') || 'Stacking Ensemble'}
            </p>
          </motion.div>
        )}

        {/* District Insights - Show only after prediction */}
        {insights && !loadingInsights && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-4"
          >
            <h2 className="flex items-center gap-2 text-lg font-semibold">
              <Info className="h-5 w-5 text-primary" />
              {insights.district} District Insights
            </h2>
            
            <div className="grid gap-4 sm:grid-cols-3">
              <div className="rounded-lg border bg-card p-4">
                <p className="text-xs text-muted-foreground">Total Area</p>
                <p className="text-xl font-bold">{formatNumber(insights.total_area)} ha</p>
              </div>
              <div className="rounded-lg border bg-card p-4">
                <p className="text-xs text-muted-foreground">Total Production</p>
                <p className="text-xl font-bold">{formatNumber(insights.total_production)} kg</p>
              </div>
              <div className="rounded-lg border bg-card p-4">
                <p className="text-xs text-muted-foreground">Average Yield</p>
                <p className="text-xl font-bold">{formatNumber(insights.avg_yield)} kg/ha</p>
              </div>
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <div className="rounded-lg border bg-card p-4">
                <div className="flex items-center gap-2 text-green-600">
                  <TrendingUp className="h-4 w-4" />
                  <p className="text-sm font-medium">Most Produced Crop</p>
                </div>
                <p className="mt-1 text-lg font-semibold">{insights.top_crop?.name || 'N/A'}</p>
                <p className="text-sm text-muted-foreground">
                  {formatNumber(insights.top_crop?.production)} kg
                </p>
              </div>
              
              <div className="rounded-lg border bg-card p-4">
                <div className="flex items-center gap-2 text-red-600">
                  <TrendingDown className="h-4 w-4" />
                  <p className="text-sm font-medium">Least Produced Crop</p>
                </div>
                <p className="mt-1 text-lg font-semibold">{insights.bottom_crop?.name || 'N/A'}</p>
                <p className="text-sm text-muted-foreground">
                  {formatNumber(insights.bottom_crop?.production)} kg
                </p>
              </div>
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <div className="rounded-lg border bg-card p-4">
                <p className="text-sm font-medium text-muted-foreground">Highest Yield Crop</p>
                <p className="text-lg font-semibold">{insights.best_yield_crop?.name || 'N/A'}</p>
                <p className="text-sm text-green-600">
                  {formatNumber(insights.best_yield_crop?.yield)} kg/ha
                </p>
              </div>
              
              <div className="rounded-lg border bg-card p-4">
                <p className="text-sm font-medium text-muted-foreground">Lowest Yield Crop</p>
                <p className="text-lg font-semibold">{insights.worst_yield_crop?.name || 'N/A'}</p>
                <p className="text-sm text-red-600">
                  {formatNumber(insights.worst_yield_crop?.yield)} kg/ha
                </p>
              </div>
            </div>

            {/* Crop Rankings */}
            {insights.crop_stats && insights.crop_stats.length > 0 && (
              <div className="rounded-lg border bg-card p-4">
                <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold">
                  <BarChart3 className="h-4 w-4 text-primary" />
                  Crop Rankings by Production
                </h3>
                <div className="space-y-2">
                  {insights.crop_stats.map((stat, index) => (
                    <div key={stat.crop} className="flex items-center justify-between border-b pb-2 last:border-0">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-muted-foreground">#{index + 1}</span>
                        <span>{stat.crop}</span>
                      </div>
                      <div className="text-right">
                        <p className="text-sm font-medium">{formatNumber(stat.production)} kg</p>
                        <p className="text-xs text-muted-foreground">{formatNumber(stat.yield)} kg/ha</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        )}

        {loadingInsights && (
          <div className="my-4 text-center text-muted-foreground">
            Loading district insights...
          </div>
        )}
      </motion.div>
    </div>
  );
};

export default YieldPrediction;