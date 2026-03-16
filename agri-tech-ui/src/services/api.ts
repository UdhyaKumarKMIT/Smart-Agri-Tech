import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:5000";

const api = axios.create({
  baseURL: API_BASE,
  timeout: 15000,
  headers: { "Content-Type": "application/json" },
});

// Crop recommendation types
export interface CropInput {
  nitrogen: number;
  phosphorus: number;
  potassium: number;
  temperature: number;
  humidity: number;
  ph: number;
  rainfall: number;
}

export interface TopPrediction {
  crop: string;
  probability: number;
}

export interface CropResult {
  crop: string;
  confidence: number;
  probabilities?: Record<string, number>;
  top_5_predictions?: TopPrediction[];
  report_path?: string;
  report_id?: string;
  error?: string;
}

// Fertilizer recommendation types
export interface FertilizerInput {
  temperature: number;
  humidity: number;
  moisture: number;
  soilType: string;
  crop: string;
  nitrogen: number;
  potassium: number;
  phosphorus: number;
}

export interface FertilizerResult {
  fertilizer: string;
  explanation: { factor: string; contribution: number }[];
}

// Yield prediction types
export interface YieldInput {
  cropType: string;
  area: number;
  rainfall: number;
  temperature: number;
  fertilizer: number;
}

export interface YieldResult {
  predictedYield: number;
  unit: string;
  factors: { factor: string; contribution: number }[];
}

// API functions - No mock fallbacks
export const predictCrop = async (input: CropInput): Promise<CropResult> => {
  try {
    const response = await api.post("/predict-crop", input);
    const data = response.data;
    
    // Transform backend response to match expected format
    return {
      crop: data.crop,
      confidence: data.confidence,
      probabilities: data.probabilities,
      top_5_predictions: data.top_5_predictions,
      report_path: data.report_path,
      report_id: data.report_id
    };
  } catch (error) {
    if (axios.isAxiosError(error)) {
      if (error.code === 'ECONNREFUSED') {
        throw new Error("Cannot connect to server. Please make sure the backend is running on port 5000.");
      }
      if (error.response) {
        // The request was made and the server responded with a status code
        // that falls out of the range of 2xx
        throw new Error(error.response.data?.error || `Server error: ${error.response.status}`);
      } else if (error.request) {
        // The request was made but no response was received
        throw new Error("No response from server. Please check your connection.");
      }
    }
    // Something happened in setting up the request that triggered an Error
    throw new Error(error instanceof Error ? error.message : "An unknown error occurred");
  }
};

export const predictFertilizer = async (input: FertilizerInput): Promise<FertilizerResult> => {
  try {
    const response = await api.post("/predict-fertilizer", input);
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      if (error.code === 'ECONNREFUSED') {
        throw new Error("Cannot connect to server. Please make sure the backend is running on port 5000.");
      }
      if (error.response) {
        throw new Error(error.response.data?.error || `Server error: ${error.response.status}`);
      } else if (error.request) {
        throw new Error("No response from server. Please check your connection.");
      }
    }
    throw new Error(error instanceof Error ? error.message : "An unknown error occurred");
  }
};

export const predictYield = async (input: YieldInput): Promise<YieldResult> => {
  try {
    const response = await api.post("/predict-yield", input);
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      if (error.code === 'ECONNREFUSED') {
        throw new Error("Cannot connect to server. Please make sure the backend is running on port 5000.");
      }
      if (error.response) {
        throw new Error(error.response.data?.error || `Server error: ${error.response.status}`);
      } else if (error.request) {
        throw new Error("No response from server. Please check your connection.");
      }
    }
    throw new Error(error instanceof Error ? error.message : "An unknown error occurred");
  }
};

// Report endpoints
export const getReport = async (reportId?: string): Promise<Blob> => {
  try {
    const endpoint = reportId ? `/report/${reportId}` : "/report/latest";
    const response = await api.get(endpoint, { 
      responseType: 'blob',
      headers: { 'Accept': 'text/html' }
    });
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      if (error.response?.status === 404) {
        throw new Error("Report not found");
      }
      if (error.code === 'ECONNREFUSED') {
        throw new Error("Cannot connect to server. Please make sure the backend is running on port 5000.");
      }
    }
    throw new Error("Failed to fetch report");
  }
};

export const listReports = async (): Promise<{ total: number; reports: any[] }> => {
  try {
    const response = await api.get("/reports/list");
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      if (error.code === 'ECONNREFUSED') {
        throw new Error("Cannot connect to server. Please make sure the backend is running on port 5000.");
      }
    }
    throw new Error("Failed to fetch reports list");
  }
};

// Helper function to open report in new tab
export const openReport = async (reportPath?: string): Promise<void> => {
  if (!reportPath) {
    throw new Error("No report path available");
  }

  try {
    // If it's a full URL, open it directly
    if (reportPath.startsWith('http')) {
      window.open(reportPath, '_blank');
    } else {
      // If it's a relative path, construct the full URL
      const baseUrl = API_BASE;
      window.open(`${baseUrl}${reportPath}`, '_blank');
    }
  } catch (error) {
    throw new Error("Failed to open report");
  }
};

// Error type for better error handling in components
export class ApiError extends Error {
  constructor(message: string, public statusCode?: number) {
    super(message);
    this.name = 'ApiError';
  }
}

export default api;