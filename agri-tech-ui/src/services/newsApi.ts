import axios from "axios";

export interface NewsArticle {
  title: string;
  description: string;
  url: string;
  urlToImage: string | null;
  source: { name: string };
  publishedAt: string;
}

// Mock news data for when API is not available
const mockNews: NewsArticle[] = [
  {
    title: "New AI Technology Revolutionizes Crop Disease Detection",
    description: "Researchers develop machine learning models that can detect crop diseases with 95% accuracy using smartphone images, helping farmers take quick action.",
    url: "#",
    urlToImage: null,
    source: { name: "AgriTech Today" },
    publishedAt: new Date().toISOString(),
  },
  {
    title: "Sustainable Farming Practices Increase Yield by 30%",
    description: "A study shows that integrating sustainable farming methods with precision agriculture can significantly boost crop yields while reducing environmental impact.",
    url: "#",
    urlToImage: null,
    source: { name: "Farm Weekly" },
    publishedAt: new Date().toISOString(),
  },
  {
    title: "Smart Irrigation Systems Save 40% Water in Drought Regions",
    description: "IoT-enabled smart irrigation systems are proving to be a game-changer for farmers in drought-prone areas, significantly reducing water consumption.",
    url: "#",
    urlToImage: null,
    source: { name: "Green Agriculture" },
    publishedAt: new Date().toISOString(),
  },
  {
    title: "Government Launches ₹5000 Crore Agricultural AI Initiative",
    description: "The Ministry of Agriculture announced a major initiative to bring artificial intelligence and data-driven solutions to millions of small-scale farmers across the country.",
    url: "#",
    urlToImage: null,
    source: { name: "National Herald" },
    publishedAt: new Date().toISOString(),
  },
  {
    title: "Drone Technology Transforms Precision Agriculture in India",
    description: "Agricultural drones are being deployed across Indian states for crop monitoring, spraying, and yield estimation, reducing labor costs by up to 60%.",
    url: "#",
    urlToImage: null,
    source: { name: "AgriDrone News" },
    publishedAt: new Date().toISOString(),
  },
  {
    title: "Climate-Resilient Crop Varieties Show Promise in Field Trials",
    description: "New crop varieties engineered for climate resilience are showing exceptional results in field trials across multiple agro-climatic zones.",
    url: "#",
    urlToImage: null,
    source: { name: "Crop Science Daily" },
    publishedAt: new Date().toISOString(),
  },
];

export const fetchAgricultureNews = async (): Promise<NewsArticle[]> => {
  const apiKey = import.meta.env.VITE_NEWS_API_KEY;

  if (apiKey) {
    try {
      const response = await axios.get("https://newsapi.org/v2/everything", {
        params: {
          q: "agriculture farming crops",
          sortBy: "publishedAt",
          pageSize: 6,
          language: "en",
          apiKey,
        },
      });
      return response.data.articles;
    } catch {
      return mockNews;
    }
  }

  // Return mock news when no API key
  await new Promise((r) => setTimeout(r, 800));
  return mockNews;
};
