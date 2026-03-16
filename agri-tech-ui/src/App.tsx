import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ThemeProvider } from "@/contexts/ThemeContext";
import "@/i18n";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import ScrollToTop from "@/components/ScrollToTop";
import MouseTrail from "@/components/MouseTrail";
import Home from "@/pages/Home";
import CropRecommendation from "@/pages/CropRecommendation";
import FertilizerRecommendation from "@/pages/FertilizerRecommendation";
import YieldPrediction from "@/pages/YieldPrediction";
import ContactUs from "@/pages/ContactUs";
import NotFound from "@/pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <ThemeProvider>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <MouseTrail />
        <BrowserRouter>
          <div className="flex min-h-screen flex-col">
            <Header />
            <main className="flex-1">
              <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/crop-recommendation" element={<CropRecommendation />} />
                <Route path="/fertilizer-recommendation" element={<FertilizerRecommendation />} />
                <Route path="/yield-prediction" element={<YieldPrediction />} />
                <Route path="/contact" element={<ContactUs />} />
                <Route path="*" element={<NotFound />} />
              </Routes>
            </main>
            <Footer />
            <ScrollToTop />
          </div>
        </BrowserRouter>
      </TooltipProvider>
    </ThemeProvider>
  </QueryClientProvider>
);

export default App;
