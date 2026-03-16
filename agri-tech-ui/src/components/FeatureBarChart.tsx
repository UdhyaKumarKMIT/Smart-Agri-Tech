import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";

interface FeatureBarChartProps {
  data: { feature: string; importance: number }[] | { factor: string; contribution: number }[];
  title: string;
}

const COLORS = [
  "hsl(142, 60%, 35%)",
  "hsl(85, 50%, 45%)",
  "hsl(42, 90%, 55%)",
  "hsl(200, 60%, 45%)",
  "hsl(25, 80%, 50%)",
  "hsl(280, 50%, 50%)",
  "hsl(340, 60%, 50%)",
];

const FeatureBarChart = ({ data, title }: FeatureBarChartProps) => {
  // Normalize data shape
  const chartData = data.map((d) => ({
    name: "feature" in d ? d.feature : d.factor,
    value: Math.round(("importance" in d ? d.importance : d.contribution) * 100),
  }));

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <h4 className="mb-4 text-sm font-semibold text-foreground">{title}</h4>
      <ResponsiveContainer width="100%" height={250}>
        <BarChart data={chartData} layout="vertical" margin={{ left: 80 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis type="number" tick={{ fontSize: 12, fill: "hsl(var(--muted-foreground))" }} />
          <YAxis
            dataKey="name"
            type="category"
            tick={{ fontSize: 12, fill: "hsl(var(--foreground))" }}
            width={75}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "hsl(var(--card))",
              border: "1px solid hsl(var(--border))",
              borderRadius: "8px",
              color: "hsl(var(--foreground))",
            }}
            formatter={(value: number) => [`${value}%`, "Importance"]}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {chartData.map((_, index) => (
              <Cell key={index} fill={COLORS[index % COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default FeatureBarChart;
