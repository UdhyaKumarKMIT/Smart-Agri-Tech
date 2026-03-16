import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from "recharts";

interface ShapPlotProps {
  data: { feature: string; value: number }[];
  title: string;
}

const ShapPlot = ({ data, title }: ShapPlotProps) => {
  const chartData = data.map((d) => ({
    name: d.feature,
    value: Math.round(d.value * 100) / 100,
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
          />
          <ReferenceLine x={0} stroke="hsl(var(--muted-foreground))" />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {chartData.map((entry, index) => (
              <Cell
                key={index}
                fill={entry.value >= 0 ? "hsl(142, 60%, 35%)" : "hsl(0, 70%, 50%)"}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default ShapPlot;
