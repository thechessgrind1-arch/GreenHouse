export interface PredictionResult {
  verdict: "HEALTHY" | "DAMAGED";
  is_healthy: boolean;
  label: number;
  label_name: string;
  confidence: number;
  probabilities: Record<string, number> | null;
}

export interface Crop {
  name: string;
  species: string;
  icon: string;
}

export const CROPS: Crop[] = [
  { name: "Bell pepper", species: "Capsicum annuum", icon: "pepper" },
  { name: "Tomato", species: "Solanum lycopersicum", icon: "tomato" },
];
