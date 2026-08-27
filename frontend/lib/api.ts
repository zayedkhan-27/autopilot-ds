const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type User = {
  id: number;
  name: string;
  email: string;
};

export type Dataset = {
  id: number;
  name: string;
  original_filename: string;
  rows: number | null;
  columns: number | null;
  created_at: string;
};

export type Experiment = {
  id: number;
  dataset_id: number;
  status: string;
  task_type?: string | null;
  target_column?: string | null;
  best_model_name?: string | null;
  health_score?: number | null;
  created_at: string;
};

export function getToken() {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("autopilot_token");
}

export function setToken(token: string) {
  localStorage.setItem("autopilot_token", token);
}

export function logout() {
  localStorage.removeItem("autopilot_token");
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = getToken();
  const headers = new Headers(options.headers || {});
  if (token) headers.set("Authorization", `Bearer ${token}`);

  const res = await fetch(`${API_URL}${path}`, {
    ...options,
    headers,
    cache: "no-store",
  });

  if (!res.ok) {
    let message = `Request failed (${res.status})`;
    try {
      const data = await res.json();
      message = data.detail || data.message || message;
    } catch {}
    throw new Error(message);
  }

  return res.json();
}

export async function register(name: string, email: string, password: string) {
  const result = await request<{ access_token: string; token_type: string }>(
    "/api/auth/register",
    {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({name, email, password}),
    }
  );
  setToken(result.access_token);
  return result;
}

export async function login(email: string, password: string) {
  const result = await request<{ access_token: string; token_type: string }>(
    "/api/auth/login",
    {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({email, password}),
    }
  );
  setToken(result.access_token);
  return result;
}

export const api = {
  me: () => request<User>("/api/me"),
  datasets: () => request<Dataset[]>("/api/datasets"),
  experiments: () => request<Experiment[]>("/api/experiments"),
  experiment: (id: number) =>
    request<{id:number; status:string; error_message?:string|null; result:any}>(
      `/api/experiments/${id}`
    ),

  uploadDataset: async (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<{id:number; name:string; rows:number; columns:number}>(
      "/api/datasets",
      {method: "POST", body: form}
    );
  },

  createExperiment: (payload: {
    dataset_id: number;
    target_column?: string | null;
    task_type?: string | null;
    max_iterations?: number;
  }) =>
    request<{id:number; status:string; result:any}>("/api/experiments", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    }),
};
