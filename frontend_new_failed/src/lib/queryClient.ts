import { QueryClient } from "@tanstack/react-query";

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchInterval: false,
      refetchOnWindowFocus: false,
      staleTime: 5 * 60 * 1000,
      retry: 1,
    },
    mutations: {
      retry: false,
    },
  },
});

export async function apiRequest(
  method: string,
  url: string,
  data?: unknown
): Promise<Response> {
  const token = localStorage.getItem("access_token");
  const headers: Record<string, string> = {};

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  if (data) {
    headers["Content-Type"] = "application/json";
  }

  const response = await fetch(url, {
    method,
    headers,
    body: data ? JSON.stringify(data) : undefined,
  });

  return response;
}

export function getQueryFn<T>(options: { on401: "returnNull" | "throw" }) {
  return async ({ queryKey }: { queryKey: string[] }): Promise<T | null> => {
    const token = localStorage.getItem("access_token");
    const headers: Record<string, string> = {};

    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }

    const response = await fetch(queryKey.join("/"), {
      headers,
    });

    if (options.on401 === "returnNull" && response.status === 401) {
      return null;
    }

    if (!response.ok) {
      throw new Error(`${response.status}: ${response.statusText}`);
    }

    return response.json();
  };
}
