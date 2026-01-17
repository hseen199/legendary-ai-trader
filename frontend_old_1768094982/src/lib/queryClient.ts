import { QueryClient } from '@tanstack/react-query';
import api from '../services/api';

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30 * 1000, // 30 seconds
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

// Helper function for API requests
export async function apiRequest(
  method: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE',
  url: string,
  data?: any
) {
  const config: any = {
    method,
    url,
  };

  if (data) {
    config.data = data;
  }

  const response = await api(config);
  return response;
}

// Default query function for react-query
export const defaultQueryFn = async ({ queryKey }: { queryKey: string[] }) => {
  const [url] = queryKey;
  const response = await api.get(url);
  return response.data;
};

queryClient.setDefaultOptions({
  queries: {
    queryFn: defaultQueryFn as any,
  },
});
