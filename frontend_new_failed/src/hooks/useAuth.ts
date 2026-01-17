import { useState, useEffect, useCallback } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";

const API_BASE = "/api/v1";

interface User {
  id: number;
  email: string;
  full_name: string | null;
  phone: string | null;
  is_active: boolean;
  is_admin: boolean;
  vip_level: number;
  referral_code: string;
  created_at: string;
}

async function fetchUser(): Promise<User | null> {
  const token = localStorage.getItem("access_token");
  if (!token) return null;

  try {
    const response = await fetch(`${API_BASE}/auth/me`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });

    if (response.status === 401) {
      localStorage.removeItem("access_token");
      return null;
    }

    if (!response.ok) {
      throw new Error("Failed to fetch user");
    }

    return response.json();
  } catch (error) {
    console.error("Error fetching user:", error);
    return null;
  }
}

export function useAuth() {
  const queryClient = useQueryClient();

  const { data: user, isLoading, refetch } = useQuery<User | null>({
    queryKey: ["user"],
    queryFn: fetchUser,
    retry: false,
    staleTime: 5 * 60 * 1000,
    refetchOnWindowFocus: false,
  });

  const login = async (email: string, password: string) => {
    const formData = new URLSearchParams();
    formData.append("username", email);
    formData.append("password", password);

    const response = await fetch(`${API_BASE}/auth/login`, {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "فشل تسجيل الدخول");
    }

    const data = await response.json();
    localStorage.setItem("access_token", data.access_token);
    await refetch();
    return data;
  };

  const register = async (userData: {
    email: string;
    password: string;
    full_name?: string;
    phone?: string;
    referral_code?: string;
  }) => {
    const response = await fetch(`${API_BASE}/auth/register`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(userData),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "فشل التسجيل");
    }

    return response.json();
  };

  const logout = useCallback(async () => {
    localStorage.removeItem("access_token");
    queryClient.setQueryData(["user"], null);
    queryClient.invalidateQueries({ queryKey: ["user"] });
    window.location.href = "/";
  }, [queryClient]);

  return {
    user,
    isLoading,
    isAuthenticated: !!user,
    isAdmin: user?.is_admin || false,
    login,
    register,
    logout,
    refetch,
  };
}

// API helper function
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

  const response = await fetch(`${API_BASE}${url}`, {
    method,
    headers,
    body: data ? JSON.stringify(data) : undefined,
  });

  return response;
}
