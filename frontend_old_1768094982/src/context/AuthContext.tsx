import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { authAPI } from '../services/api';

interface User {
  id: number;
  email: string;
  full_name?: string;
  phone?: string;
  status: string;
  is_admin: boolean;
  is_verified: boolean;
  two_factor_enabled: boolean;
  created_at: string;
  last_login?: string;
}

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (data: { email: string; password: string; full_name?: string; phone?: string }) => Promise<void>;
  logout: () => void;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const refreshUser = async () => {
    try {
      const token = localStorage.getItem('token');
      if (!token) {
        setUser(null);
        return;
      }
      const response = await authAPI.getMe();
      setUser(response.data);
    } catch (error) {
      localStorage.removeItem('token');
      setUser(null);
    }
  };

  useEffect(() => {
    const initAuth = async () => {
      await refreshUser();
      setIsLoading(false);
    };
    initAuth();
  }, []);

  const login = async (email: string, password: string) => {
    const response = await authAPI.login(email, password);
    localStorage.setItem('token', response.data.access_token);
    await refreshUser();
  };

  const register = async (data: { email: string; password: string; full_name?: string; phone?: string }) => {
    await authAPI.register(data);
    await login(data.email, data.password);
  };

  const logout = () => {
    localStorage.removeItem('token');
    setUser(null);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        isAuthenticated: !!user,
        login,
        register,
        logout,
        refreshUser,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
