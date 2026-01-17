// Shared types and schemas

export interface User {
  id: string;
  email: string;
  full_name: string;
  is_admin: boolean;
  is_active: boolean;
  vip_level: number;
  referral_code: string;
  created_at: string;
}

export interface Balance {
  id: string;
  user_id: string;
  currency: string;
  available: number;
  locked: number;
  total: number;
}

export interface Transaction {
  id: string;
  user_id: string;
  type: "deposit" | "withdrawal" | "trade" | "referral_bonus";
  amount: number;
  currency: string;
  status: "pending" | "completed" | "failed" | "cancelled";
  created_at: string;
  updated_at: string;
}

export interface Trade {
  id: string;
  user_id: string;
  symbol: string;
  side: "buy" | "sell";
  amount: number;
  price: number;
  status: "open" | "closed" | "cancelled";
  profit_loss: number;
  created_at: string;
  closed_at?: string;
}

export interface Position {
  id: string;
  symbol: string;
  side: "long" | "short";
  entry_price: number;
  current_price: number;
  amount: number;
  profit_loss: number;
  profit_loss_percent: number;
}

export interface PortfolioStats {
  total_value: number;
  total_profit: number;
  profit_percent: number;
  nav_price: number;
  units: number;
}

export interface MarketData {
  symbol: string;
  price: number;
  change_24h: number;
  change_percent_24h: number;
  volume_24h: number;
  high_24h: number;
  low_24h: number;
}

export interface Notification {
  id: string;
  title: string;
  message: string;
  type: "info" | "success" | "warning" | "error";
  read: boolean;
  created_at: string;
}

export interface WithdrawalRequest {
  id: string;
  user_id: string;
  amount: number;
  currency: string;
  wallet_address: string;
  network: string;
  status: "pending" | "approved" | "rejected" | "completed";
  created_at: string;
  processed_at?: string;
}

export interface SupportTicket {
  id: string;
  user_id: string;
  subject: string;
  status: "open" | "in_progress" | "resolved" | "closed";
  priority: "low" | "medium" | "high";
  created_at: string;
  updated_at: string;
}

export interface TicketMessage {
  id: string;
  ticket_id: string;
  sender_id: string;
  is_admin: boolean;
  message: string;
  created_at: string;
}
